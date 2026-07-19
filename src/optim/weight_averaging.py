import tempfile
from copy import deepcopy
from pathlib import Path

import torch

try:
    import wandb
except ImportError:
    wandb = None

from .utils import atomic_torch_save, eval


AVERAGER_STATE_VERSION = 1


class WeightAverager:
    def __init__(
        self,
        model,
        horizon=100,
        interval=1,
        save_dir=None,
        device=None,
        dtype=torch.float32,
        count=0,
    ):
        super().__init__()
        self.device = device  # Where to keep avg model
        self.dtype = dtype  # Precision for accumulation (>= float32)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.module = deepcopy(model).to(dtype=self.dtype, device=device)

        assert horizon % interval == 0, "Interval should divide period"
        self.interval = interval
        self.horizon = horizon
        self.period = horizon // interval
        if save_dir is None:
            # Keep in tempdir
            self._tempdir = tempfile.TemporaryDirectory()
            self.save_dir = Path(self._tempdir.name)
        else:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.count = count
        # check if there are any checkpoints saved in the directory and set
        # num_saved to number of checkpoints with name <= count
        self.num_saved = len(_saved_average_counts(self.save_dir, count))

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state
        if self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                rate = 1 / ((self.count % self.horizon) // self.interval + 1)
                avg.copy_(torch.lerp(avg, curr, rate))

        self.count += 1

        if self.count % self.horizon == 0 and is_master_rank:
            atomic_torch_save(
                self.module.to().state_dict(),
                self.save_dir / f"{self.count}.pt",
            )
            self.num_saved += 1

    def publishes_on_next_step(self):
        """Return whether the next step reaches a persisted horizon."""
        return (self.count + 1) % self.horizon == 0

    def state_dict(self):
        saved_counts = _saved_average_counts(self.save_dir, self.count)
        if len(saved_counts) != self.num_saved:
            raise RuntimeError(
                "Weight-average checkpoint files do not match num_saved: "
                f"files={saved_counts}, num_saved={self.num_saved}."
            )
        return {
            "averager_kind": "weight-average",
            "format_version": AVERAGER_STATE_VERSION,
            "module": deepcopy(self.module.state_dict()),
            "count": self.count,
            "num_saved": self.num_saved,
            "saved_counts": saved_counts,
            "horizon": self.horizon,
            "interval": self.interval,
        }

    def load_state_dict(self, state_dict):
        state = _validate_averager_state(
            state_dict,
            kind="weight-average",
            expected_config={
                "horizon": self.horizon,
                "interval": self.interval,
            },
        )
        map_and_load_state_dict(self.module, state["module"])
        count = _nonnegative_int(state["count"], "count")
        num_saved = _nonnegative_int(state["num_saved"], "num_saved")
        saved_counts = [_nonnegative_int(value, "saved count") for value in state["saved_counts"]]
        if saved_counts != sorted(set(saved_counts)) or len(saved_counts) != num_saved:
            raise ValueError(
                "averager saved_counts must be unique, sorted, and match num_saved."
            )
        available_counts = _saved_average_counts(self.save_dir, count)
        if available_counts != saved_counts:
            raise ValueError(
                "weight-average history required for exact resume is missing or "
                f"incompatible: saved={saved_counts}, available={available_counts}."
            )
        self.count = count
        self.num_saved = num_saved

    def get_latest_like(self, model):
        # Return model for latest completed period
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        # Assumes that we saved at a specific iteration, will fail otherwise
        count = self.count - self.count % self.horizon
        latest_path = self.save_dir / f"{count}.pt"
        map_and_load_state_dict(new_model, torch.load(latest_path))

        return new_model

    def sweep_horizon_like(self, model, max_num=None):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)
        avg_state = deepcopy(self.module.state_dict())
        if max_num is None:
            max_num = self.num_saved
        # Assumes all points exist
        for n in range(min(self.num_saved, max_num)):
            # Load state from the corresponding checkpoint
            count = self.count - self.count % self.horizon - n * self.horizon
            state = torch.load(self.save_dir / f"{count}.pt")

            # Update average state
            for key, avg in avg_state.items():
                new = state[key].to(dtype=avg.dtype, device=avg.device)
                rate = 1 / (n + 1)
                avg.copy_(torch.lerp(avg, new, rate))

            # Set new_model state and yield it
            map_and_load_state_dict(new_model, avg_state)
            yield ((n + 1) * self.horizon, new_model)


def map_and_load_state_dict(model, state_dict):
    for key, m_val in model.state_dict().items():
        for alias in (
            f"_orig_mod.{key}",
            f"_orig_mod.module.{key}",
        ):  # handle compiled / nested model
            if key not in state_dict and alias in state_dict:
                key = alias
                break
        s_val = state_dict[key]
        m_val.copy_(s_val.to(device=m_val.device, dtype=m_val.dtype))


def _nonnegative_int(value, name):
    value = int(value)
    if value < 0:
        raise ValueError(f"averager {name} must be non-negative, got {value}.")
    return value


def _saved_average_counts(save_dir, maximum_count):
    counts = []
    for path in Path(save_dir).glob("*.pt"):
        try:
            count = int(path.stem)
        except ValueError:
            continue
        if count <= int(maximum_count):
            counts.append(count)
    return sorted(set(counts))


def _validate_averager_state(state_dict, *, kind, expected_config):
    if not isinstance(state_dict, dict):
        raise TypeError("averager state must be a dictionary.")
    if state_dict.get("averager_kind") != kind:
        raise ValueError(
            f"averager kind mismatch: expected {kind!r}, "
            f"got {state_dict.get('averager_kind')!r}."
        )
    if state_dict.get("format_version") != AVERAGER_STATE_VERSION:
        raise ValueError(
            "unsupported averager state format version: "
            f"{state_dict.get('format_version')!r}."
        )
    required = {"module", "count", "num_saved", *expected_config}
    if kind == "weight-average":
        required.add("saved_counts")
    missing = sorted(required.difference(state_dict))
    if missing:
        raise ValueError(f"averager state is missing: {', '.join(missing)}.")
    for name, expected in expected_config.items():
        if state_dict[name] != expected:
            raise ValueError(
                f"averager {name} mismatch: saved={state_dict[name]!r}, "
                f"expected={expected!r}."
            )
    return state_dict


def _get_eval_limits(curr_iter, val_reader, cfg, full_eval):
    if curr_iter != cfg.iterations and not full_eval:
        return cfg.eval_batches, None

    available_batches = val_reader.num_batches()
    final_eval_batches = getattr(cfg, "final_eval_batches", None)
    max_num_batches = (
        available_batches
        if final_eval_batches is None
        else min(available_batches, final_eval_batches)
    )
    return max_num_batches, getattr(cfg, "final_eval_tokens", None)


def _evaluation_count_logs(prefix, counts):
    return {
        f"{prefix}/evaluated_batches": counts["evaluated_batches"],
        f"{prefix}/evaluated_tokens": counts["evaluated_tokens"],
    }


def eval_wa(
    curr_iter,
    model,
    weight_averager,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    if weight_averager.num_saved == 0:
        return
    if not cfg.wa_sweep_horizon:
        max_num_batches, max_num_tokens = _get_eval_limits(
            curr_iter, val_reader, cfg, full_eval
        )
        val_reader.set_step(0)
        val_acc, val_loss, val_perplexity, _, _, counts = eval(
            weight_averager.get_latest_like(model).eval(),
            val_reader,
            cfg.device,
            max_num_batches=max_num_batches,
            max_num_tokens=max_num_tokens,
            ctx=type_ctx,
            moe=cfg.moe,
            get_router_logits=False,  # we dont track router logits for WA
            cfg=cfg,
            return_counts=True,
        )

        if cfg.wandb:
            if curr_iter == cfg.iterations or full_eval:
                logs = {
                    "iter": curr_iter,
                    "final-val/loss_wa": val_loss,
                    "final-val/perplexity_wa": val_perplexity,
                    "final-val/acc_wa": val_acc,
                    **_evaluation_count_logs("final-val/wa", counts),
                }
            else:
                logs = {
                    "iter": curr_iter,
                    "val/loss_wa": val_loss,
                    "val/perplexity_wa": val_perplexity,
                    "val/acc_wa": val_acc,
                    **_evaluation_count_logs("val/wa", counts),
                }
            wandb.log(logs)
        print(
            f">WA Eval: Iter={curr_iter} "
            f"val_loss={val_loss:.3f} "
            f"val_pp={val_perplexity:.3f} "
            f"val_acc={val_acc:3f}"
        )
        return counts
    else:
        losses = []
        counts = None
        for horizon, avg_model in weight_averager.sweep_horizon_like(
            model, cfg.max_num_wa_sweeps
        ):
            avg_model.eval()
            val_reader.set_step(0)
            max_num_batches, max_num_tokens = _get_eval_limits(
                curr_iter, val_reader, cfg, full_eval
            )
            _, val_loss, _, _, _, counts = eval(
                avg_model,
                val_reader,
                cfg.device,
                max_num_batches=max_num_batches,
                max_num_tokens=max_num_tokens,
                ctx=type_ctx,
                moe=cfg.moe,
                get_router_logits=False,
                cfg=cfg,
                return_counts=True,
            )

            losses.append((val_loss, horizon))
        if len(losses) == 0:  # in case of none saved yet
            return
        best_loss, best_horizon = sorted(losses)[0]

        print(f"WA Eval: {[(h, f'{l:0.3e}') for (l,h) in losses]}")

        if cfg.wandb:
            if curr_iter == cfg.iterations or full_eval:
                logs = {
                    "iter": curr_iter,
                    "final-val/loss_wa": losses[0][0],
                    "final-val/perplexity_wa": 2.71828 ** losses[0][0],
                    "final-val/best_loss_wa": best_loss,
                    "final-val/best_perplexity_wa": 2.71828**best_loss,
                    **_evaluation_count_logs("final-val/wa", counts),
                }
            else:
                logs = {
                    "iter": curr_iter,
                    "val/loss_wa": losses[0][0],
                    "val/perplexity_wa": 2.71828 ** losses[0][0],
                    "val/best_loss_wa": best_loss,
                    "val/best_perplexity_wa": 2.71828**best_loss,
                    "wa_best_horizon": best_horizon,
                    **_evaluation_count_logs("val/wa", counts),
                }
            wandb.log(logs)
        return counts


class ExponentialWeightAverager:
    def __init__(
        self,
        model,
        interval=1,
        decay=0.95,
        device=None,
        warmup=0,
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = device  # Where to keep avg model
        self.dtype = dtype  # Precision for accumulation (>= float32)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.module = deepcopy(model).to(dtype=self.dtype, device=device)

        self.interval = interval
        self.decay = decay
        self.num_saved = 0
        self.warmup = warmup
        self.count = 0

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state

        if self.count < self.warmup:
            self.count += 1
            return

        if self.count == self.warmup:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                avg.copy_(curr)

        elif self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                avg.copy_(torch.lerp(avg, curr, 1 - self.decay))
            self.num_saved += 1

        self.count += 1

        # if self.count % self.horizon == 0 and is_master_rank:
        #     torch.save(
        #         self.module.to(dtype=torch.bfloat16).state_dict(),
        #         self.save_dir / f"{self.count}.pt",
        #     )
        #     self.num_saved += 1

    def get_latest_like(self, model):
        # Return model for latest completed period
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        map_and_load_state_dict(new_model, self.module.state_dict())

        return new_model

    def state_dict(self):
        return {
            "averager_kind": "exponential-weight-average",
            "format_version": AVERAGER_STATE_VERSION,
            "module": deepcopy(self.module.state_dict()),
            "count": self.count,
            "num_saved": self.num_saved,
            "interval": self.interval,
            "decay": self.decay,
            "warmup": self.warmup,
        }

    def load_state_dict(self, state_dict):
        state = _validate_averager_state(
            state_dict,
            kind="exponential-weight-average",
            expected_config={
                "interval": self.interval,
                "decay": self.decay,
                "warmup": self.warmup,
            },
        )
        map_and_load_state_dict(self.module, state["module"])
        self.count = _nonnegative_int(state["count"], "count")
        self.num_saved = _nonnegative_int(state["num_saved"], "num_saved")


def eval_ewa(
    curr_iter,
    model,
    ewa,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    max_num_batches, max_num_tokens = _get_eval_limits(
        curr_iter, val_reader, cfg, full_eval
    )
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, _, _, counts = eval(
        ewa.get_latest_like(model).eval(),
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        max_num_tokens=max_num_tokens,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=False,  # we dont track router logits for EWA
        cfg=cfg,
        return_counts=True,
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss_ewa": val_loss,
                "final-val/perplexity_ewa": val_perplexity,
                "final-val/acc_ewa": val_acc,
                **_evaluation_count_logs("final-val/ewa", counts),
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss_ewa": val_loss,
                "val/perplexity_ewa": val_perplexity,
                "val/acc_ewa": val_acc,
                **_evaluation_count_logs("val/ewa", counts),
            }
        if cfg.moe and cfg.plot_router_logits:
            pass
        wandb.log(logs)
    print(
        f">EWA Eval: Iter={curr_iter} "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )
    return counts
