"""Memory-efficient workers for real two-rank CPU behavior tests."""

import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _write_result(result_dir, payload):
    path = Path(result_dir) / f"rank-{int(os.environ['RANK'])}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _filesystem_rendezvous(result_dir, name, timeout_seconds=15):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    directory = Path(result_dir)
    (directory / f".{name}-ready-{rank}").touch()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(
            (directory / f".{name}-ready-{peer}").exists()
            for peer in range(world_size)
        ):
            return
        time.sleep(0.05)
    raise TimeoutError(f"workers did not reach {name} rendezvous")


def run_control_plane(result_dir):
    import distributed.ddp as ddp_module

    real_init_process_group = ddp_module.init_process_group

    def init_with_short_training_timeout(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=2)
        kwargs["init_method"] = f"file://{Path(result_dir) / 'control-pg'}"
        kwargs["rank"] = int(os.environ["RANK"])
        kwargs["world_size"] = int(os.environ["WORLD_SIZE"])
        return real_init_process_group(*args, **kwargs)

    ddp_module.init_process_group = init_with_short_training_timeout
    backend = None
    payload = {"status": "error", "stage": "setup"}
    try:
        # Worker imports can differ by several seconds. Synchronize immediately
        # before init so its deliberately short timeout measures the PG path,
        # not Python import skew.
        _filesystem_rendezvous(result_dir, "control-init")
        backend = ddp_module.DataParallelDistributedBackend(
            SimpleNamespace(
                device="cuda:0",
                distributed_backend="gloo",
                distributed_control_timeout_seconds=8,
            )
        )
        from main import _run_synchronized_action

        _filesystem_rendezvous(result_dir, "control-action")

        def slow_success():
            time.sleep(3)
            return "rank-zero-result"

        success_result = _run_synchronized_action(
            backend,
            "slow master success",
            slow_success,
            master_only=True,
        )
        reduced_mean = backend.reduce_mean(
            torch.tensor(float(backend.rank + 1))
        ).item()
        backend.barrier()

        def slow_error():
            time.sleep(3)
            raise RuntimeError("synthetic slow-action failure")

        error = None
        try:
            _run_synchronized_action(
                backend,
                "slow master error",
                slow_error,
                master_only=True,
            )
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}

        payload = {
            "status": "ok",
            "rank": backend.rank,
            "success_result": success_result,
            "reduced_mean": reduced_mean,
            "error": error,
        }
    except BaseException as exc:
        payload = {
            "status": "error",
            "stage": payload.get("stage"),
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if backend is not None:
            try:
                backend.finalize()
            except BaseException as exc:
                payload = {
                    "status": "error",
                    "stage": "finalize",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        elif dist.is_initialized():
            dist.destroy_process_group()
        _write_result(result_dir, payload)


class ScalarReader:
    num_tokens = 1_000

    def __init__(self, target):
        self.target = target
        self.step = 0

    def sample_batch(self):
        self.step += 1
        return torch.zeros(1), self.target.clone()

    def set_step(self, step):
        self.step = step

    def num_batches(self):
        return 1


def _schedulefree_cfg():
    return SimpleNamespace(
        compile=False,
        device="cpu",
        dtype="float32",
        resume_from=None,
        allow_legacy_checkpoint_resume=False,
        run_identity=None,
        evaluation_protocol={
            "identity": "two-rank-schedulefree",
            "final_and_full": {"mode": "batch_cap", "max_batches": 1},
        },
        expected_world_size=2,
        iterations=0,
        acc_steps=1,
        sequence_length=1,
        batch_size=1,
        opt="sf-adamw",
        scheduler="none",
        moe=False,
        grad_clip=0.0,
        warmup_steps=0,
        eval_interval=1,
        eval_batches=1,
        final_eval_batches=1,
        final_eval_tokens=None,
        full_eval_at=[],
        permanent_ckpt_interval=0,
        latest_ckpt_interval=0,
        weight_average=False,
        wa_horizon=1,
        wa_interval=1,
        wa_use_temp_dir=True,
        wa_dtype="float32",
        exponential_weight_average=False,
        ewa_interval=1,
        ewa_decay=0.99,
        ewa_after_warmup=False,
        log_dynamics=False,
        dynamics_logger_cfg="",
        results_base_folder="",
        wandb=False,
        log_interval=0,
        notify_interval=0,
        log_parameter_norms=False,
        norm_order=2,
        eval_seq_prefix="none",
        plot_router_logits=False,
        gn_inner_iters=1,
        gn_inner_wd=0.0,
        gn_log_inner_steps=False,
        gn_linesearch=False,
        gn_ls_range=[1.0],
        sophia_bs=1,
        precondition_frequency=10,
    )


def run_schedulefree_roundtrip(result_dir):
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        "gloo",
        init_method=f"file://{Path(result_dir) / 'schedulefree-pg'}",
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    payload = {"status": "error", "stage": "setup"}
    try:
        training_base = importlib.import_module("optim.base")
        schedulefree = importlib.import_module("optim.schedulefree")

        class Backend:
            def __init__(self):
                self.rank = rank

            def is_master_process(self):
                return rank == 0

            def get_world_size(self):
                return 2

            def get_raw_model(self, model):
                return model

            def all_gather_object(self, value):
                values = [None, None]
                dist.all_gather_object(values, value)
                return values

            def broadcast_object(self, value, src=0):
                values = [value]
                dist.broadcast_object_list(values, src=src)
                return values[0]

            def barrier(self):
                dist.barrier()

            def reduce_mean(self, value):
                reduced = value.detach().clone()
                dist.all_reduce(reduced)
                return reduced / 2

            def get_context_for_microstep_forward(self, *_args, **_kwargs):
                return nullcontext()

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([27.12765121459961], dtype=torch.float32)
                )

        backend = Backend()
        model = ScalarModel()
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=0.1,
            betas=(0.9, 0.99),
            foreach=False,
        )
        optimizer.state[model.weight]["z"] = torch.tensor(
            [-127.29183197021484], dtype=torch.float32
        )
        optimizer.param_groups[0]["train_mode"] = True
        original_weight = model.weight.detach().clone()
        original_eval = training_base.eval

        def fake_eval(eval_model, *_args, **_kwargs):
            if eval_model.training:
                raise AssertionError("master validation model must be in eval mode")
            return (
                1.0,
                0.0,
                1.0,
                {},
                [],
                {"evaluated_batches": 1, "evaluated_tokens": 1},
            )

        training_base.eval = fake_eval
        try:
            training_base.train(
                model=model,
                opt=optimizer,
                datareaders={
                    "train": ScalarReader(torch.tensor([3.0])),
                    "val": ScalarReader(torch.tensor([3.0])),
                },
                scheduler=None,
                exp_dir=Path(result_dir) / "run",
                distributed_backend=backend,
                cfg=_schedulefree_cfg(),
            )
        finally:
            training_base.eval = original_eval

        gathered = [torch.empty_like(model.weight) for _ in range(2)]
        dist.all_gather(gathered, model.weight.detach())
        payload = {
            "status": "ok",
            "rank": rank,
            "rank0_weight": gathered[0].item(),
            "rank1_weight": gathered[1].item(),
            "original_weight": original_weight.item(),
            "train_mode": optimizer.param_groups[0]["train_mode"],
        }
    except BaseException as exc:
        payload = {
            "status": "error",
            "stage": payload.get("stage"),
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        _write_result(result_dir, payload)


def _validated_results(result_dir, mode):
    results = [
        json.loads((Path(result_dir) / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    if [result.get("status") for result in results] != ["ok", "ok"]:
        raise AssertionError(f"{mode} workers failed: {results!r}")

    if mode == "control":
        if results[0].get("success_result") != "rank-zero-result":
            raise AssertionError(f"rank 0 result was not broadcast: {results!r}")
        if results[1].get("success_result") is not None:
            raise AssertionError(f"non-master unexpectedly returned a result: {results!r}")
        if [result.get("reduced_mean") for result in results] != [1.5, 1.5]:
            raise AssertionError(f"training-group reduction disagreed: {results!r}")
        if [result.get("error", {}).get("type") for result in results] != [
            "RuntimeError",
            "RuntimeError",
        ]:
            raise AssertionError(f"master error was not propagated: {results!r}")
        if not all(
            "synthetic slow-action failure" in result.get("error", {}).get("message", "")
            for result in results
        ):
            raise AssertionError(f"propagated error message disagreed: {results!r}")
        return results

    for result in results:
        if result.get("rank0_weight") != result.get("rank1_weight"):
            raise AssertionError(f"Schedule-Free ranks diverged: {results!r}")
        if result.get("rank0_weight") == result.get("original_weight"):
            raise AssertionError(f"test round trip was accidentally reversible: {results!r}")
        if result.get("train_mode") is not True:
            raise AssertionError(f"optimizer was not restored to train mode: {results!r}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["control", "schedulefree"], required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    # Preload the heavy Torch/project modules once, then fork. This keeps the
    # real two-rank CPU gate viable in memory-constrained containers through
    # copy-on-write while every child still creates its own process groups.
    if args.mode == "control":
        import distributed.ddp  # noqa: F401
        import main as benchmark_main  # noqa: F401
    else:
        import optim.base  # noqa: F401
        import optim.schedulefree  # noqa: F401

    context = multiprocessing.get_context("fork")

    def child_entry(rank):
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": "2",
            }
        )
        if args.mode == "control":
            run_control_plane(args.result_dir)
        else:
            run_schedulefree_roundtrip(args.result_dir)

    workers = [context.Process(target=child_entry, args=(rank,)) for rank in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(45)
    failed = []
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
            failed.append(f"pid {worker.pid} timed out")
        elif worker.exitcode != 0:
            failed.append(f"pid {worker.pid} exited {worker.exitcode}")
    if failed:
        raise RuntimeError("; ".join(failed))
    _validated_results(args.result_dir, args.mode)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()
