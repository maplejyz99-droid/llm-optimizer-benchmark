import importlib
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules

try:
    import torch

    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc

try:
    import yaml  # noqa: F401

    YAML_IMPORT_ERROR = None
except ImportError as exc:
    YAML_IMPORT_ERROR = exc


def dependency_skip_reason():
    missing = []
    if TORCH_IMPORT_ERROR is not None:
        missing.append(f"torch unavailable: {TORCH_IMPORT_ERROR}")
    if YAML_IMPORT_ERROR is not None:
        missing.append(f"yaml unavailable: {YAML_IMPORT_ERROR}")
    return "; ".join(missing)


class TinyDataReader:
    num_tokens = 1_000

    def __init__(self, batch):
        self.batch = batch
        self.steps = []

    def sample_batch(self):
        x, y = self.batch
        return x.clone(), y.clone()

    def set_step(self, step):
        self.steps.append(step)

    def num_batches(self):
        return 1


class SingleProcessBackend:
    def is_master_process(self):
        return True

    def get_world_size(self):
        return 1

    def get_raw_model(self, model):
        return model

    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        return nullcontext()


@unittest.skipIf(
    TORCH_IMPORT_ERROR is not None or YAML_IMPORT_ERROR is not None,
    dependency_skip_reason(),
)
class TorchSmokeTest(unittest.TestCase):
    def test_tiny_cpu_training_step_updates_parameters(self):
        class TinyLanguageModel(torch.nn.Module):
            def __init__(self, vocab_size=11, hidden_size=8):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, x, targets=None, get_logits=False, moe=False, **kwargs):
                logits = self.head(self.embedding(x))
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
                return {"loss": loss, "logits": logits, "aux_losses": {}}

        cfg = SimpleNamespace(
            compile=False,
            device="cpu",
            dtype="float32",
            resume_from=None,
            iterations=1,
            acc_steps=1,
            sequence_length=4,
            batch_size=2,
            opt="adamw",
            scheduler="none",
            moe=False,
            grad_clip=1.0,
            warmup_steps=0,
            eval_interval=100,
            eval_batches=1,
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

        model = TinyLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        y = (x + 1) % 11
        datareaders = {
            "train": TinyDataReader((x, y)),
            "val": TinyDataReader((x, y)),
        }
        before = [param.detach().clone() for param in model.parameters()]

        old_path = list(sys.path)
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with isolated_modules("optim"):
                training_base = importlib.import_module("optim.base")
                with tempfile.TemporaryDirectory() as tmpdir:
                    stats = training_base.train(
                        model=model,
                        opt=optimizer,
                        datareaders=datareaders,
                        scheduler=None,
                        exp_dir=Path(tmpdir),
                        distributed_backend=SingleProcessBackend(),
                        cfg=cfg,
                    )
        finally:
            sys.path[:] = old_path

        changed = any(
            not torch.equal(before_param, after_param)
            for before_param, after_param in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertEqual(stats["completed_iterations"], 1)
        self.assertIn("train_loss", stats)
        self.assertIn("val_loss", stats)
        self.assertIn("wall_clock_seconds", stats)


if __name__ == "__main__":
    unittest.main()
