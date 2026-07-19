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


class ScalarReader:
    num_tokens = 1_000

    def __init__(self, target):
        self.target = target
        self.step = 0

    def sample_batch(self):
        self.step += 1
        return torch.zeros(1, dtype=torch.float64), self.target.clone()

    def set_step(self, step):
        self.step = step

    def num_batches(self):
        return 1


class SingleProcessBackend:
    rank = 0

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

    def reduce_mean(self, value):
        return value

    def all_gather_object(self, value):
        return [value]

    def barrier(self):
        return None


@unittest.skipIf(
    TORCH_IMPORT_ERROR is not None or YAML_IMPORT_ERROR is not None,
    dependency_skip_reason(),
)
class ScheduleFreeTrainingModeTest(unittest.TestCase):
    class ScalarModel(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))

        def forward(self, _x, targets=None, **_kwargs):
            loss = (self.weight - targets.mean()).pow(2).sum()
            return {"loss": loss, "aux_losses": {}}

    @staticmethod
    def _cfg(opt_name):
        return SimpleNamespace(
            compile=False,
            device="cpu",
            dtype="float32",
            resume_from=None,
            allow_legacy_checkpoint_resume=False,
            run_identity=None,
            evaluation_protocol={
                "identity": "schedulefree-test-protocol",
                "final_and_full": {"mode": "batch_cap", "max_batches": 1},
            },
            expected_world_size=1,
            iterations=3,
            acc_steps=1,
            sequence_length=1,
            batch_size=1,
            opt=opt_name,
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

    def test_eval_cycles_preserve_correct_training_parameter_point(self):
        old_path = list(sys.path)
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with isolated_modules("optim"):
                training_base = importlib.import_module("optim.base")
                schedulefree = importlib.import_module("optim.schedulefree")

                cases = (
                    (
                        "sf-sgd",
                        schedulefree.SGDScheduleFree,
                        {"lr": 0.2, "momentum": 0.8},
                    ),
                    (
                        "sf-adamw",
                        schedulefree.AdamWScheduleFree,
                        {"lr": 0.2, "betas": (0.8, 0.9), "eps": 1e-8},
                    ),
                )
                for opt_name, optimizer_type, optimizer_kwargs in cases:
                    with self.subTest(opt=opt_name):
                        model = self.ScalarModel()
                        reference = self.ScalarModel()
                        optimizer = optimizer_type(
                            model.parameters(),
                            foreach=False,
                            weight_lr_power=0,
                            **optimizer_kwargs,
                        )
                        reference_optimizer = optimizer_type(
                            reference.parameters(),
                            foreach=False,
                            weight_lr_power=0,
                            **optimizer_kwargs,
                        )
                        target = torch.tensor([3.0], dtype=torch.float64)

                        original_eval = training_base.eval

                        def fake_eval(eval_model, *_args, **_kwargs):
                            self.assertFalse(eval_model.training)
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
                            with tempfile.TemporaryDirectory() as tmpdir:
                                training_base.train(
                                    model=model,
                                    opt=optimizer,
                                    datareaders={
                                        "train": ScalarReader(target),
                                        "val": ScalarReader(target),
                                    },
                                    scheduler=None,
                                    exp_dir=Path(tmpdir),
                                    distributed_backend=SingleProcessBackend(),
                                    cfg=self._cfg(opt_name),
                                )
                                exported_payload = torch.load(
                                    Path(tmpdir) / "model_eval.pt",
                                    weights_only=False,
                                )
                                versioned_export = (
                                    Path(tmpdir)
                                    / "evaluations"
                                    / "schedulefree-test-protocol"
                                    / "model_eval.pt"
                                )
                                self.assertTrue(versioned_export.exists())
                        finally:
                            training_base.eval = original_eval

                        reference_optimizer.train()
                        for _ in range(3):
                            reference_optimizer.eval()
                            reference.eval()
                            reference.train()
                            reference_optimizer.train()
                            reference_optimizer.zero_grad(set_to_none=True)
                            loss = (reference.weight - target).pow(2).sum()
                            loss.backward()
                            reference_optimizer.step()
                        reference_optimizer.eval()
                        reference.eval()
                        expected_eval_weight = reference.weight.detach().clone()
                        reference.train()
                        reference_optimizer.train()

                        self.assertTrue(torch.equal(model.weight, reference.weight))
                        self.assertTrue(
                            torch.equal(
                                exported_payload["model"]["weight"],
                                expected_eval_weight,
                            )
                        )
                        self.assertEqual(
                            exported_payload["parameterization"],
                            "schedulefree-eval",
                        )
                        self.assertEqual(
                            exported_payload["evaluation_protocol"]["identity"],
                            "schedulefree-test-protocol",
                        )
                        self.assertTrue(
                            all(group["train_mode"] for group in optimizer.param_groups)
                        )
        finally:
            sys.path[:] = old_path

if __name__ == "__main__":
    unittest.main()
