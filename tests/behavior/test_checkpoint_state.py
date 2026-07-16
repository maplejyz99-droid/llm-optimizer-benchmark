import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests._helpers.behavior_harness import SRC_ROOT, isolated_modules


def load_optim_modules():
    old_path = list(sys.path)
    sys.path.insert(0, str(SRC_ROOT))
    try:
        with isolated_modules("optim"):
            utils = importlib.import_module("optim.utils")
            averaging = importlib.import_module("optim.weight_averaging")
            muon = importlib.import_module("optim.muon")
            magma = importlib.import_module("optim.magma")
            softeq = importlib.import_module("optim.experimental.softeq_muon")
            newton = importlib.import_module("optim.newton_muon")
            return utils, averaging, muon, magma, softeq, newton
    finally:
        sys.path[:] = old_path


class CheckpointStateBehaviorTest(unittest.TestCase):
    def setUp(self):
        (
            self.utils,
            self.averaging,
            self.muon,
            self.magma,
            self.softeq,
            self.newton,
        ) = load_optim_modules()

    def make_training_objects(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        return model, optimizer, scheduler

    @staticmethod
    def run_linear_step(model, optimizer, scheduler, inputs, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

    def test_versioned_checkpoint_round_trips_metadata_and_averagers(self):
        model, optimizer, scheduler = self.make_training_objects()
        ewa = self.averaging.ExponentialWeightAverager(
            model,
            interval=2,
            decay=0.8,
            warmup=1,
        )
        ewa.step(model)
        ewa.step(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            snapshot_id = self.utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                7,
                ckpt_dir,
                training_state={"train_reader_step": 23},
                run_identity="tiny-sha256",
                averagers={"ewa": ewa},
            )

            raw = torch.load(ckpt_dir / "main.pt", weights_only=False)
            self.assertEqual(raw["checkpoint_kind"], self.utils.CHECKPOINT_KIND)
            self.assertEqual(self.utils.CHECKPOINT_FORMAT_VERSION, 3)
            self.assertEqual(self.utils.WORKER_STATE_FORMAT_VERSION, 2)
            self.assertEqual(raw["format_version"], self.utils.CHECKPOINT_FORMAT_VERSION)
            self.assertIsInstance(raw["snapshot_id"], str)
            self.assertTrue(raw["snapshot_id"])
            self.assertEqual(snapshot_id, raw["snapshot_id"])
            self.assertEqual(raw["world_size"], 1)
            self.assertEqual(raw["optimizer_state_scope"], "global")
            self.assertIsNotNone(raw["optimizer"])

            restored_model, restored_optimizer, restored_scheduler = (
                self.make_training_objects()
            )
            restored_ewa = self.averaging.ExponentialWeightAverager(
                restored_model,
                interval=2,
                decay=0.8,
                warmup=1,
            )
            result = self.utils.load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                ckpt_dir / "main.pt",
                "cpu",
                averagers={"ewa": restored_ewa},
                expected_run_identity="tiny-sha256",
                expected_world_size=1,
                return_metadata=True,
            )

            self.assertEqual(result.iteration, 7)
            self.assertEqual(result.training_state["train_reader_step"], 23)
            self.assertEqual(result.run_identity, "tiny-sha256")
            self.assertEqual(result.world_size, 1)
            self.assertEqual(result.averager_names, ("ewa",))
            self.assertEqual(result.snapshot_id, raw["snapshot_id"])
            self.assertEqual(restored_ewa.count, ewa.count)
            self.assertEqual(restored_ewa.num_saved, ewa.num_saved)
            for expected, actual in zip(
                ewa.module.parameters(), restored_ewa.module.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

    def test_standard_optimizer_resume_matches_the_next_continuous_step(self):
        torch.manual_seed(1234)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        self.run_linear_step(
            model,
            optimizer,
            scheduler,
            torch.randn(4, 3),
            torch.randn(4, 2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            snapshot_id = self.utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                1,
                ckpt_dir,
                training_state={"train_reader_step": 1, "substep": 1},
                run_identity="continuous-equivalence",
            )
            self.utils.save_worker_state(
                ckpt_dir,
                opt=optimizer,
                training_state={"iteration": 1, "train_reader_step": 1, "substep": 1},
                snapshot_id=snapshot_id,
            )

            continuous_inputs = torch.randn(4, 3)
            continuous_targets = torch.randn(4, 2)
            self.run_linear_step(
                model,
                optimizer,
                scheduler,
                continuous_inputs,
                continuous_targets,
            )

            resumed_model = torch.nn.Linear(3, 2)
            resumed_optimizer = torch.optim.SGD(
                resumed_model.parameters(), lr=0.1, momentum=0.9
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1, gamma=0.8
            )
            self.utils.load_checkpoint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                ckpt_dir / "main.pt",
                "cpu",
                expected_run_identity="continuous-equivalence",
                expected_world_size=1,
            )
            self.utils.load_worker_state(
                ckpt_dir,
                opt=resumed_optimizer,
                expected_world_size=1,
                expected_snapshot_id=snapshot_id,
            )
            resumed_inputs = torch.randn(4, 3)
            resumed_targets = torch.randn(4, 2)
            self.assertTrue(torch.equal(continuous_inputs, resumed_inputs))
            self.assertTrue(torch.equal(continuous_targets, resumed_targets))
            self.run_linear_step(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_inputs,
                resumed_targets,
            )

        for expected, actual in zip(model.parameters(), resumed_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())
        expected_momentum = [
            state["momentum_buffer"] for state in optimizer.state.values()
        ]
        actual_momentum = [
            state["momentum_buffer"] for state in resumed_optimizer.state.values()
        ]
        for expected, actual in zip(expected_momentum, actual_momentum):
            self.assertTrue(torch.equal(expected, actual))

    def test_worker_state_from_different_snapshot_is_rejected_before_rng_restore(self):
        model, optimizer, scheduler = self.make_training_objects()
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first_path = Path(first_dir)
            second_path = Path(second_dir)
            first_snapshot_id = self.utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                3,
                first_path,
                run_identity="snapshot-binding",
            )
            torch.manual_seed(101)
            self.utils.save_worker_state(
                first_path,
                opt=optimizer,
                training_state={"iteration": 3},
                snapshot_id=first_snapshot_id,
            )
            first_worker = torch.load(
                first_path / "worker_0.pt",
                weights_only=False,
            )
            self.assertEqual(first_worker["snapshot_id"], first_snapshot_id)
            self.assertEqual(
                first_worker["format_version"],
                self.utils.WORKER_STATE_FORMAT_VERSION,
            )

            second_snapshot_id = self.utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                3,
                second_path,
                run_identity="snapshot-binding",
            )
            self.assertNotEqual(first_snapshot_id, second_snapshot_id)
            torch.manual_seed(202)
            self.utils.save_worker_state(
                second_path,
                opt=optimizer,
                training_state={"iteration": 3},
                snapshot_id=second_snapshot_id,
            )
            shutil.copyfile(
                second_path / "worker_0.pt",
                first_path / "worker_0.pt",
            )

            restored_model, restored_optimizer, restored_scheduler = (
                self.make_training_objects()
            )
            checkpoint = self.utils.load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                first_path / "main.pt",
                "cpu",
                expected_run_identity="snapshot-binding",
                expected_world_size=1,
                return_metadata=True,
            )
            rng_before_restore = torch.random.get_rng_state().clone()
            with self.assertRaisesRegex(
                self.utils.CheckpointCompatibilityError,
                "different snapshot",
            ):
                self.utils.load_worker_state(
                    first_path,
                    opt=restored_optimizer,
                    expected_world_size=1,
                    expected_snapshot_id=checkpoint.snapshot_id,
                )
            self.assertTrue(
                torch.equal(rng_before_restore, torch.random.get_rng_state())
            )

    def test_existing_checkpoint_survives_failed_atomic_save(self):
        model, optimizer, scheduler = self.make_training_objects()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            ckpt_dir.mkdir(exist_ok=True)
            target = ckpt_dir / "main.pt"
            target.write_bytes(b"last-known-good")

            def fail_after_partial_write(_payload, file_obj):
                file_obj.write(b"partial")
                raise RuntimeError("simulated write failure")

            with mock.patch.object(
                self.utils.torch,
                "save",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    self.utils.save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        1,
                        ckpt_dir,
                    )

            self.assertEqual(target.read_bytes(), b"last-known-good")
            self.assertEqual(list(ckpt_dir.glob(".main.pt.*.tmp")), [])

    def test_legacy_and_incompatible_checkpoints_are_rejected_by_default(self):
        model, optimizer, scheduler = self.make_training_objects()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "itr": 4,
                },
                path,
            )

            with self.assertRaisesRegex(
                self.utils.CheckpointCompatibilityError,
                "legacy",
            ):
                self.utils.load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    path,
                    "cpu",
                )

            self.assertEqual(
                self.utils.load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    path,
                    "cpu",
                    allow_legacy=True,
                ),
                4,
            )

    def test_world_size_and_run_identity_mismatches_are_rejected(self):
        model, optimizer, scheduler = self.make_training_objects()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            self.utils.save_checkpoint(
                model,
                optimizer,
                scheduler,
                2,
                ckpt_dir,
                run_identity={"run_id": "expected"},
            )

            with self.assertRaisesRegex(
                self.utils.CheckpointCompatibilityError,
                "world size",
            ):
                self.utils.load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    ckpt_dir / "main.pt",
                    "cpu",
                    expected_world_size=2,
                )

            with self.assertRaisesRegex(
                self.utils.CheckpointCompatibilityError,
                "run identity",
            ):
                self.utils.load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    ckpt_dir / "main.pt",
                    "cpu",
                    expected_run_identity={"run_id": "other"},
                )

    def test_owner_sharded_worker_state_round_trips_only_for_multiple_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "1"},
            clear=False,
        ):
            parameter = torch.nn.Parameter(torch.ones(2, 2))
            optimizer = self.softeq.SoftEqK2000Muon(
                [parameter],
                adamw_lr=0.01,
            )
            optimizer.global_step = 17
            optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter,
                3.0,
            )
            ckpt_dir = Path(tmpdir)
            snapshot_id = self.utils.new_checkpoint_snapshot_id()
            self.utils.save_worker_state(
                ckpt_dir,
                opt=optimizer,
                snapshot_id=snapshot_id,
            )
            raw = torch.load(ckpt_dir / "worker_1.pt", weights_only=False)
            self.assertIn("optimizer", raw)
            self.assertEqual(raw["rank"], 1)
            self.assertEqual(raw["world_size"], 2)

            optimizer.global_step = 0
            optimizer.state[parameter].pop("momentum_buffer")
            self.utils.load_worker_state(
                ckpt_dir,
                opt=optimizer,
                expected_world_size=2,
                expected_snapshot_id=snapshot_id,
            )
            self.assertEqual(optimizer.global_step, 17)
            self.assertTrue(
                torch.equal(
                    optimizer.state[parameter]["momentum_buffer"],
                    torch.full_like(parameter, 3.0),
                )
            )

        ordinary = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            self.utils.save_worker_state(
                ckpt_dir,
                opt=ordinary,
                snapshot_id=self.utils.new_checkpoint_snapshot_id(),
            )
            raw = torch.load(ckpt_dir / "worker_0.pt", weights_only=False)
            self.assertNotIn("optimizer", raw)

    def test_owner_sharded_optimizer_is_restored_only_from_worker_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "0"},
            clear=False,
        ):
            ckpt_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 2, bias=False)
            optimizer = self.softeq.SoftEqK2000Muon(
                [model.weight],
                adamw_lr=0.01,
            )
            optimizer.global_step = 17
            optimizer.state[model.weight]["momentum_buffer"] = torch.full_like(
                model.weight,
                3.0,
            )
            snapshot_id = self.utils.save_checkpoint(
                model,
                optimizer,
                None,
                4,
                ckpt_dir,
                world_size=2,
            )
            self.utils.save_worker_state(
                ckpt_dir,
                opt=optimizer,
                rank=0,
                world_size=2,
                snapshot_id=snapshot_id,
            )

            raw_main = torch.load(ckpt_dir / "main.pt", weights_only=False)
            self.assertEqual(
                raw_main["optimizer_state_scope"],
                "rank-local-workers",
            )
            self.assertIsNone(raw_main["optimizer"])

            restored_model = torch.nn.Linear(2, 2, bias=False)
            restored_optimizer = self.softeq.SoftEqK2000Muon(
                [restored_model.weight],
                adamw_lr=0.01,
            )
            self.utils.load_checkpoint(
                restored_model,
                restored_optimizer,
                None,
                ckpt_dir / "main.pt",
                "cpu",
                expected_world_size=2,
            )
            self.assertEqual(restored_optimizer.global_step, 0)
            self.assertNotIn(
                "momentum_buffer",
                restored_optimizer.state[restored_model.weight],
            )

            self.utils.load_worker_state(
                ckpt_dir,
                opt=restored_optimizer,
                rank=0,
                expected_world_size=2,
                expected_snapshot_id=snapshot_id,
            )
            self.assertEqual(restored_optimizer.global_step, 17)
            self.assertTrue(
                torch.equal(
                    restored_optimizer.state[restored_model.weight][
                        "momentum_buffer"
                    ],
                    torch.full_like(restored_model.weight, 3.0),
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"WORLD_SIZE": "1", "RANK": "0"},
            clear=False,
        ):
            single_rank_parameter = torch.nn.Parameter(torch.ones(2, 2))
            single_rank_optimizer = self.softeq.SoftEqK2000Muon(
                [single_rank_parameter],
                adamw_lr=0.01,
            )
            ckpt_dir = Path(tmpdir)
            self.utils.save_worker_state(
                ckpt_dir,
                opt=single_rank_optimizer,
                snapshot_id=self.utils.new_checkpoint_snapshot_id(),
            )
            raw = torch.load(ckpt_dir / "worker_0.pt", weights_only=False)
            self.assertFalse(raw["requires_rank_local_optimizer_state"])
            self.assertNotIn("optimizer", raw)

    def test_muon_family_marks_owner_sharded_state_requirement(self):
        self.assertTrue(self.muon.Muon.requires_rank_local_state)
        self.assertTrue(self.magma.MagmaMuon.requires_rank_local_state)
        self.assertTrue(self.softeq.SoftEqK2000Muon.requires_rank_local_state)

    def test_weight_averagers_round_trip_partial_accumulators(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            wa = self.averaging.WeightAverager(
                model,
                horizon=4,
                interval=1,
                save_dir=first_dir,
            )
            for value in (2.0, 4.0, 8.0):
                model.weight.data.fill_(value)
                wa.step(model)

            restored_wa = self.averaging.WeightAverager(
                model,
                horizon=4,
                interval=1,
                save_dir=second_dir,
            )
            restored_wa.load_state_dict(wa.state_dict())
            self.assertEqual(restored_wa.count, wa.count)
            self.assertEqual(restored_wa.num_saved, wa.num_saved)
            self.assertTrue(
                torch.equal(restored_wa.module.weight, wa.module.weight)
            )

            incompatible = self.averaging.WeightAverager(
                model,
                horizon=8,
                interval=1,
                save_dir=second_dir,
            )
            with self.assertRaisesRegex(ValueError, "horizon"):
                incompatible.load_state_dict(wa.state_dict())

        ewa = self.averaging.ExponentialWeightAverager(
            model,
            interval=2,
            decay=0.75,
            warmup=1,
        )
        for value in (1.0, 3.0, 9.0, 12.0):
            model.weight.data.fill_(value)
            ewa.step(model)
        restored_ewa = self.averaging.ExponentialWeightAverager(
            model,
            interval=2,
            decay=0.75,
            warmup=1,
        )
        restored_ewa.load_state_dict(ewa.state_dict())
        self.assertEqual(restored_ewa.count, ewa.count)
        self.assertEqual(restored_ewa.num_saved, ewa.num_saved)
        self.assertTrue(torch.equal(restored_ewa.module.weight, ewa.module.weight))

    def test_weight_average_resume_requires_completed_history_files(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as other_dir:
            wa = self.averaging.WeightAverager(
                model,
                horizon=2,
                interval=1,
                save_dir=source_dir,
            )
            wa.step(model)
            wa.step(model)
            state = wa.state_dict()
            self.assertEqual(state["saved_counts"], [2])

            same_history = self.averaging.WeightAverager(
                model,
                horizon=2,
                interval=1,
                save_dir=source_dir,
            )
            same_history.load_state_dict(state)
            self.assertEqual(same_history.num_saved, 1)

            missing_history = self.averaging.WeightAverager(
                model,
                horizon=2,
                interval=1,
                save_dir=other_dir,
            )
            with self.assertRaisesRegex(ValueError, "history.*missing"):
                missing_history.load_state_dict(state)

    def test_newton_load_state_dict_does_not_mutate_input(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = self.newton.NewtonMuon([parameter], adamw_lr=0.01)
        optimizer.global_step = 11
        state = optimizer.state_dict()

        first = self.newton.NewtonMuon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            adamw_lr=0.01,
        )
        second = self.newton.NewtonMuon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            adamw_lr=0.01,
        )
        first.load_state_dict(state)
        second.load_state_dict(state)

        self.assertIn("newton_muon_global_step", state)
        self.assertEqual(first.global_step, 11)
        self.assertEqual(second.global_step, 11)


if __name__ == "__main__":
    unittest.main()
