import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

from tests._helpers.behavior_harness import (
    SRC_ROOT,
    isolated_modules,
    patched_modules,
)


class DistributedBackendBehaviorTest(unittest.TestCase):
    @staticmethod
    def _fake_torch_modules(*, world_size=2):
        calls = {
            "all_reduce": [],
            "barrier": [],
            "broadcast": [],
            "destroy": 0,
            "destroy_groups": [],
            "ddp": [],
            "gather": [],
            "init": 0,
            "new_group": [],
        }
        control_group = object()
        calls["control_group"] = control_group

        class CapturingDDP:
            def __init__(self, model, **kwargs):
                self.module = model
                self.kwargs = kwargs
                calls["ddp"].append({"model": model, "kwargs": kwargs})

        torch_module = types.ModuleType("torch")
        torch_distributed = types.ModuleType("torch.distributed")
        def init_process_group(backend):
            calls["init"] += 1

        torch_distributed.init_process_group = init_process_group
        torch_distributed.get_world_size = lambda: world_size

        def new_group(backend, timeout):
            calls["new_group"].append((backend, timeout))
            return control_group

        torch_distributed.new_group = new_group

        def destroy_process_group(group=None):
            calls["destroy"] += 1
            calls["destroy_groups"].append(group)

        torch_distributed.destroy_process_group = destroy_process_group

        def barrier(device_ids=None):
            calls["barrier"].append(device_ids)

        torch_distributed.barrier = barrier

        def broadcast_object_list(payload, src=0, group=None):
            calls["broadcast"].append((list(payload), src, group))

        def all_gather_object(output, value, group=None):
            calls["gather"].append((value, group))
            for index in range(len(output)):
                output[index] = value

        def all_reduce(value, op=None, group=None):
            calls["all_reduce"].append((value, op, group))
            value.value *= world_size

        class ReduceOp:
            SUM = object()

        torch_distributed.broadcast_object_list = broadcast_object_list
        torch_distributed.all_gather_object = all_gather_object
        torch_distributed.all_reduce = all_reduce
        torch_distributed.ReduceOp = ReduceOp
        torch_nn = types.ModuleType("torch.nn")
        torch_nn_parallel = types.ModuleType("torch.nn.parallel")
        torch_nn_parallel.DistributedDataParallel = CapturingDDP
        torch_nn.parallel = torch_nn_parallel
        torch_module.distributed = torch_distributed
        torch_module.nn = torch_nn

        replacements = {
            "torch": torch_module,
            "torch.distributed": torch_distributed,
            "torch.nn": torch_nn,
            "torch.nn.parallel": torch_nn_parallel,
        }
        return replacements, calls, CapturingDDP

    @classmethod
    @contextmanager
    def _load_distributed_module(
        cls, module_name, *, rank="0", local_rank="0", world_size=2
    ):
        replacements, calls, capturing_ddp = cls._fake_torch_modules(
            world_size=world_size
        )
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with patched_modules(replacements), patch.dict(
                os.environ, {"RANK": rank, "LOCAL_RANK": local_rank}, clear=False
            ), isolated_modules("distributed"):
                yield (
                    importlib.import_module(f"distributed.{module_name}"),
                    calls,
                    capturing_ddp,
                )
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass

    def test_ddp_disables_buffer_broadcasts(self):
        with self._load_distributed_module(
            "ddp", local_rank="1"
        ) as (ddp_module, calls, capturing_ddp):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            wrapped = backend.transform_model("raw-model")

        self.assertIsInstance(wrapped, capturing_ddp)
        self.assertEqual(calls["ddp"][0]["model"], "raw-model")
        self.assertEqual(calls["ddp"][0]["kwargs"]["device_ids"], [1])
        self.assertIs(calls["ddp"][0]["kwargs"]["broadcast_buffers"], False)

    def test_ddp_creates_gloo_control_group_with_configured_timeout(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(
                device="cuda:0",
                distributed_backend="nccl",
                distributed_control_timeout_seconds=17,
            )
            ddp_module.DataParallelDistributedBackend(args)

        self.assertEqual(calls["new_group"], [("gloo", timedelta(seconds=17))])

    def test_ddp_nccl_barrier_uses_local_device(self):
        with self._load_distributed_module(
            "ddp", local_rank="1"
        ) as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            backend.barrier()

        self.assertEqual(calls["barrier"], [[1]])

    def test_ddp_gloo_barrier_does_not_pass_cuda_device(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="gloo")
            backend = ddp_module.DataParallelDistributedBackend(args)
            backend.barrier()

        self.assertEqual(calls["barrier"], [None])

    def test_single_barrier_is_noop(self):
        with self._load_distributed_module("single") as (single_module, calls, _):
            backend = single_module.SinlgeNodeBackend(types.SimpleNamespace())
            self.assertIsNone(backend.barrier())

        self.assertEqual(calls["barrier"], [])

    def test_object_collectives_delegate_to_torch_distributed(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            self.assertEqual(backend.broadcast_object("snapshot", src=0), "snapshot")
            self.assertEqual(backend.all_gather_object({"ok": True}), [{"ok": True}] * 2)

        control_group = calls["control_group"]
        self.assertEqual(calls["broadcast"], [(["snapshot"], 0, control_group)])
        self.assertEqual(calls["gather"], [({"ok": True}, control_group)])

    def test_reduce_mean_uses_default_training_group_without_mutating_input(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = float(value)

            def detach(self):
                return self

            def clone(self):
                return FakeTensor(self.value)

            def div_(self, divisor):
                self.value /= divisor
                return self

        with self._load_distributed_module(
            "ddp", world_size=4
        ) as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            original = FakeTensor(3.5)
            reduced = backend.reduce_mean(original)

        self.assertEqual(original.value, 3.5)
        self.assertEqual(reduced.value, 3.5)
        self.assertEqual(len(calls["all_reduce"]), 1)
        _value, op, group = calls["all_reduce"][0]
        self.assertIs(op, ddp_module.ReduceOp.SUM)
        self.assertIsNone(group)

    def test_single_reduce_mean_is_noop(self):
        with self._load_distributed_module("single") as (single_module, _, _):
            backend = single_module.SinlgeNodeBackend(types.SimpleNamespace())
            value = object()
            self.assertIs(backend.reduce_mean(value), value)

    def test_invalid_rank_environment_fails_before_process_group_initialization(self):
        replacements, calls, _ = self._fake_torch_modules()
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with patched_modules(replacements), patch.dict(
                os.environ, {"RANK": "0"}, clear=True
            ), isolated_modules("distributed"):
                ddp_module = importlib.import_module("distributed.ddp")
                args = types.SimpleNamespace(
                    device="cuda:0",
                    distributed_backend="nccl",
                )
                with self.assertRaisesRegex(ValueError, "RANK and LOCAL_RANK"):
                    ddp_module.DataParallelDistributedBackend(args)
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass

        self.assertEqual(calls["init"], 0)
        self.assertEqual(calls["destroy"], 0)

    def test_process_group_is_destroyed_when_constructor_fails_after_init(self):
        replacements, calls, _ = self._fake_torch_modules()

        def fail_world_size():
            raise RuntimeError("synthetic world-size failure")

        replacements["torch.distributed"].get_world_size = fail_world_size
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with patched_modules(replacements), patch.dict(
                os.environ, {"RANK": "0", "LOCAL_RANK": "0"}, clear=True
            ), isolated_modules("distributed"):
                ddp_module = importlib.import_module("distributed.ddp")
                args = types.SimpleNamespace(
                    device="cuda:0",
                    distributed_backend="nccl",
                )
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    ddp_module.DataParallelDistributedBackend(args)
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass

        self.assertEqual(calls["init"], 1)
        self.assertEqual(calls["destroy"], 1)
        self.assertEqual(calls["destroy_groups"], [None])

    def test_control_group_creation_failure_destroys_default_group(self):
        replacements, calls, _ = self._fake_torch_modules()

        def fail_new_group(*_args, **_kwargs):
            raise RuntimeError("synthetic control-group failure")

        replacements["torch.distributed"].new_group = fail_new_group
        sys.path.insert(0, str(SRC_ROOT))
        try:
            with patched_modules(replacements), patch.dict(
                os.environ, {"RANK": "0", "LOCAL_RANK": "0"}, clear=True
            ), isolated_modules("distributed"):
                ddp_module = importlib.import_module("distributed.ddp")
                args = types.SimpleNamespace(
                    device="cuda:0",
                    distributed_backend="nccl",
                )
                with self.assertRaisesRegex(RuntimeError, "control-group"):
                    ddp_module.DataParallelDistributedBackend(args)
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass

        self.assertEqual(calls["destroy_groups"], [None])

    def test_finalize_destroys_control_before_default_and_is_idempotent(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            backend.finalize()
            backend.finalize()

        self.assertEqual(
            calls["destroy_groups"],
            [calls["control_group"], None],
        )

    def test_finalize_still_destroys_default_when_control_cleanup_fails(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            destroyed = []

            def fail_control_cleanup(group=None):
                destroyed.append(group)
                if group is calls["control_group"]:
                    raise RuntimeError("synthetic control cleanup failure")

            ddp_module.destroy_process_group = fail_control_cleanup
            with self.assertRaisesRegex(RuntimeError, "control cleanup"):
                backend.finalize()

        self.assertEqual(destroyed, [calls["control_group"], None])

    def test_invalid_control_timeout_fails_before_group_initialization(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(
                device="cuda:0",
                distributed_backend="nccl",
                distributed_control_timeout_seconds=0,
            )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                ddp_module.DataParallelDistributedBackend(args)

        self.assertEqual(calls["init"], 0)
        self.assertEqual(calls["new_group"], [])

    def test_ddp_offsets_seed_by_global_rank(self):
        with self._load_distributed_module(
            "ddp", rank="5", local_rank="1", world_size=8
        ) as (ddp_module, _, _):
            args = types.SimpleNamespace(
                device="cuda:0",
                distributed_backend="nccl",
                batch_size=8,
                acc_steps=1,
                seed=11,
                data_seed=1337,
            )
            backend = ddp_module.DataParallelDistributedBackend(args)
            adjusted_args = backend.get_adjusted_args_for_process(args)

        self.assertEqual(adjusted_args.seed, 16)
        self.assertEqual(adjusted_args.data_seed, 1337)

    def test_ddp_batch_error_includes_effective_batch_and_world_size(self):
        with self._load_distributed_module(
            "ddp", rank="2", world_size=3
        ) as (ddp_module, _, _):
            args = types.SimpleNamespace(
                device="cuda:0",
                distributed_backend="nccl",
                batch_size=2,
                acc_steps=1,
                seed=0,
                data_seed=1337,
            )
            backend = ddp_module.DataParallelDistributedBackend(args)
            with self.assertRaisesRegex(
                ValueError,
                r"^Effective batch size 2 is not divisible by the world size 3\.$",
            ):
                backend.get_adjusted_args_for_process(args)
