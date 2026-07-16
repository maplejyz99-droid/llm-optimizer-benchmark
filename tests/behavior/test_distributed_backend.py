import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
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
            "barrier": 0,
            "broadcast": [],
            "destroy": 0,
            "ddp": [],
            "gather": [],
            "init": 0,
        }

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

        def destroy_process_group():
            calls["destroy"] += 1

        torch_distributed.destroy_process_group = destroy_process_group

        def barrier():
            calls["barrier"] += 1

        torch_distributed.barrier = barrier

        def broadcast_object_list(payload, src=0):
            calls["broadcast"].append((list(payload), src))

        def all_gather_object(output, value):
            calls["gather"].append(value)
            for index in range(len(output)):
                output[index] = value

        torch_distributed.broadcast_object_list = broadcast_object_list
        torch_distributed.all_gather_object = all_gather_object
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

    def test_ddp_barrier_delegates_to_torch_distributed(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            backend.barrier()

        self.assertEqual(calls["barrier"], 1)

    def test_single_barrier_is_noop(self):
        with self._load_distributed_module("single") as (single_module, calls, _):
            backend = single_module.SinlgeNodeBackend(types.SimpleNamespace())
            self.assertIsNone(backend.barrier())

        self.assertEqual(calls["barrier"], 0)

    def test_object_collectives_delegate_to_torch_distributed(self):
        with self._load_distributed_module("ddp") as (ddp_module, calls, _):
            args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
            backend = ddp_module.DataParallelDistributedBackend(args)
            self.assertEqual(backend.broadcast_object("snapshot", src=0), "snapshot")
            self.assertEqual(backend.all_gather_object({"ok": True}), [{"ok": True}] * 2)

        self.assertEqual(calls["broadcast"], [(["snapshot"], 0)])
        self.assertEqual(calls["gather"], [{"ok": True}])

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
