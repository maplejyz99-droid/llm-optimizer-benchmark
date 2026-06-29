import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

from tests._helpers.behavior_harness import SRC_ROOT, patched_modules


class DistributedBackendBehaviorTest(unittest.TestCase):
    def test_ddp_disables_buffer_broadcasts(self):
        calls = []

        class CapturingDDP:
            def __init__(self, model, **kwargs):
                self.module = model
                self.kwargs = kwargs
                calls.append({"model": model, "kwargs": kwargs})

        torch_module = types.ModuleType("torch")
        torch_distributed = types.ModuleType("torch.distributed")
        torch_distributed.init_process_group = lambda backend: None
        torch_distributed.get_world_size = lambda: 2
        torch_distributed.destroy_process_group = lambda: None
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
        for module_name in ["distributed", "distributed.backend", "distributed.ddp"]:
            sys.modules.pop(module_name, None)

        sys.path.insert(0, str(SRC_ROOT))
        try:
            with patched_modules(replacements), patch.dict(
                os.environ, {"RANK": "0", "LOCAL_RANK": "1"}, clear=False
            ):
                ddp_module = importlib.import_module("distributed.ddp")
                args = types.SimpleNamespace(device="cuda:0", distributed_backend="nccl")
                backend = ddp_module.DataParallelDistributedBackend(args)
                wrapped = backend.transform_model("raw-model")
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass
            for module_name in ["distributed", "distributed.backend", "distributed.ddp"]:
                sys.modules.pop(module_name, None)

        self.assertIsInstance(wrapped, CapturingDDP)
        self.assertEqual(calls[0]["model"], "raw-model")
        self.assertEqual(calls[0]["kwargs"]["device_ids"], [1])
        self.assertIs(calls[0]["kwargs"]["broadcast_buffers"], False)
