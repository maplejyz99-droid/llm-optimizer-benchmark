import math
import os
from contextlib import contextmanager
from datetime import timedelta

from torch.distributed import (
    ReduceOp,
    all_reduce as distributed_all_reduce,
    all_gather_object as distributed_all_gather_object,
    barrier as distributed_barrier,
    broadcast_object_list,
    destroy_process_group,
    get_world_size,
    init_process_group,
    new_group,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from .backend import DistributedBackend


DEFAULT_CONTROL_TIMEOUT_SECONDS = 24 * 60 * 60


class DataParallelDistributedBackend(DistributedBackend):
    def __init__(self, args):
        try:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "DDP requires integer RANK and LOCAL_RANK environment variables."
            ) from exc
        if self.rank < 0 or self.local_rank < 0:
            raise ValueError("DDP RANK and LOCAL_RANK must be non-negative.")
        if "cuda" not in args.device:
            raise ValueError("DDP backend can not be used on non-CUDA devices.")

        try:
            control_timeout_seconds = int(
                getattr(
                    args,
                    "distributed_control_timeout_seconds",
                    DEFAULT_CONTROL_TIMEOUT_SECONDS,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "distributed_control_timeout_seconds must be a positive integer."
            ) from exc
        if control_timeout_seconds <= 0:
            raise ValueError(
                "distributed_control_timeout_seconds must be a positive integer."
            )

        self._training_backend = str(args.distributed_backend).lower()
        self._default_group_initialized = False
        self._control_group = None
        self._finalized = False
        try:
            init_process_group(backend=args.distributed_backend)
            self._default_group_initialized = True
            get_world_size()
            # Python-object collectives coordinate long master-only operations such as
            # final evaluation. Keeping them off the NCCL training group prevents an
            # otherwise idle rank from holding a pending NCCL collective for the whole
            # evaluation.
            self._control_group = new_group(
                backend="gloo",
                timeout=timedelta(seconds=control_timeout_seconds),
            )
        except BaseException:
            self._cleanup_process_groups(suppress_errors=True)
            raise

    def _cleanup_process_groups(self, *, suppress_errors):
        first_error = None
        if self._control_group is not None:
            try:
                destroy_process_group(self._control_group)
            except BaseException as exc:
                first_error = exc
            finally:
                self._control_group = None

        if self._default_group_initialized:
            try:
                destroy_process_group()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._default_group_initialized = False

        self._finalized = True
        if first_error is not None and not suppress_errors:
            raise first_error

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(
                f"Effective batch size {effective_batch_size} is not divisible "
                f"by the world size {world_size}."
            )
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.rank
        args.data_seed = args.data_seed
        return args

    def transform_model(self, model):
        # Llama carries opt-specific Newton-Muon stats buffers; ordinary DDP train/eval
        # does not need to broadcast them on every forward.
        return DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

    @contextmanager
    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        model.require_backward_grad_sync = (
            microstep_idx == gradient_accumulation_steps - 1
        )
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        return model.module

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f"module.{parameter_name}"]

    def get_world_size(self):
        return get_world_size()

    def barrier(self):
        if self._training_backend == "nccl":
            distributed_barrier(device_ids=[self.local_rank])
            return
        distributed_barrier()

    def broadcast_object(self, value, src=0):
        payload = [value]
        broadcast_object_list(payload, src=src, group=self._control_group)
        return payload[0]

    def all_gather_object(self, value):
        values = [None] * self.get_world_size()
        distributed_all_gather_object(values, value, group=self._control_group)
        return values

    def reduce_mean(self, value):
        reduced = value.detach().clone()
        distributed_all_reduce(reduced, op=ReduceOp.SUM)
        reduced.div_(self.get_world_size())
        return reduced

    def finalize(self):
        if self._finalized:
            return
        self._cleanup_process_groups(suppress_errors=False)
