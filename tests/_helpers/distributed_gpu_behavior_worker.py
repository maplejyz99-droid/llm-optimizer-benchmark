"""Real two-GPU NCCL training-group and Gloo control-group smoke worker."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    import distributed.ddp as ddp_module

    real_init_process_group = ddp_module.init_process_group

    def init_with_short_training_timeout(*init_args, **init_kwargs):
        init_kwargs["timeout"] = timedelta(seconds=2)
        return real_init_process_group(*init_args, **init_kwargs)

    ddp_module.init_process_group = init_with_short_training_timeout
    backend = None
    payload = {"status": "error", "stage": "setup", "rank": rank}
    try:
        backend = ddp_module.DataParallelDistributedBackend(
            SimpleNamespace(
                device="cuda:0",
                distributed_backend="nccl",
                distributed_control_timeout_seconds=15,
            )
        )
        from main import _run_synchronized_action

        def slow_success():
            time.sleep(3)
            return "rank-zero-result"

        success_result = _run_synchronized_action(
            backend,
            "slow GPU master success",
            slow_success,
            master_only=True,
        )
        reduced_mean = backend.reduce_mean(
            torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        ).item()
        backend.barrier()

        def slow_error():
            time.sleep(3)
            raise RuntimeError("synthetic GPU slow-action failure")

        error = None
        try:
            _run_synchronized_action(
                backend,
                "slow GPU master error",
                slow_error,
                master_only=True,
            )
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}

        from optim.schedulefree import AdamWScheduleFree

        parameter = torch.nn.Parameter(
            torch.tensor([27.12765121459961], device=f"cuda:{local_rank}")
        )
        optimizer = AdamWScheduleFree(
            [parameter], lr=0.1, betas=(0.9, 0.99), foreach=False
        )
        optimizer.state[parameter]["z"] = torch.tensor(
            [-127.29183197021484], device=f"cuda:{local_rank}"
        )
        optimizer.param_groups[0]["train_mode"] = True
        original_weight = parameter.detach().item()
        optimizer.eval()
        optimizer.train()
        gathered = [torch.empty_like(parameter) for _ in range(2)]
        dist.all_gather(gathered, parameter.detach())

        if success_result != ("rank-zero-result" if rank == 0 else None):
            raise AssertionError(f"unexpected master result: {success_result!r}")
        if reduced_mean != 1.5:
            raise AssertionError(f"unexpected NCCL reduction: {reduced_mean!r}")
        if error is None or error["type"] != "RuntimeError":
            raise AssertionError(f"master error was not propagated: {error!r}")
        if "synthetic GPU slow-action failure" not in error["message"]:
            raise AssertionError(f"unexpected propagated error: {error!r}")
        if gathered[0].item() != gathered[1].item():
            raise AssertionError("Schedule-Free ranks diverged")
        if gathered[0].item() == original_weight:
            raise AssertionError("Schedule-Free test round trip was reversible")
        if optimizer.param_groups[0]["train_mode"] is not True:
            raise AssertionError("Schedule-Free optimizer did not return to train mode")

        payload = {
            "status": "ok",
            "rank": rank,
            "success_result": success_result,
            "reduced_mean": reduced_mean,
            "error": error,
            "original_weight": original_weight,
            "roundtrip_weight": gathered[0].item(),
            "train_mode": optimizer.param_groups[0]["train_mode"],
        }
    except BaseException as exc:
        payload = {
            "status": "error",
            "rank": rank,
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
                    "rank": rank,
                    "type": type(exc).__name__,
                    "message": f"finalize failed: {exc}",
                }
        elif dist.is_initialized():
            dist.destroy_process_group()
        (result_dir / f"rank-{rank}.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    if payload["status"] != "ok":
        raise RuntimeError(payload)


if __name__ == "__main__":
    main()
