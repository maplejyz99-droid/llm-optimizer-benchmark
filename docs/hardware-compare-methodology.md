# Hardware Compare Methodology

## Batch Semantics

Every row must distinguish requested CLI batch fields from actual per-GPU fields. The global effective batch used for comparison is:

```text
effective_batch_global = microbs_per_gpu_actual * acc_steps_actual * world_size
tokens_per_update_global = effective_batch_global * sequence_length
```

Rows where reported `effbs` disagrees with the formula are retained but flagged for review.

## Comparison Classes

- `exact_success`: exact same model, sequence length, optimizer, effective batch, per-GPU micro batch, accumulation, world size, steps, seed, dtype, scheduler, warmup, eval/checkpoint/W&B state, optimizer hyperparams, and runtime.
- `same_effbs_fallback`: same global batch and training target, but different micro batch or accumulation because one hardware cannot run the other hardware's exact config.
- `hardware_best`: best observed stable config for max micro batch, best tokens/s, or best wall-clock for the target effective batch.

Short probes are system-performance evidence only. They do not prove final convergence quality.
