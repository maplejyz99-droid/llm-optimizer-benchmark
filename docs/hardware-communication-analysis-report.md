# Hardware Communication Analysis Report

## Observed DDP Overhead

These rows are estimated from single-card and DDP step time. They include communication, synchronization, launch overhead, load imbalance, memory pressure, and accumulation effects.

| hardware | case       | model | optimizer         | world_size | observed_ddp_overhead_s | observed_parallel_efficiency |
| -------- | ---------- | ----- | ----------------- | ---------- | ----------------------- | ---------------------------- |
| 5090     | 124m-large | 124m  | adamw             | 2          | 0.027921849750000116    | 0.9498146397420082           |
| 5090     | 124m-large | 124m  | muon              | 2          | 0.04152050829999987     | 0.929948100573323            |
| 5090     | 124m-large | 124m  | softeq-k2000-muon | 2          | 0.03402966837499999     | 0.9429359942846496           |
| 5090     | 124m-large | 124m  | sophiag           | 2          | 0.03929913585000011     | 0.9313702014474295           |
| 5090     | 124m-small | 124m  | adamw             | 2          | 0.02782215652499999     | 0.700804300394251            |
| 5090     | 124m-small | 124m  | muon              | 2          | 0.033646015625          | 0.7251396913905817           |
| 5090     | 124m-small | 124m  | softeq-k2000-muon | 2          | 0.031069149850000033    | 0.7531526650838439           |
| 5090     | 124m-small | 124m  | sophiag           | 2          | 0.04436137365000001     | 0.6407709350343654           |
| 5090     | 210m-main  | 210m  | adamw             | 2          | 0.03547085052500021     | 0.9632787189900467           |
| 5090     | 210m-main  | 210m  | muon              | 2          | 0.05263327017500041     | 0.948473365065402            |
| 5090     | 210m-main  | 210m  | softeq-k2000-muon | 2          | 0.046949400300000055    | 0.9546532864235573           |
| 5090     | 210m-main  | 210m  | sophiag           | 2          | 0.06968050145000004     | 0.9272666965222203           |
| 5090     | 720m-main  | 720m  | sophiag           | 2          | 0.14832503819999943     | 0.991210097108934            |
| 5090     | track3     | 124m  | muon              | 2          | 0.047816547150000055    | 0.9787036382824053           |
| 5090     | track3     | 124m  | muon              | 2          | 0.04297589054000017     | 0.98073809252263             |
| 5090     | track3     | 124m  | softeq-k2000-muon | 2          | 0.044245088749999884    | 0.9803429408335319           |
| 5090     | track3     | 124m  | softeq-k2000-muon | 2          | 0.037620289199999934    | 0.9831978574422389           |
| 5090     | track3     | 124m  | sophiag           | 2          | 0.030392911724999472    | 0.9860609786095175           |
| 5090     | track3     | 124m  | sophiag           | 2          | 0.04070046401999994     | 0.9812867782057043           |
| a800     | 124m-large | 124m  | adamw             | 2          | 0.006731247724999889    | 0.988550718897746            |

## Evidence Still Needed

- Profiler rows: `nccl_time_s`, `nccl_time_pct`, `all_reduce_time_s`.
- All-reduce microbenchmark rows for matched tensor sizes.
- A800 `nvidia-smi topo -m`; 5090 topology should be recorded as PCIe Gen5 x16, NODE, non-NVLink when confirmed from the target host.
