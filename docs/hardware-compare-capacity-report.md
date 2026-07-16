# Hardware Capacity Comparison Report

## Hardware-Best Preview

| hardware | phase           | case       | model | sequence_length | optimizer         | world_size | max_stable_microbs | best_tokens_per_second | best_wall_clock_iter_dt |
| -------- | --------------- | ---------- | ----- | --------------- | ----------------- | ---------- | ------------------ | ---------------------- | ----------------------- |
| 5090     | benchmark-probe | 124m-large | 124m  | 512             | adamw             | 1          | 32                 | 124014.91837097218     | 1.0569051024            |
| 5090     | benchmark-probe | 124m-large | 124m  | 512             | muon              | 1          | 32                 | 118899.07414666528     | 1.1023803250000002      |
| 5090     | benchmark-probe | 124m-large | 124m  | 512             | newton-muon       | 1          | 32                 | 118099.2925029472      | 1.1098457681            |
| 5090     | benchmark-probe | 124m-large | 124m  | 512             | softeq-k2000-muon | 1          | 32                 | 116547.30999389615     | 1.12462484125           |
| 5090     | benchmark-probe | 124m-large | 124m  | 512             | sophiag           | 1          | 16                 | 122881.7294720211      | 1.0666516540999997      |
| 5090     | benchmark-probe | 124m-small | 124m  | 512             | adamw             | 1          | 32                 | 125706.50239398627     | 0.13033534215000003     |
| 5090     | benchmark-probe | 124m-small | 124m  | 512             | muon              | 1          | 32                 | 92288.3016165913       | 0.17753062644999998     |
| 5090     | benchmark-probe | 124m-small | 124m  | 512             | newton-muon       | 1          | 32                 | 90843.76232597695      | 0.1803536047            |
| 5090     | benchmark-probe | 124m-small | 124m  | 512             | softeq-k2000-muon | 1          | 32                 | 86418.35584953293      | 0.18958935099999996     |
| 5090     | benchmark-probe | 124m-small | 124m  | 512             | sophiag           | 1          | 16                 | 103526.99820235392     | 0.15825823489999996     |
| 5090     | benchmark-probe | 210m-main  | 210m  | 512             | adamw             | 1          | 32                 | 70432.6754247807       | 1.86095443925           |
| 5090     | benchmark-probe | 210m-main  | 210m  | 512             | muon              | 1          | 32                 | 67643.52289170386      | 1.9376873704499993      |
| 5090     | benchmark-probe | 210m-main  | 210m  | 512             | newton-muon       | 1          | 32                 | 67230.1666086664       | 1.9496009992499999      |
| 5090     | benchmark-probe | 210m-main  | 210m  | 512             | softeq-k2000-muon | 1          | 32                 | 66305.57219233699      | 1.9767871035000002      |
| 5090     | benchmark-probe | 210m-main  | 210m  | 512             | sophiag           | 1          | 16                 | 73772.97790594213      | 1.7766939023            |
| 5090     | benchmark-probe | 720m-main  | 720m  | 512             | adamw             | 1          | 16                 | 29972.434318361233     | 33.89140799209999       |
| 5090     | benchmark-probe | 720m-main  | 720m  | 512             | muon              | 1          | 16                 | 29515.91998760051      | 34.41559675005          |
| 5090     | benchmark-probe | 720m-main  | 720m  | 512             | newton-muon       | 1          | 16                 | 29554.010968350412     | 34.37123986615          |
| 5090     | benchmark-probe | 720m-main  | 720m  | 512             | softeq-k2000-muon | 1          | 16                 | 29457.412614705354     | 34.48395191005001       |
| 5090     | benchmark-probe | 720m-main  | 720m  | 512             | sophiag           | 1          | 8                  | 30365.855705286525     | 33.4523093918           |

## Rows Needing Review

_No rows._
