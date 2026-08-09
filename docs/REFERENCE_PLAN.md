# Reference consumer plan

Command:

```bash
stargate plan examples/manifests/deepseek-v4-flash-0731.research.sgm \
  --reference-pc --profile balanced
```

Current alpha output:

```text
OS reserve:         6.40 GiB
Emergency reserve:  2.00 GiB
Stargate RAM max:  23.04 GiB
VRAM reserve:       1.56 GiB
Stargate VRAM max: 10.22 GiB
Storage reserve:  100.00 GiB

Dense in RAM:      14.65 GiB
Expert RAM cache:   3.66 GiB
KV in RAM:          3.23 GiB
RAM staging:        1.50 GiB
GPU resident:       7.97 GiB
GPU KV:             1.64 GiB
GPU staging:      627.92 MiB
```

This proves the planner keeps hard headroom on the target machine. It does not prove the
provisional DeepSeek footprint or any tokens-per-second result. The exact checkpoint
inspector in M2 will replace provisional values.
