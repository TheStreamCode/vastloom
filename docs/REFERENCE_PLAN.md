# Reference consumer plan

Command:

```bash
vastloom plan examples/manifests/deepseek-v4-flash.research.sgm \
  --reference-pc --profile balanced
```

Current alpha output:

```text
OS reserve:         6.40 GiB
Emergency reserve:  2.00 GiB
Vastloom RAM max:  23.04 GiB
VRAM reserve:       1.56 GiB
Vastloom VRAM max: 10.22 GiB
Storage reserve:  100.00 GiB

Storage to add:   148.67 GiB

Dense in RAM:       8.40 GiB
Expert RAM cache:   9.92 GiB
KV in RAM:          3.23 GiB
RAM staging:        1.50 GiB
GPU resident:       7.97 GiB
GPU KV:             1.64 GiB
GPU staging:      627.92 MiB
NVMe backing:     148.66 GiB
```

This proves the planner keeps hard headroom on the target machine. It does not prove the
generic placement strategy is executable or any tokens-per-second result. The capacity
inputs now come from the exact M6 checkpoint/header catalog; DeepSeek fixture parity,
kernels, KV sizing and the architecture-specific execution plan remain unimplemented.
