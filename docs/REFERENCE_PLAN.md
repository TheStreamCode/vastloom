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
kernels, KV sizing and the architecture-specific execution plan remain incomplete.

## Reference-PC model admission gate

Installed capacity is never the usable budget. Before downloading or executing a large MoE,
Vastloom must derive a live plan and prove that this minimum irreducible working set fits while
preserving host headroom:

```text
largest required always-active block
  + one complete routed expert bundle
  + router/reduction state
  + bounded KV for the requested context
  + activation and kernel scratch
  + one bounded read/transfer staging slot
  <= safe RAM and optional VRAM budgets after OS reserves
```

Cache, extra resident experts and prefetch depth are optional and must shrink first. The complete
selected expert is not delegated to uncontrolled OS swap and is loaded only after its destination
and staging reservations succeed. A future CUDA path may tile compute in VRAM, but its complete
expert backing and transfer buffers must still fit the governed RAM plan.

On the reference PC, the nominal 32 GiB RAM and 12 GiB VRAM are upper physical capacities, not
allocation targets. The planner must use current availability and keep the OS/driver reserves
shown above. A model/profile whose minimum working set exceeds either derived hard budget is
ineligible for execution on this machine; Vastloom must reject it and recommend a smaller model or
explicitly validated quantization instead of swapping, overcommitting or silently losing quality.

The 6,488,064-byte DeepSeek 0731 expert tested in M7 passes the isolated expert-residency part of
this gate under an 8 MiB test RAM limit. The full DeepSeek trunk/KV/layer/token plan has not passed
the complete admission gate yet. Kimi K3 is not a reference-PC execution target: it remains only a
future capacity/traffic study unless an exact published profile independently passes this rule.
