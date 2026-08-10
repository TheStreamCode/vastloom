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

Minimum RAM set:   16.00 GiB
Complete expert:   12.75 MiB
Expert tier cap:    9.00 GiB
Expert tier:        RAM

Dense in RAM:       8.40 GiB
Expert RAM cache:   9.00 GiB
KV in RAM:          3.23 GiB
RAM staging:        1.50 GiB
RAM scratch:      943.72 MiB
GPU resident:       7.97 GiB
GPU KV:             1.64 GiB
GPU staging:      627.92 MiB
NVMe backing:     148.66 GiB
```

This proves that the metadata profile passes the implemented capacity-admission calculation and
that the planner keeps hard headroom on the target machine. It does not make a metadata-only
manifest executable, validate its declared 16 GiB working-set estimate against the full packed
graph, or prove a tokens-per-second result. The capacity inputs come from the exact M6
checkpoint/header catalog; full payload binding, complete packed layer/token parity and the
architecture-specific executable plan remain incomplete.

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

The planner now enforces the manifest-level gate and reports the actual selected expert tier. A
20 GiB simulated-RAM plan rejects the DeepSeek research profile because its post-reserve RAM
budget is below the declared 16 GiB minimum. The 6,488,064-byte DeepSeek 0731 expert tested in M7
also passes the isolated expert-residency part under a 10 MiB aggregate host-memory ceiling. The
full DeepSeek trunk/KV/layer/token implementation has not validated the metadata profile's minimum
working-set declaration yet, so DeepSeek execution remains unsupported. Kimi K3 is not a
reference-PC execution target: it remains only a future capacity/traffic study unless an exact
published profile independently passes this rule.
