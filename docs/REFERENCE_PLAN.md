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
runtime state (router/reduction + bounded KV + compression state)
  + activation/kernel scratch and one bounded trunk tile
  <= safe host RAM after OS reserves

one complete routed-expert bundle
  <= one actual remaining RAM tier OR one actual VRAM tier

resident-trunk profile only:
  all always-active encoded weights
  + runtime state
  + expert tier
  <= their separately governed destinations
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
also passes the isolated expert-residency part under a 10 MiB aggregate host-memory ceiling.

The official 0731 `UD-IQ1_S` GGUF headers now produce a derived batch-1, 4,096-token plan: the
runtime state is 116,123,672 bytes, the largest complete expert bundle is 8,781,824 bytes, the
streamed-trunk CPU floor is 124,905,496 bytes, and the resident-trunk CPU floor is 7,110,994,292
bytes. This proves metadata-level capacity accounting, not DeepSeek execution; full payload
verification on the official local objects, official-payload trunk compute and a layer/token path
remain absent. Scalar storage handling now covers every type in the 1,193 planned layer tensors,
and a reduced first layer now composes Q8_0/I32 packed storage through the architecture graph.
A file-backed repetition retains the bounded trunk/expert loader buffers and their Governor leases
directly through execution. A subsequent router-first reduced test loads only two selected complete
bundles at a time through a two-entry governed RAM cache, including reuse and safe LRU eviction.
The separately pinned 6,488,064-byte real expert now also binds its canonical 0731 GGUF identities
to that cache/kernel boundary: one miss reads the complete bundle and the following hit reads zero
logical bytes. It still does not compose official trunk payload into a complete layer/token, so it
is not a full DeepSeek execution result. The same complete bundle now passes a one-worker
asynchronous begin/acquire path. A controlled reduced two-expert fixture demonstrates that later
read service can overlap earlier CPU expert compute without changing numerical reduction order;
this is not measured physical-device overlap or evidence that the 82.54 GB profile is interactive.
The pinned importer/checkpoint gate can now bind complete shards only after full SHA-256 and this
same live single-tier admission; its real validation used only the 5,343,584-byte metadata import.

Kimi K3 is not a reference-PC execution target. The project treats it as metadata/capacity-only
because no exact profile has proved that its complete expert bundle fits one real tier on the
32 GiB RAM / 12 GiB VRAM machine. It must not be downloaded or forced through swap to manufacture
a result.
