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
bytes. Layer 0 now also has an exact plan-only checkpoint-shaped range set: 138,867,160 trunk
bytes in 21 ranges plus six 6,488,064-byte expert bundles in 18 ranges, or 177,795,544 logical
bytes total. The selected IDs `0..5` validate layout/capacity only; they are not authoritative
`tid2eid` output. This proves metadata-level capacity accounting, not DeepSeek execution; full
payload verification on the official local objects, authoritative routing, official-payload trunk
compute and a layer/token path remain absent. Scalar storage handling now covers every type in the
1,193 planned layer tensors,
and a reduced first layer now composes Q8_0/I32 trunk with IQ1_S/IQ3_XXS routed storage through
the architecture graph. An 8,180-byte file-backed repetition retains the bounded trunk/expert
loader buffers and their Governor leases directly through execution. A subsequent router-first
reduced test loads only two selected complete mixed-format bundles at a time through a two-entry
governed RAM cache, including reuse and safe LRU eviction.
The separately pinned 6,488,064-byte real expert now also binds its canonical 0731 GGUF identities
to that cache/kernel boundary: one miss reads the complete bundle and the following hit reads zero
logical bytes. It still does not compose official trunk payload into a complete layer/token, so it
is not a full DeepSeek execution result. The same complete bundle now passes a one-worker
asynchronous begin/acquire path. A controlled reduced two-expert fixture demonstrates that later
read service can overlap earlier CPU expert compute without changing numerical reduction order;
this is not measured physical-device overlap or evidence that the 82.54 GB profile is interactive.
The reduced graph and checkpoint-shaped real expert remain separate dimension domains, so this is
still not an official-payload layer/token result.
The pinned importer/checkpoint gate can now bind complete shards only after full SHA-256 and this
same live single-tier admission; its real validation used only the 5,343,584-byte metadata import.
The bounded partial-range importer has separately accepted the 21-range / 138,867,160-byte trunk
scope in dry-run mode. It created no target and made no payload request. A real transfer requires
explicit authorization, remains capped and publishes a non-executable derived artifact; the
native closed-world reader and exact trunk binder now pass synthetic corruption and source-identity
tests. One complete four-byte F32 range also passes an aggregate-budgeted load, second in-memory
hash, retained residency and scalar matvec. The first closed-set fixture assembled all 12
non-routed Q8_0/I32 entries from the earlier reduced packed catalog into one Governor-owned
artifact buffer.
The newer complete layer-0 fixture extends that boundary to 21 canonical objects and 2,844 trunk
bytes: 11 matrices, nine retained F32 vectors and one I32 table bind through an explicit 0731
name/shape adapter. With four separately governed expert bundles, complete reduced prefill/decode
matches its oracle and releases the aggregate lease cleanly. The exact pinned GGUF metadata now
derives the native reference configuration and validates the 21 real layer-0 descriptors without
payload access. That contract includes both official SwiGLU clamp schedules instead of applying a
single fixture scalar to routed and shared experts. The same fail-closed pass now validates all
1,193 main-layer non-routed descriptors and 6,390,088,008 encoded bytes across the three official
consumer-layout classes, still without tensor allocation or payload reads.

The next zero-payload contract closes the whole GGUF catalog rather than only the layer trunks.
It maps the six global embedding/output/normalization/Hyper-Connection objects to their exact
consumers, checks all 11,008 deterministic expert bundles and their 33,024 one-to-one logical
slices, and reconciles physical, logical and byte partitions. The real pinned result is 1,328
physical tensors, 34,223 logical descriptors and 82,533,892,444 known tensor bytes. This also
surfaces the remaining execution formats explicitly: global Q4_K, routed IQ2_XXS and routed
MXFP4 have bounded byte-layout validation but no native kernel claim.

After their parity kernels and packed global binding exist, the payload gate requires explicit
user authorization to acquire the exact 138,867,160-byte trunk ranges, execute the real
`tid2eid` route for a pinned input and fetch only its six selected complete experts.

Kimi K3 is not a reference-PC execution target. The project treats it as metadata/capacity-only
because no exact profile has proved that its complete expert bundle fits one real tier on the
32 GiB RAM / 12 GiB VRAM machine. It must not be downloaded or forced through swap to manufacture
a result.
