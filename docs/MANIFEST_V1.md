# Vastloom manifest v1

The v1 format is intentionally line-oriented and bounded. It can be parsed without a
large third-party serialization dependency before any model data is trusted.

New artifacts use `vastloom.manifest.v1`. The reader also accepts the former
`stargate.manifest.v1` identifier and emits a compatibility warning so verified existing
artifacts remain usable. Generators never write the legacy identifier.

## Limits

- maximum manifest file: 16 MiB;
- maximum line: 64 KiB;
- duplicate keys: rejected;
- duplicate chunk paths: rejected;
- absolute paths, `.` and `..`: rejected;
- required chunk hash: exactly 64 hexadecimal SHA-256 characters.

## Scalar fields

```text
format=vastloom.manifest.v1
model_id=organization/model
revision=exact-checkpoint-revision
architecture=adapter_identifier
tokenizer_family=optional_identifier
metadata_only=false
requires_avx2=true
supports_speculation=true

model_total_bytes=159617149040
required_storage_bytes=159628890287
dense_bytes=9017195080
routed_expert_bytes=150592290816
active_bytes_per_token=3449290752
estimated_expert_bytes=13369344
minimum_ram_bytes=16GiB
recommended_ram_bytes=32GiB
minimum_vram_bytes=0
recommended_vram_bytes=12GiB

layer_count=43
expert_count=256
selected_experts_per_token=6
max_context_tokens=1048576
chunk_count=96
```

Byte fields accept decimal (`GB`, `TB`) and binary (`GiB`, `TiB`) suffixes.

For MoE manifests, `estimated_expert_bytes` is the encoded size of one complete indivisible routed
expert bundle for the adapter's selected representation. `minimum_ram_bytes` is the adapter's
minimum live host working set after accounting for the required always-active block, router and
reduction state, requested/context-bounded KV, activation scratch and one bounded staging slot;
it is not installed RAM and does not include optional cache. The planner checks this value against
the live post-reserve RAM budget and independently requires one complete expert to fit an actual
RAM or VRAM tier.

The reader rejects MoE metadata unless all of the following are true:

- `expert_count`, `layer_count` and `max_context_tokens` are nonzero;
- `selected_experts_per_token` is between one and `expert_count`;
- routed and one-expert byte counts are nonzero, and one expert does not exceed all routed bytes;
- `minimum_ram_bytes` is nonzero;
- dense, routed and active byte counts do not exceed the checkpoint total.

A dense manifest uses zero for all expert fields. A metadata-only manifest may pass capacity
planning, but it still cannot authorize weight execution.

For persisted compatibility, an older `stargate.manifest.v1` artifact with incomplete MoE
admission metadata remains readable and receives an explicit warning. The native planner still
rejects it fail-closed. New `vastloom.manifest.v1` artifacts are rejected immediately when these
fields are incomplete; the generator never emits the legacy identifier.

## Chunk rows

Chunk rows use exactly nine tab-separated fields:

```text
chunk<TAB>relative_path<TAB>size<TAB>sha256<TAB>role<TAB>dtype<TAB>layer<TAB>expert<TAB>required
```

Example:

```text
chunk	weights/model-00001.safetensors	17179869184	<sha256>	expert	mxfp4	12	73	true
```

Use `-` when layer or expert is not applicable.

Supported roles in v1:

```text
dense expert router embedding output tokenizer config support other
```

## Metadata-only manifests

`metadata_only=true` permits `chunk_count=0` for capacity planning and design experiments.
It cannot establish checkpoint integrity and must be replaced by an exact generated
manifest before inference.

## Generating an exact manifest

Use the offline standard-library tool against a completed local checkpoint:

```bash
python tools/generate_manifest.py /models/deepseek-v4-flash \
  --output /models/deepseek-v4-flash/model.sgm \
  --model-id deepseek-ai/DeepSeek-V4-Flash \
  --architecture deepseek_v4 \
  --revision exact-revision \
  --minimum-ram 16GiB \
  --recommended-ram 32GiB \
  --recommended-vram 12GiB \
  --layer-count 43 \
  --expert-count 256 \
  --selected-experts-per-token 6 \
  --max-context-tokens 1048576 \
  --routed-expert-bytes 150592290816 \
  --estimated-expert-bytes 13369344 \
  --requires-avx2 \
  --supports-speculation
```

The tool rejects symlinks by default, hashes every chunk in an 8 MiB streaming buffer,
checks that the file did not change while hashing, and publishes through `os.replace`
only after the full manifest is complete.
