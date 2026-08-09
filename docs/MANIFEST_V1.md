# Stargate manifest v1

The v1 format is intentionally line-oriented and bounded. It can be parsed without a
large third-party serialization dependency before any model data is trusted.

## Limits

- maximum manifest file: 16 MiB;
- maximum line: 64 KiB;
- duplicate keys: rejected;
- duplicate chunk paths: rejected;
- absolute paths, `.` and `..`: rejected;
- required chunk hash: exactly 64 hexadecimal SHA-256 characters.

## Scalar fields

```text
format=stargate.manifest.v1
model_id=organization/model
revision=exact-checkpoint-revision
architecture=adapter_identifier
tokenizer_family=optional_identifier
metadata_only=false
requires_avx2=true
supports_speculation=true

model_total_bytes=156GiB
required_storage_bytes=180GiB
dense_bytes=20GiB
routed_expert_bytes=136GiB
active_bytes_per_token=13GiB
estimated_expert_bytes=48MiB
minimum_ram_bytes=16GiB
recommended_ram_bytes=32GiB
minimum_vram_bytes=0
recommended_vram_bytes=12GiB

layer_count=43
expert_count=256
selected_experts_per_token=6
max_context_tokens=131072
chunk_count=96
```

Byte fields accept decimal (`GB`, `TB`) and binary (`GiB`, `TiB`) suffixes.

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
  --model-id deepseek-ai/DeepSeek-V4-Flash-0731 \
  --architecture deepseek_v4_flash \
  --revision exact-revision \
  --minimum-ram 16GiB \
  --recommended-ram 32GiB \
  --recommended-vram 12GiB \
  --requires-avx2 \
  --supports-speculation
```

The tool rejects symlinks by default, hashes every chunk in an 8 MiB streaming buffer,
checks that the file did not change while hashing, and publishes through `os.replace`
only after the full manifest is complete.
