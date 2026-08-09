# Bundled validation model

`stargate-tinychat-v1` is a tiny synthetic causal Transformer committed solely to test
Stargate's native inference path.

Files:

- `stargate-tinychat-v1.sgtm` — native FP32 TinyTransformer weights;
- `stargate-tinychat-v1.sgm` — exact Stargate manifest and SHA-256;
- `stargate-tinychat-v1.tests.json` — deterministic response vectors.

It is **not** a general-purpose assistant. It memorizes a small synthetic corpus and has
a 128-byte-token context. Each CLI line is treated as an independent prompt.

Validate it with:

```bash
stargate verify examples/models/stargate-tinychat-v1.sgm --full
stargate self-test examples/models/stargate-tinychat-v1.sgm
stargate chat examples/models/stargate-tinychat-v1.sgm
```

Rebuild all three artifacts with:

```bash
python3 tools/train_tiny_chat.py
```

PyTorch is required only to retrain/export the fixture. Native inference has no Python
or PyTorch dependency.
