# Bundled validation model

`vastloom-tinychat-v1` is a tiny synthetic causal Transformer committed solely to test
Vastloom's native inference path.

Files:

- `vastloom-tinychat-v1.sgtm` — native FP32 TinyTransformer weights;
- `vastloom-tinychat-v1.sgm` — exact Vastloom manifest and SHA-256;
- `vastloom-tinychat-v1.tests.json` — deterministic response vectors.

It is **not** a general-purpose assistant. It memorizes a small synthetic corpus and has
a 128-byte-token context. Each CLI line is treated as an independent prompt.

Validate it with:

```bash
vastloom verify examples/models/vastloom-tinychat-v1.sgm --full
vastloom self-test examples/models/vastloom-tinychat-v1.sgm
vastloom chat examples/models/vastloom-tinychat-v1.sgm
```

Rebuild all three artifacts with:

```bash
python3 tools/train_tiny_chat.py
```

PyTorch is required only to retrain/export the fixture. Native inference has no Python
or PyTorch dependency.
