# Roadmap and acceptance gates

Vastloom advances only when each milestone has a measurable correctness and safety gate.
A command existing in the CLI is not enough; it must run against a validated model path.

> **Pivot architetturale (2026-08-26):** il track DeepSeek V4 Flash 0731 (M0–M13) è
> congelato in `docs/archive/deepseek_0731/`. Nuovo track: **Qwen3.8-Flash-Next**
> (preview architettura Qwen4). Roadmap Q1–Q13 sotto.

---

## Q0 - Skeleton e budget gate Qwen3.8 - complete

- [x] Verificare supporto llama.cpp mainline per `qwen4_exp`/DeltaNet/QSA/n-gram
  (commit `84e908c`: nessuna modella dedicata; solo `qwen3next` e `qwen35` con DeltaNet +
  full-attention ibrida; QSA / n-gram / Gated Residual sono ancora da implementare
  anche upstream).
- [x] Skeleton `include/vastloom/inference/qwen3_8_flash_next.hpp` con `Qwen38FlashNextConfig`,
  `Qwen38FlashNextInspection`, parser config.json pinned al 2026-08-26.
- [x] Test always-on `test_qwen38_flash_next_config` che valida parse, strict invariants
  e rejection su model_type sbagliato, senza download del checkpoint reale.
- [x] Stima budget carta in `docs/planning/QWEN38_BUDGET_ESTIMATE.md` (FP8 Unsloth come
  scelta vincolante; nessuna quota GGUF IQ chiude la qualità + RAM gate).

**Gate:** build verde con `VASTLOOM_WARNINGS_AS_ERRORS=ON`, `vastloom_tests` ALL PASSED,
ctest verde. Niente download di pesi reali a questo punto.

## Q1 - Safetensors catalog reader per Qwen3.8-Flash-Next FP8 - complete

- [x] Ispezione header remoto e `model.safetensors.index.json` per
  `unsloth/Qwen3.8-Flash-Next-FP8`: 131 shard safetensors, 152.089 tensori totali,
  range ~840 MB – 1.2 GB per shard, totale ~170 GB su disco.
- [x] POC scaricato (`model-00001-of-00131.safetensors` ~1 GB) per ispezione byte-level
  reale del safetensors header.
- [x] Scoperte architetturali dal POC reale:
  - Esperti MoE in **FP8 block-scale** (`weight` F8_E4M3 + `weight_scale_inv` BF16,
    `quant_method=fp8`, `weight_block_size=[128,128]`, shape scale `[5,20]` per
    `gate_proj` `[640,2560]`).
  - **hyper_connection_mixer** globale = Gated Residual di Qwen (4 branch,
    `hc_count=4`, `hc_lowrank=320`).
  - **attn_hyper_connection** per-layer = istanza locale della Gated Residual.
  - **linear_attn** = Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `conv1d`,
    `A_log`, `dt_bias`, `norm`, `out_proj`).
  - **layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}** = N-gram table
    PLE su layer 1 (0-indexed, corrispondente a `ple_layer_ids=[2]` 1-indexed).
  - 73.728 tensori expert non-scale + 75.264 scale_inv (FP8) = 147.456 expert
    tensors su 48 layer.
- [x] Parser safetensors header C++20 nativo in
  `read_qwen38_safetensors_header()` con attachment automatico delle coppie
  `weight` + `weight_scale_inv`.
- [x] Parser `model.safetensors.index.json` in
  `inspect_qwen38_flash_next_safetensors()` con conteggio FP8 block-scale,
  ngram tensor count, presenza di Gated Residual globale/per-layer.
- [x] Mock shard safetensors valido (3.6 KiB) in
  `tests/fixtures/qwen38_flash_next/` con FP8 block-scale experts + DeltaNet
  tensors + hyper_connection_mixer global + attn_hyper_connection.
- [x] Test always-on `test_qwen38_flash_next_safetensors` che esercita il
  parser su mock e verifica conteggi, dtype, scale_inv attachment, integrità
  dei data_offsets contro header_end e total_bytes.

**Gate:** build release verde, `vastloom_tests` ALL PASSED, `ctest --preset release`
13/14 (1 skipped intenzionale). Niente download del 125B completo (170 GB
fuori scala per POC). L'opzione "POC su 27B FP8" è stata scartata dopo
ispezione: il 27B è un modello **diverso** (`qwen3_5`, 64 layer, dense, no MoE,
no n-gram, no Gated Residual) e non avrebbe validato le tre primitive nuove.

## Q2 - Reference operators per Qwen3.8-Flash-Next FP8 - complete

> **Upstream bloccato:** llama.cpp `d0132a6` (2026-08-27) non ha ancora `qwen4_exp`.
> Nessun `src/models/qwen4*.cpp`, nessun supporto QSA / n-gram / Gated Residual
> upstream. Il gate originale “5 token oracolari pinned-ggml via llama.cpp” è
> quindi **non eseguibile** e Q2 è stato ridefinito su reference self-pinned.

- [x] Implementati reference operator puri in
  `include/vastloom/inference/qwen3_8_flash_next_reference.hpp` /
  `src/inference/qwen3_8_flash_next_reference.cpp`:
  - `qwen38_fp8_e4m3_to_f32` (OCP E4M3, exp 15 => NaN, subnormal 2^-6),
  - `qwen38_bf16_to_f32` / `qwen38_f32_to_bf16` (bit-exact),
  - `qwen38_decode_bf16` (payload -> F32 row-major),
  - `qwen38_decode_fp8_block_scale` (weight F8_E4M3 + scale BF16 per block
    `[128,128]`, shape scale `[ceil(R/128), ceil(C/128)]`),
  - `qwen38_rms_norm` (reference RMSNorm).
- [x] Mock shard rigenerato con `weight_scale_inv` in **BF16** (shape `[1,1]`,
  valore `1.0` => bytes `[0x80,0x3F]`) e payload non-zero deterministico,
  con `data_offsets` payload-relative (0-based) corretti.
- [x] Test always-on `test_qwen38_flash_next_reference_operators` che esercita:
  - FP8 E4M3 scalar smoke (0x00=0, 0x38=1.0, 0xB8=-1.0, 0x3C=1.5, 0x01 subnormal
    1/512, 0x7F/0xFF=NaN),
  - BF16 round-trip (0,1,-1,0.5,2,448,-448),
  - BF16 decode 2x2,
  - FP8 block-scale 4x4 con block 2x2 (scale uniforme e per-block),
  - RMSNorm 3-elementi,
  - lettura payload reale da mock shard (`A_log` BF16 [4]).
- [x] Verifica header remoto SHARD 2 (`model-00002-of-00131`): header 282800 byte,
  `gate_proj.weight` F8_E4M3 `[640,2560]` + `gate_proj.weight_scale_inv` BF16
  `[5,20]`, 2048 tensori per shard (1024 BF16 + 1024 F8_E4M3).

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` verde. Nessun download del 125B completo oltre al POC shard 1.

**Prossimo passo (Q4):** range importer + safety governor per il payload FP8
(lettura esatta di `weight` + `weight_scale_inv` con validazione SHA-256 e
transazione bounded, in preparazione al download del 125B).

## Q3 - Tensor & expert catalog Qwen3.8-Flash-Next FP8 - complete

- [x] Esteso `Qwen38InspectionResult` con breakdown per-layer:
  `Qwen38LayerSummary` (total, linear_attn 9 / self_attn 9, attn_hyper 4,
  mlp_hyper 4, mlp_gate 1, mlp_shared_gate 1, mlp_shared_expert 3,
  mlp_experts 3072 (1536 weight F8_E4M3 + 1536 scale BF16), ple 0/138).
  Funzione `qwen38_summarize_layers(weight_map)` classifica ogni entry per
  layer e tipo.
- [x] Aggiunto `Qwen38ExpectedCatalog` con costanti pinned per il release
  2026-08-26: `total_tensors=152089`, `total_shards=131`,
  `language_tensors=148650` (47×3094 + 3232), `per_layer 3094/3232`,
  `vision=333`, `mtp=3101`, `global=5`, `ngram=128`, `fp8_scale_inv=75264`.
  Validazione in `qwen38_validate_catalog(result)` che rifiuta qualsiasi
  catalog che non chiude i gate (total, shards, per-layer, ngram, scale).
- [x] Popolamento automatico di `layers`, `vision_tensors`, `mtp_tensors`,
  `global_tensors` dentro `inspect_qwen38_flash_next_safetensors` via
  `qwen38_summarize_layers`.
- [x] Scoperte confermate dal POC reale (già in Q1/Q2):
  48 layer con 3094 tensori ciascuno (layer 1 =3232 con PLE), 6 global
  hyper, 196+196 per-layer hyper, 333 vision, 3101 mtp.
- [x] Test always-on `test_qwen38_catalog_validation`:
  costruisce in memoria un synthetic full catalog da 152089 entries con la
  distribuzione esatta (9 linear/self, 8 hyper, 5 gate/shared, 3072 experts
  con 1536 weight +1536 scale, 128 ngram +10 ple proj per layer 1, 333 vision,
  3101 mtp, 5 global) e verifica che `summarize` produca 48 layer con
  `total 3094/3232` e che `validate` passi; verifica inoltre che il mock
  fixture (25 tensors) venga **rifiutato** dal validatore (prova di strictness).

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` verde. Nessun file da 17 MB committato: il full catalog è
sintetico in-memory per il gate, mentre il mock fixture resta 3.6 KiB.

## Q4 - Range importer + safety governor - complete

- [x] `qwen38_read_tensor_payload(shard, header, tensor_name)`: legge il payload
  esatto di un tensore dal file safetensors con bounds-check contro
  `json_header_bytes` e `total_bytes`. Se il tensore ha companion
  `weight_scale_inv`, lo legge insieme e lo restituisce separato.
- [x] Hash SHA-256 del payload calcolato al volo (`Sha256`), senza scrivere nulla.
- [x] `qwen38_import_tensor_transaction(request, header, governor)`: transazione
  tre-oggetti — (1) source range dal shard (tensor + scale), (2) expected
  SHA-256 verificato **prima** di qualsiasi write, (3) scrittura bounded su
  temp + atomic rename nel destination dir. Il Governor deve ammettere
  l'aggregato (weight+scale) come riserva RAM.
- [x] Rifiuti atomici: range out-of-bounds, hash mismatch, destination
  collision, governor refusal (`resource_exhausted`), shard/dir mancanti.
  Nessun `.tmp` residuo sui path rifiutati.
- [x] Test always-on `test_qwen38_import_transaction` su mock shard:
  (a) legge `A_log` BF16 [4] senza scale, (b) legge `up_proj.weight` F8_E4M3
  [6,5] + scale_inv BF16 [1,1], (c) tensore mancante rifiutato,
  (d) transazione completa crea `al_log_range.bin` verificato,
  (e) hash sbagliato rifiutato senza file residuo, (f) collisione
  destination rifiutata, (g) governor pieno => `resource_exhausted`.

**Gate:** `vastloom_tests` ALL PASSED, `ctest --preset release` 13/14 (1 skipped),
`compileall` verde. Import reale del payload 125B rimandato a Q5 (richiede
download).

## Q5 - Operatori nativi pinned-ggml per le tre primitive

- [ ] Decoder FP8 → F32 con kernel pinned-ggml.
- [ ] Decoder DeltaNet state (linear-conv + gated-delta update) con kernel pinned.
- [ ] Decoder QSA indexer (MQA 4-query/1-key con budget 2048) pinned.
- [ ] Decoder n-gram embedding table (lookup async prefetch) pinned.
- [ ] Decoder Gated Residual (4-branch read/write gate) pinned.

**Gate:** decoder per ogni primitiva dà output bit-exact contro llama.cpp pinned
su almeno 1 layer ridotto, e il prodotto `dequant * matvec` chiude la parità.

## Q6 - Layer graph non-routed 48 layer Qwen3.8

- [ ] Planner che produce un piano non-routed (3 linear_attn + 1 full_attn) × 12.
- [ ] Mappatura dei tensori FP8 → consumer C++20 (DeltaNet state, QSA indexer,
  Gated Residual, MoE shared, MTP).

**Gate:** piano non-routed copre esattamente 48 layer + 1 MTP layer.

## Q7 - Prefill/decode reduced layer (2–3 layer + MTP)

- [ ] Modello ridotto: 2 layer ridotti + 1 MTP.
- [ ] Forward deterministico, parità con grafo indipendentemente dequantizzato.

**Gate:** parità pinned-ggml bit-exact.

## Q8 - MoE routing 10 routed + 1 shared

- [ ] Implementare router top-k 10/512 con softmax learned (no hash shortcut,
  a differenza di DeepSeek).
- [ ] Selected expert cache con hit/miss/eviction/logical-byte telemetry.
- [ ] Confrontare routing decisions e final logits vs llama.cpp pinned.

**Gate:** routing identico su un seed pinned.

## Q9 - Gated DeltaNet state persistence + N-gram lookup resident

- [ ] State FP32 (`mamba_ssm_dtype=float32`) pinned in RAM per tutti i 48 layer.
- [ ] Lookup table n-gram 20M entries × 2560 dim × BF16 = ~102 GB; cache paginata
  con prefetch asincrono (scenario B, 4 GB working set).
- [ ] Convalida che un singolo forward di 1 token completa senza page-fault su disco
  (page cache calda) e che la banda per token è entro il budget previsto.

**Gate:** RAM viva misurata < 12.3 GB su host reference; persist window 48 layer
chiude.

## Q10 - QSA sparse attention (MQA indexer + gather + matmul sparse)

- [ ] Implementare indexer con 4 query heads / 1 key head (dim 128) e budget 2048.
- [ ] Gather dei blocchi selezionati e matmul sparse pinned-ggml.
- [ ] Test parità pinned-ggml su una lunghezza di context 4096 pinned.

**Gate:** logits identici su contesto corto.

## Q11 - Full-depth generate validate (48 layer + MTP)

- [ ] Generare 20+ token dal modello 125B FP8 sul prompt italiano pinned
  (analogo al prompt "Roma" usato per DeepSeek).
- [ ] Confronto bit-exact con llama.cpp pinned-ggml.

**Gate:** gli ultimi 8 token generati coincidono.

## Q12 - prefix-KV MoE stream (porting M13)

- [ ] Estendere `LlamaSession` al percorso MoE stream.
- [ ] KV cache DeltaNet/QSA persistita attraverso richieste consecutive.
- [ ] Endpoint OpenAI compatibile come per M13 SmolLM2.

**Gate:** due turni consecutivi con cached_prefix_tokens > 0 sul MoE stream.

## Q13 - vastloom serve per Qwen3.8

- [ ] Dispatch `serve <directory>` che carica Qwen3.8-Flash-Next FP8.
- [ ] Endpoint `/v1/models`, `/v1/chat/completions` (streaming + non-stream).
- [ ] Test Python opt-in via `tests/test_qwen38_cli.py`.

**Gate:** smoke test verde end-to-end.

---

## Track archiviato

Il track M0–M13 (DeepSeek V4 Flash 0731) è congelato in `docs/archive/deepseek_0731/`.
I test C++ che difendono la parità pinned-ggml per IQ1_S/IQ2_XXS/Q4_K restano
compilati e verdi — sono contratti storici. Per il M14+ originale (multi-backend,
Kimi K3, Laguna, MiniMax, Vulkan/ROCm, multi-NVMe) vedi la roadmap archiviata.
