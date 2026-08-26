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

## Q1 - POC Qwen3.8-27B FP8 (gate di fattibilità per il 125B)

- [ ] Scaricare `unsloth/Qwen3.8-27B-FP8` (~15 GB) e verificare range del file
  (`Content-Length`, dtype) via header remoto.
- [ ] Implementare il catalog reader per FP8 in `src/inference/qwen3_8_flash_next.cpp`.
- [ ] Implementare decoder FP8 → F32 pinned-ggml con un kernel di riferimento.
- [ ] Implementare **un singolo layer ridotto** Qwen3.8-27B (DeltaNet + MoE shared +
  QSA opzionale) e produrre forward di 1 token bit-exact contro llama.cpp pinned-ggml.
- [ ] Misurare RAM viva in single-token forward: deve stare sotto 12.3 GB.

**Gate:** parità pinned-ggml su 1 layer + RAM viva < 12.3 GB. Se boccia: cancelliamo
e basta (vincolo utente) e ripristiniamo la roadmap DeepSeek.

## Q2 - Reference pinned-ggml su Qwen3.8-Flash-Next FP8 (125B)

- [ ] Clonare llama.cpp mainline (commit `84e908c` o superiore che includa Qwen3.8
  upstream) in `build/research-llama-cpp` con tutte e tre le primitive.
- [ ] Costruire localmente `llama-cli` + `llama-gguf-hash` con `GGML_CUDA=OFF`
  (target CPU puro) o `GGML_CUDA=ON` (target RTX 4070 Super) secondo gate RAM
  misurato in Q1.
- [ ] Validare con `unsloth/Qwen3.8-Flash-Next-FP8` (5 token oracolari pinned) sul
  prompt italiano che useremo in Q11.

**Gate:** output token-0 e token-1 deterministici, identici tra esecuzioni ripetute.

## Q3 - Tensor & expert catalog Qwen3.8-Flash-Next FP8

- [ ] Estendere `Qwen38FlashNextInspection` con enumeration completa dei tensori
  dal Safetensors header del 125B FP8 (layer-trunk, shared, routed, n-gram tables,
  gated-residual, DeltaNet state, QSA indexer, MTP).
- [ ] Test pinned-ggml che confronta il nostro catalog con quello generato da
  `tools/gguf-ls.py` su llama.cpp pinned.

**Gate:** stesso numero di tensori e stessi nomi canonici `blk.<i>.*`.

## Q4 - Range importer + safety governor

- [ ] Range-binding esatto (tre-object gate come per DeepSeek 0731).
- [ ] Hash SHA-256 del payload FP8 vs pinned upstream.
- [ ] Importatore pubblica artifact read-only con rename atomico.

**Gate:** transazione fallisce su duplicati, source sbagliato, hash corrotto.

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
