# Roadmap and acceptance gates

Vastloom advances only when each milestone has a measurable correctness and safety gate.
A command existing in the CLI is not enough; it must run against a validated model path.

> **Pivot architetturale (2026-08-26):** il track DeepSeek V4 Flash 0731 (M0–M13) è
> congelato in `docs/archive/deepseek_0731/`. Nuovo track: **Qwen3.8-Flash-Next**
> (preview architettura Qwen4). Roadmap Q0–Q14 below (active track kept in Italian; section titles are self-explanatory).

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
`compileall` verde. Import reale del payload 125B resta rimandato (richiede
download dopo i kernel e il POC).

## Q5 - Operatori nativi self-pinned per le tre primitive - complete

> **Upstream bloccato:** llama.cpp `d0132a6` non ha `qwen4_exp`. Q5 è self-pinned
> sulle equazioni di `transformers` `modeling_qwen4_exp.py` / `modular_qwen4_exp.py`
> (Hugging Face, 2026-08), con golden values da oracolo Python ridotto.

- [x] Decoder FP8 → F32 già in Q2 (`qwen38_fp8_e4m3_to_f32`, block-scale).
- [x] `qwen38_rms_norm_affine`: RMS raggruppato + affine `(1 + weight)` come
  `Qwen4ExpTextRMSNorm`.
- [x] Gated Residual: `qwen38_gated_residual` + `qwen38_gated_residual_combine`
  (SiLU bottleneck / `hc_count`, sigmoid read-gate, media sui branch,
  write-gate `2 * sigmoid(inject / hc_count)`).
- [x] Gated DeltaNet one-token: `qwen38_delta_net_decode` (conv1d causale + SiLU,
  L2-norm Q/K, repeat_interleave, `g = -exp(A_log) * softplus(a + dt_bias)`,
  update ricorrente FP32, RMSNorm gated-sigmoid, `out_proj`).
- [x] QSA indexer: `qwen38_qsa_select` (pool `compress_ratio`, k-RMSNorm,
  `relu(q·k).sum / sqrt(d)`, top-k blocchi + coda incompleta).
- [x] N-gram: `qwen38_splitmix64`, `qwen38_nth_prime_after`,
  `qwen38_ngram_multipliers` (seed 1234), `qwen38_ngram_hash_ids`
  (shift-right ignore-EOS + XOR hash), `qwen38_ngram_lookup`.
- [x] Test always-on esteso in `test_qwen38_flash_next_reference_operators`.

**Gate:** `vastloom_tests` ALL PASSED. Nessun download del 125B. Prefetch async
n-gram e RoPE indexer restano Q9/Q10.

## Q6 - Layer graph non-routed 48 layer Qwen3.8 - complete

- [x] `qwen38_plan_non_routed_graph(config)`: 36 Gated DeltaNet + 12 QSA
  (`full_attention_interval=4`) + 1 MTP QSA. Nessun tensore routed expert.
- [x] Nomi canonici dal catalogo reale Unsloth FP8: 22 non-routed/layer
  (9 attn + 8 hyper + 5 shared/router), +138 PLE su layer 1, 29 MTP
  (22 + 4 head + 3 mixer), 5 global (embed, lm_head, mixer).
- [x] Mapping consumer: DeltaNet, QSA indexer/attention, Gated Residual
  attn/mlp, MoE router/shared, n-gram PLE, MTP head, global embed/lm_head/mixer.
- [x] `qwen38_validate_non_routed_plan` chiude 48+1, 36/12, 1194+5+29=1228.
- [x] `qwen38_bind_non_routed_plan` attacca gli shard; manca un nome → refuse.
- [x] Test always-on `test_qwen38_non_routed_graph_plan`.

**Gate:** piano non-routed copre esattamente 48 layer + 1 MTP. Nessun download.

## Q7 - Prefill/decode reduced layer (2–3 layer + MTP) - complete

> Self-pinned: llama.cpp non ha `qwen4_exp`. Il grafo ridotto ricompone i
> kernel Q5; l'oracolo è la stessa sequenza di primitive, non ggml.

- [x] Grafo ridotto F32: embed → stream `hc_count` → layer0 Gated Residual +
  DeltaNet + shared SwiGLU → layer1 Gated Residual + QSA (indexer + GQA
  mascherata + gate sigmoide) + shared SwiGLU → mixer globale → `lm_head`.
- [x] MTP: `rms(h)`/`rms(embed)` + `fc_hidden`/`fc_embed` → un layer QSA+MLP
  + mixer → logits extra.
- [x] `qwen38_reduced_prefill` / `qwen38_reduced_decode` con state DeltaNet e
  cache QSA/MTP. Prefill di 2 token == due decode incrementali.
- [x] Primitive pubbliche `qwen38_linear`, `qwen38_shared_mlp`,
  `qwen38_expand_stream`, `qwen38_qsa_layer_token`.
- [x] Test always-on `test_qwen38_reduced_graph`. Nessun esperto routed (Q8).

**Gate:** `vastloom_tests` ALL PASSED. Nessun download.

## Q8 - MoE routing 10 routed + 1 shared - complete

> Self-pinned su `Qwen4ExpTextTopKRouter` / `Qwen4ExpTextSparseMoeBlock`
> (`norm_topk_prob=true`). llama.cpp non ha `qwen4_exp`.

- [x] `qwen38_moe_route`: logits = xW^T, softmax F64, top-k, rinormalizza i
  pesi selezionati. Nessuno hash-route.
- [x] `qwen38_expert_mlp` + `qwen38_moe_block`: somma pesata dei routed +
  shared SwiGLU (gate sigmoide).
- [x] LRU `qwen38_expert_cache_acquire` con hit/miss/eviction/logical-byte.
  Chiave `(layer << 16) | expert`.
- [x] Grafo ridotto Q7: 4 esperti, top-2, cache da 2 slot su layer0/1/MTP.
- [x] Test `test_qwen38_moe_router` (golden [1,0]→esperti 0+2, pesi 0.5/0.5)
  e telemetry nel reduced prefill.

**Gate:** routing self-pinned sul seed del test. Nessun download. Nessun
confronto llama.cpp.

## Q9 - Gated DeltaNet state persistence + N-gram lookup resident - complete (infra)

> Nessun download del 125B. Il gate “RAM viva < 12.3 GB sul generate ufficiale”
> resta **non misurato**. Q9 chiude il persist planner e la cache paginata su
> fixture.

- [x] `qwen38_plan_persist_budget`: 36 layer DeltaNet, conv+SSM FP32 =
  117 669 888 B (~112.2 MiB) + working set n-gram (default 4 GiB scenario B).
  Totale carta 4.11 GiB < gate 12.3 GiB.
- [x] `qwen38_deltanet_persist_create`: riserva Governor e azzera 36 state
  (conv 10240×3, SSM 48×128×128).
- [x] `qwen38_ngram_page_cache_open` / `qwen38_ngram_page_gather`: pagine
  bounded, LRU, prefetch della pagina successiva, telemetry
  hit/miss/eviction/prefetch/logical-byte.
- [x] Test always-on `test_qwen38_persist_and_ngram_pages`.

**Gate infra:** budget e cache verdi. **Non** è una misura live sul 125B.

## Q10 - QSA sparse attention (MQA indexer + gather + matmul sparse) - complete (self-pinned)

> llama.cpp non ha `qwen4_exp`. Il gate “pinned-ggml 4096” è self-pinned.

- [x] `qwen38_qsa_select` già in Q5 (indexer MQA 4q/1k, budget 2048, compress 4).
- [x] `qwen38_qsa_sparse_attention`: GQA gather sui soli `selected`, scaled-dot,
  softmax, weighted-sum, gate sigmoide — estratto da `qwen38_qsa_layer_token`.
- [x] Test always-on `test_qwen38_qsa_sparse`: sparse su 2 token vs dense su 8,
  budget enforcement e pipeline indexer→sparse su 8 token sintetici.

**Gate:** parità self-pinned su contesto corto. 4096 reale non misurato.

## Q11 - Full-depth generate validate (48 layer + MTP) - complete (infra)

> Il gate reale (20+ token sul 125B FP8, confronto llama.cpp) resta **non
> eseguibile** senza checkpoint. Q11 chiude il driver generate sul grafo
> ridotto, pronto per il full-depth.

- [x] `qwen38_reduced_generate`: prefill + decode loop, greedy/sampled
  (softmax+top-k seeded), `stop_token_ids`, `context_limit_tokens`,
  cancellazione, timeout, `on_token`, stats (prompt/generated/forward,
  expert telemetry, ms, tok/s).
- [x] Test always-on: greedy 3 token, determinismo seeded, stop/cancel/
  timeout/context-limit, prompt vuoto rifiutato.

**Gate infra:** driver verde su grafo ridotto. 125B non scaricato.

## Q11b - Catalogo reale 125B FP8 - complete

> Il checkpoint `unsloth/Qwen3.8-Flash-Next-FP8` (172.8 GiB, 131 shard, 140
> file) è stato importato tramite il path canonico
> (`tools/import_qwen38_fp8.py` su `import_hf_model.import_model`: store
> `.vastloom-store`, lock, SHA-256 per file, manifest atomico, read-only) in
> `C:\vastloom-store\models\original\unsloth-Qwen3.8-Flash-Next-FP8--main`.
> Il download ha richiesto il token HF (throttle anonimo ~30 KB/s dopo burst;
> autenticato ~78 MB/s, ~35 min totali).

- [x] `vastloom qwen38-inspect <root> [--load-header] [--json]`: ispezione
  metadata-only config + index (+ header primo shard) con validazione pin.
- [x] Validazione completa del catalogo reale: 131 shard, 152 089 tensori,
  75 264 esperti FP8 block-scale, 129 tensori n-gram (128 shard + weight_scale),
  48 layer, vision 333, MTP 3101, globali 5.
- [x] Drift fixture→reale corretto: `ExpectedCatalog::ngram_tensors` 128→129
  (il tensore `ple_embedding.ngram_embedding.weight_scale` esiste nel
  checkpoint reale e non nella fixture Q1; già previsto nel piano Q6).
- [x] Test opt-in seriale `vastloom_qwen38_checkpoint`
  (`-DVASTLOOM_QWEN38_MODEL=<root>`), metadata-only: PASS sul checkpoint reale.

**Gate:** catalogo reale validato end-to-end in C++ nativo. Nessun payload
tensore ancora letto per il compute (prossimo passo: bind e layer reale).

## Q11c - Layer-0 reale in esecuzione - complete

- [x] `vastloom qwen38-layer0 <root> --token N [--json] [--dump-hidden]`:
  forward completo del layer-0 reale — riga embedding (lettura bounded di una
  sola riga, 5 KB), attn gated residual, Gated DeltaNet (state azzerato),
  combine, mlp gated residual, router top-10 su 512, shared expert, carico
  on-demand dei 10 esperti FP8 selezionati, combine finale.
- [x] **Bug FP8 trovato sul payload reale e corretto**: `qwen38_fp8_e4m3_to_f32`
  trattava tutto `exp==15` come NaN; in E4M3FN (formato HF) NaN è solo
  `exp==15 && mant==7`, mentre `0x78–0x7E` sono finiti fino a 448. Il
  checkpoint ufficiale contiene tali encoding (3 370 su un solo gate_proj);
  il vecchio decoder corrompeva qualunque inference reale. Test Q2 esteso con
  i nuovi pin (0x78→256, 0x7E→448, 0xFE→−448).
- [x] Guardie di finitezza per stadio (embed/stream/gated/DeltaNet/shared/
  esperti) con errori che nominano lo stadio.
- [x] Test opt-in esteso: forward reale su token BOS, finitezza, route
  top-k esatta, determinismo bit-pari su doppia esecuzione.
- [x] Router verificato sensibile all'input: token 100 → esperti
  {215,226,122,448,483,363,459,40,239,500}; token BOS → {411,334,286,152,291,
  106,421,304,226,331}.

**Gate:** layer-0 completo eseguito su pesi reali (204 MB payload letti,
~1.2 s load, ~46 ms forward). Output deterministico e finito. È un run
self-pinned: nessun oracolo esterno esiste per `qwen4_exp`.

## Q11d - Prefill multi-token reale layer-0 - complete

- [x] Rifattorizzazione a trunk/row: `qwen38_load_layer0_trunk` (22 tensori +
  mappa esperto→shard costruita una sola volta, niente re-parse dell'indice
  da 17 MiB per esperto), `qwen38_load_embed_row` (lettura bounded 5 KiB),
  stato DeltaNet di proprietà del chiamante.
- [x] `vastloom qwen38-prefill <root> --tokens <csv> [--expert-cache N]`:
  prefill multi-token con conv history e SSM che attraversano i token, LRU
  esperti bounded (cap default 16 ≈ 315 MiB decodificati) e telemetry.
- [x] Regression implicita: il token BOS nel prefill riproduce il sum
  single-token di Q11c (3.0418) bit-pari.
- [x] Streaming verificato: il token 100 isolato e lo stesso token dopo BOS
  producono route diverse ({215,...} vs {283,...}) — la route dipende dallo
  stato, com'è corretto.
- [x] LRU reale su 5 token: 44 miss / 6 hits / 28 evictions; gli hit derivano
  dalla ricorrenza reale degli esperti (226 nei token 0–2, 334 nei token 0/4).
- [x] Test opt-in esteso: prefill 5 token, replay bit-pari, state-carry
  assertion, cross-check CLI/test sugli stessi sum.

**Gate:** prefill multi-token reale deterministico (replay bit-pari) con
stato che attraversa i token. Self-pinned; QSA/MTP e full-depth restano
aperti (Q11e).

## Q11e - Layer QSA reale (layer 3) - complete

- [x] Loader generalizzato a layer arbitrario: `qwen38_load_layer_trunk`
  risolve i 22 tensori non-routed dal weight_map (niente shard hardcoddati),
  carica il ramo DeltaNet (9 tensori) o QSA (9 tensori) secondo
  `full_attention_interval`, e mappa gli esperti del layer richiesto.
- [x] Forward generico `qwen38_layer_forward` con `Qwen38LayerRuntimeState`
  caller-owned: conv/SSM per DeltaNet, cache indexer-keys + KV per QSA.
- [x] **Layout QSA reale confermato dall'operatore pinnato**: `q_proj`
  [12288, 2560] = query+gate per head (24×256×2), `index_qk_proj` [640, 2560]
  = (4+1)×128, `k/v_proj` [512, 2560] = 2×256, `o_proj` [2560, 6144].
- [x] CLI: `qwen38-layer --layer N` e `qwen38-prefill --layer N` (alias
  `qwen38-layer0` compatibile).
- [x] Layer 3 QSA reale: BOS sum 28.0353, prefill 5 token deterministico
  (replay bit-pari), route dipendente dallo stato, esperti FP8 selezionati
  {169,418,141,...}; CLI e test concordano.

**Gate:** entrambi i tipi di layer (36 DeltaNet + 12 QSA) eseguono su pesi
reali con prefill multi-token deterministico. Self-pinned.

## Q11f - Catena e full-depth reale (48 layer) - complete

- [x] `qwen38_layer_forward` accetta lo **stream di input** (hc_count ×
  hidden) invece della riga embedding: l'output gated-residual di un layer
  alimenta il successivo; solo il layer 0 parte dall'embedding espanso.
- [x] `vastloom qwen38-chain --layers <csv> --tokens <csv> --expert-cache N
  [--no-trunk-resident] [--trace-layers]`: catena di layer reali con stato
  per layer e LRU esperti condivisa (chiave layer<<16|expert).
- [x] **Bug di memoria trovato e corretto**: in modalità streaming (trunk
  non residente) il print finale indicizzava `trunks[i]` oltre l'unico
  elemento caricato — lettura OOB che causava l'access violation
  (0xC0000005) del full-depth; le catene corte passavano solo per layout
  heap fortunato. Il tipo di attenzione ora deriva dall'id del layer.
- [x] Modalità streaming verificata identica alla residente (stessi sum,
  catena 0–3): trunk caricato ed eseguito un layer alla volta.
- [x] **Run full-depth reale 0→47**: 48/48 layer (36 DeltaNet + 12 QSA),
  480 esperti FP8 reali, 9 633 094 400 byte di payload letti, replay
  bit-pari (sum −28.9797). Nota documentata: il blocco PLE n-gram del
  layer 1 non è ancora integrato nel forward.

**Gate:** full-depth trunk+MoE eseguito su pesi reali, deterministico.
Self-pinned; il confronto esterno arriva con il Q8_0 + llama.cpp.

## Q11g - Checkpoint Q8_0 GGUF (qwen4exp) - complete

- [x] Download del Q8_0 ufficiale (`unsloth/Qwen3.8-Flash-Next-GGUF`, 6 shard,
  175,3 GiB, SHA-256 verificati, read-only, manifest atomico) tramite il
  driver esteso `tools/import_qwen38_fp8.py --repo --gguf-dir`. Il checkpoint
  FP8 safetensors è stato **eliminato** su richiesta: lo store contiene ora il
  solo Q8_0.
- [x] Metadati GGUF letti e verificati: `general.architecture = qwen4exp`,
  1224 tensori (811 Q8_0, 389 F32, 24 BF16), tutti gli invarianti combaciano
  e i metadati portano anche rope_sections, compress_ratios per layer,
  ple.head_offsets/vocab_sizes, `embedding_length_per_layer_input = 160`.
- [x] `src/inference/qwen3_8_gguf.cpp`: decode Q8_0 (scala F16 + 32 int8),
  `qwen38_f16_to_f32`, apertura split-catalog validata, lettura tensore con
  verifica shape + dequant + conversione dal layout ggml `[in, out]`;
  `ssm_a` confermato F32 come A_log (range tutto negativo).
- [x] Mappatura completa gguf→operatori: `attn_qkv`→in_proj_qkv,
  `attn_gate`→in_proj_z, `ssm_alpha/beta`→in_proj_a/b, `ssm_out`→out_proj,
  `hc_attn/ffn_*`→gated residual, indexer **split** in q/k_proj (il fused
  index_qk è ricostruito per concat), esperti come **slice a stride fissa**
  dei tensori fused `[in, out, 512]`, embedding a colonna da `token_embd`.
- [x] Dispatcher automatico safetensors/GGUF in `qwen38_load_layer_trunk`,
  `qwen38_load_embed_row` e `qwen38_load_layer_expert`: tutti i CLI
  (`qwen38-layer`, `qwen38-prefill`, `qwen38-chain`) girano su entrambi.
- [x] Full-depth reale su Q8_0: 48/48 layer, 6,58 GB payload, exit 0.
  Sum (differiscono dal FP8 per la ricuantizzazione, atteso): layer-0 BOS
  6.65287, layer-3 17.7864, chain 0–3 replay bit-pari (23.0238).
- [x] Test opt-in biforcuto safetensors/GGUF; reconfigurato
  `VASTLOOM_QWEN38_MODEL` sul root Q8_0.

**Gate:** il runtime C++ nativo ora esegue il checkpoint **llama.cpp-compatible**
end-to-end (layer singoli, prefill, catena, full-depth) — la base per
l'oracolo llama.cpp e per la residenza esperti in VRAM.

## Q11h - Parità greedy con llama.cpp - complete

> **Oracolo esterno raggiunto**: il runtime nativo genera gli stessi token di
> llama.cpp sullo stesso checkpoint Q8_0. Fine dell'era self-pinned.

- [x] llama.cpp `6fe7498` clonato e compilato (CPU Release, MSVC);
  `llama-completion`, `llama-tokenize`, `llama-debug` (+ patch di dump
  opzionale via env, solo nel clone locale, non committata).
- [x] Debug sistematico per stadio con dump intermedi (`hc_norm`, `hc_mixed`,
  `attn_output`, `final_output`) e replica numpy — sei difetti trovati e
  corretti:
  1. **z-gate del DeltaNet: sigmoid, non silu** — llama: "the one numerical
     difference from Qwen3.5's GDN: sigmoid output gate, not silu"; il
     modeling HF main non lo porta (era il mio errore nel fix Q11c).
  2. **ssm_a pre-trasformato**: il GGUF porta già −exp(A_log) (range tutto
     negativo, verificato); il loader lo converte in log-space per l'operatore.
  3. **Norm weights già (1+w)** nel GGUF (mean ≈ 1 verificato): shift −1 al
     load, tranne `ssm_norm` (RMSNormGated, weight diretto).
  4. **Mapping k-heads a tile** (vh % k_heads, semantica ggml repeat),
     confermato dal kernel ggml `ik1 = iv1 % nek1`.
  5. **RoPE parziale** (64 su 256 dim, theta 1e7) su attention q/k e indexer
     q/pooled-keys + **PLE integrato** (hash n-gram → lookup tabella →
     key/value → gate → conv dilatata), mancanti nel Q7/Q10.
  6. **Sfasamento trunk nel greedy CLI**: il layer 0 girava col trunk 47 del
     token precedente (trunks.size()==1 sempre).
- [x] Verifica numerica stadio-per-stadio con input identico (hc_mixed di
  llama): qkv/conv ~0.01%, delta rule raw sum 0.8686 vs 0.8687 (0.01%),
  core post-norm con sigmoid ~0.2%.
- [x] **Parità greedy**: "The capital of France is" → " Paris. The capital of
  Germany is Berlin. The capital of Italy is Rome. The" (16 token) =
  identico a llama.cpp greedy. Secondo prompt ("Hello world") plausibile e
  deterministico.

**Gate:** il runtime nativo Vastloom produce gli stessi token di llama.cpp
sul checkpoint ufficiale Q8_0. L'oracolo esterno è stabilito per Q12+.

## Q11j - Throughput P2b-P3a - complete

- [x] Q8_0-native expert matvec + LRU O(1) + decode DeltaNet parallelo
  (`71b8a5a`): bit-identico, 33.75 → ~7 s/token.
- [x] Prefetch concorrente dei top-k esperti (`8e9c970`): ~4 s/token.
- [x] Trunk Q8_0-native in residenza, P2d (`19b3319`): startup residenza
  16,8 → 3,1 s; streaming senza override 194 → 32 s; parità invariata.
- [x] P3a `--cuda-experts` (`70a8e6b`): bit-identico ma MISURATO più lento
  del kernel CPU a 12 thread (FP64 1/64-rate + griglie vincolate dalla
  parità) → resta opt-in, il CPU kernel è la product path.

**Gate:** parità greedy 16/16 token mantenuta a ogni passo; numeri onesti in
AGENTS.md/CHANGELOG. ~3,3-4 s/token generato, contro 0,32 s/token di
llama.cpp (gap ~10x, strutturale: page cache vs cache nostre).

## Q11k - Prefill a chunk (PRIORITÀ sopra Q12/Q13)

Motivazione: il contesto dichiarato dal checkpoint è 262.144 token; gli
stati DeltaNet sono O(1) e i cache QSA crescono ~72 KB/token in F32 (128k ≈
9,4 GB, dimezzabili con cache F16 — fattibile in RAM), ma il prefill seriale
a 1 token (~3,5-4 s/token) rende il contesto lungo irraggiungibile in
pratica. Senza prefill veloce, Q12 (prefix-KV) e Q13 (serve) restano demo
corte.

- [x] Leva trovata prima del chunking: backend SIMD ggml OPT-IN per i
      matvec Q8_0 (`4b42627`): forward 6,1→1,7 s, expert 19,6→11,2 s su 8
      forward; parità 16/16 mantenuta sulla batteria esistente; il GDN
      chunked in riferimento scalare sarebbe NEGATIVO (O(C²·d) in scalare >
      ricorrenza seriale O(d) per token) → utile solo insieme ai GEMM SIMD.
- [x] DIFETTO D1 RISOLTO COME DIAGNOSI (`299988a`): bisezione di stage su
      "Once" — hc_norm pari a 1e-6, hc_gate deviazza 3,6e-2; la replica
      numpy di ggml_vec_dot_q8_0_q8_0 (attivazioni quantizzate Q8_0) chiude
      il gap 8,6x. Causa: llama.cpp quantizza le ATTIVAZIONI per ogni matvec
      con pesi Q8_0; il nostro percorso double è più preciso del reference e
      sui near-tie (margine 0,36 su quel prompt) può flippare. Non è un
      difetto dei kernel.
- [x] Decisione di parità chiusa (agosto 2026): la parità è strumento di
      verifica margin-aware (token identici dove il margine logit > soglia;
      near-tie = deviazione documentata perché il nostro percorso double è
      più preciso dell'oracolo che quantizza le attivazioni). La modalità
      `--llama-activation-quant` fu considerata e SCARTATA: non si clona
      l'arrotondamento dell'oracolo per parità.
- [ ] Q11l - ggml SIMD di default + lm_head Q8_0 (`output.weight` è Q8_0:
      660 MB residenti vs 2,5 GB F32, matvec SIMD) e batteria di parità
      margin-aware (tools/parity_battery.py, ≥6 prompt EN/IT con dump dei
      margini; exit-code gate). Misura onesta vs llama.cpp E ik_llama.cpp.
- [ ] Q11m - re-import esperti Q4_K (profilo nuovo accanto a Q8_0, gating
      da minimo-working-set; ~1,3 GB/token di traffico esperti) con decoder
      pinned-ggml già in repo.
- [ ] Q11n - tier VRAM model-aware: pinning esperti per statistiche di
      routing + prefetch layer N+1 durante N, sotto Governor. Obiettivo
      falsificabile: battere la page cache dell'OS sull'hit-rate esperti a
      parità di RAM - verificabile SOLO Vastloom può farlo; benchmark contro
      ik_llama.cpp (`--cpu-moe`/`-ot`) e mainline llama.cpp.
- [ ] Q14 - capability transfer: solo DESIGN + ladder falsificabile per ora
      (task suite fissa, frontman residente, wake-up selettivo N+1 via hash
      già risolto, ponti distillati con il tooling `distill_*` esistente).
- [ ] Prefill a chunk pilotato dal backend ggml: batch T token per layer
      (GEMM riutilizzano le righe Q8 in cache L2), conv/PLE allineati.
- [ ] Forma chunked della regola delta GDN solo se il profilo lo richiede.
- [ ] Prefill QSA batchato: punteggio indexer per più query per passata.
- [ ] Misura: throughput prefill token/s su prompt lunghi vs baseline
      seriale, bytes/token letti, in report dedicato.

**Gate:** parità margin-aware (token uguali dove il margine supera soglia,
near-tie documentati) su batteria ≥ 6 prompt × 24 token; replay
tolerance-dichiarata del prefill multi-token esteso; throughput misurato e
documentato.

## Q12 - prefix-KV MoE stream (porting M13)

- [ ] Estendere `LlamaSession` al percorso MoE stream.
- [ ] KV cache DeltaNet/QSA persistita attraverso richieste consecutive.
- [ ] Endpoint OpenAI compatibile come per M13 SmolLM2.

**Gate:** due turni consecutivi con cached_prefix_tokens > 0 sul MoE stream.

## Q13 - vastloom serve per Qwen3.8

- [ ] Tokenizer nativo text→id per qwen4exp (gpt2-BPE: `tokenizer.ggml.tokens`
      + `merges` + pre-tokenizer dal metadata GGUF), validato contro
      `llama-tokenize.exe` (oracolo) su fixture EN/IT.
- [ ] Framing ChatML dei messaggi (im_start/im_end dal vocab GGUF; il
      template jena nel file è decorativo, la forma è nota e pinnata).
- [ ] Dispatch `serve <directory>`: rilevamento Q8_0 GGUF qwen4exp →
      chat engine Qwen3.8 (residenza trunk, LRU esperti, opzione
      `--cuda-experts`); la generazione è il loop qwen38-greedy estratto
      dall CLI, con cancel/timeout dal pattern reduce_generate.
- [ ] Endpoint `/v1/models`, `/v1/chat/completions` (streaming + non-stream).
- [ ] Test Python opt-in via `tests/test_qwen38_cli.py`.

**Gate:** smoke test verde end-to-end. Nota di onestà: prima di Q11k la
sessione serve riprefilla ogni turno a ~3,5-4 s/token → demo con contesti
brevi; il contesto lungo realistico dipende da Q11k+Q12, non da Q13.

---

## Track archiviato

Il track M0–M13 (DeepSeek V4 Flash 0731) è congelato in `docs/archive/deepseek_0731/`.
I test C++ che difendono la parità pinned-ggml per IQ1_S/IQ2_XXS/Q4_K restano
compilati e verdi — sono contratti storici. Per il M14+ originale (multi-backend,
Kimi K3, Laguna, MiniMax, Vulkan/ROCm, multi-NVMe) see the archived roadmap preserved in git history (the previous content of this file, retrievable with `git log --follow docs/ROADMAP.md`).
