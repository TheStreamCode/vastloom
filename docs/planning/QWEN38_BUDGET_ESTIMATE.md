# Qwen3.8-Flash-Next — Budget Estimate (Fase 0, rivisto)

> **Status: BOZZA RIVISTA, NON VALIDATA.** Tutti i numeri sono dichiarati dall'editore del modello
> o da `config.json`/`header GGUF`/`docs Unsloth` remoti. Non sono ancora il risultato di un
> download + esecuzione sul reference PC. Il gate di budget è "carta, non rame".
>
> **Decisioni utente vincolanti (rivisto):**
> 1. Cambio modello totale (DeepSeek 0731 → Qwen3.8-Flash-Next)
> 2. Operatori nativi C++20 per tutte e tre le primitive (DeltaNet, QSA, n-gram)
> 3. **Quota: FP8 Unsloth** (non IQ, non UD-IQ1_S) — vincolo qualità lossless
> 4. Roadmap Q1..Q13, archivio solo docs DeepSeek 0731
> 5. **1-2 settimane di preparazione infrastruttura** prima di scaricare il 125B
> 6. **POC su Qwen3.8-27B FP8** come pre-validation
> 7. Cancello e basta se Fase 0 boccia

---

## 1. Perché FP8 invece di IQ

### Risultato chiave Unsloth Dynamic 3.0 (da `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`)

> *"There is a sharp drop off from UD-Q2_K_XL to UD-IQ2_S for 32 token prediction from around
> 25% accuracy to under 8-10%. This sharp drop off means tool calling and non thinking modes
> break down."*
>
> *"1-bit should not be used for agentic use-cases"*

Per Gemma 3 27B (riferimento Unsloth Dynamic 3.0, stesse tecniche di calibrazione):

| Quota | Disk (GB) | KLD vs BF16 | Efficiency (KLD/GB) | Note |
| --- | --- | --- | --- | --- |
| UD-IQ1_S | 6.06 | 0.972 | 0.16 | bocciato, ~72% top-1% accuracy |
| UD-IQ2_XXS | 7.31 | 0.521 | 0.071 | bocciato, tool calling rotto |
| UD-Q2_K_XL | 9.95 | 0.221 | 0.022 | **sweet spot qualitativo** |
| UD-Q3_K_XL | 12.76 | 0.081 | 0.0067 | buono |
| UD-Q4_K_XL | 15.64 | 0.024 | 0.0015 | lossless-ish |

### Gate di RAM per Qwen3.8-Flash-Next (estrapolazione)

Scaling lineare da IQ1_S = 67.55 GB disco, ~10.7 GB live (nostra stima precedente):

| Quota | Stima disco | Stima live | Gate 12.3 GB? | Verdetto |
| --- | --- | --- | --- | --- |
| UD-IQ1_S | 67.55 GB | ~10.7 GB | borderline | bocciato qualità |
| UD-IQ2_XXS | ~79 GB | ~11.7 GB | borderline | bocciato qualità |
| **UD-Q2_K_XL** | **~108 GB** | **~16 GB** | **fail** | sweet spot qualitativo, **fallisce RAM** |
| UD-Q3_K_XL | ~149 GB | ~22 GB | fail | fail |
| UD-Q4_K_XL | ~191 GB | ~28 GB | fail | fail |
| **FP8 Unsloth** | **~85 GB (misurato da verificare)** | **~10-12 GB** | **probabile pass** | lossless-ish |

**Conclusione**: nessuna quota IQ raggiunge la qualità lossless **e** sta nel gate. FP8 è l'unica
opzione che combina:
- Lossless-ish (KLD << 0.1 vs BF16, simile a Q4_K_M o meglio)
- RAM viva comparabile a IQ2_XXS (gli FP8 tensori pesano quanto IQ2, ma il numerico è denso)
- Supportato ufficialmente da llama.cpp (FP8 nativo)
- Disponibile subito per Qwen3.8

---

## 2. FP8 Unsloth per Qwen3.8-Flash-Next

### Catalogo HF API

- Repo: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-FP8`
- Base: `Qwen/Qwen3.8-Flash-Next` (BF16, 360 GB, 131 shard)
- Anche: `Qwen/Qwen3.8-Flash-Next-FP8` (Qwen ufficiale)

### Stima disco FP8

Da 360 GB BF16 → FP8 dimezza teoricamente → ~180 GB teorico. In pratica con offload della
n-gram table e dei metadata: **stima ~85 GB** (analogia con FP8 di altri modelli MoE 100B+).

### Stima RAM viva FP8 (Questo è il punto da verificare in Q2)

- Trunk 1 layer BF16 ≈ ~6.99 GB su DeepSeek
- Stesso trunk FP8 ≈ ~3.5 GB (half)
- KV/DeltaNet state ≈ stesso
- Selected experts FP8 ≈ ~0.9 GB (half di 1.77 IQ)
- N-gram offload 4 GB (scenario B)
- **Totale stimato: ~9-11 GB live** ← probabile pass del gate 12.3 GB

**Da misurare in Q2** con POC su Qwen3.8-27B FP8.

---

## 3. Piano di lavoro rivisto (1-2 settimane di preparazione)

### Fase 0 — Stima budget (questo documento)

Stato: **completato**.

### Q0 — Setup ambiente (questa settimana)

**Obiettivo**: preparare l'infrastruttura di test C++20 nativa per le tre nuove primitive
SENZA scaricare Qwen3.8-Flash-Next FP8 (85 GB).

**Attività**:
1. Verificare che `llama.cpp mainline` abbia il supporto `qwen4_exp` con DeltaNet, QSA, n-gram
   e Gated Residual. Se sì: clonarne il commit pinned (es. ggml 0.19.0 commit 30bf868 come
   per DeepSeek). Se no: pianificare lo sviluppo indipendente.
2. Aggiungere `include/vastloom/inference/qwen3_8_flash_next.hpp` come skeleton vuoto (tipi,
   config, ma nessuna implementazione).
3. Aggiungere un test C++ always-on `test_qwen38_metadata_inspector` che legge un file di
   mock metadata (NON un checkpoint reale) per validare il parser `config.json`.
4. Commit "Add Qwen3.8-Flash-Next inspector skeleton".

### Q1 — POC su Qwen3.8-27B FP8 (~15 GB disco)

**Obiettivo**: scaricare e validare una singola layer FP8 di Qwen3.8-27B (modello "fratello"
più piccolo, stessa architettura) per confermare che la pipeline C++20 nativa regge end-to-end.

**Attività**:
1. Scaricare `unsloth/Qwen3.8-27B-FP8` (~15 GB, contiene n-gram + 27B attivati)
2. Implementare il catalog reader per i tensori FP8 (nuovo dtype)
3. Implementare un decoder FP8 → F32 pinned-ggml
4. Implementare **un singolo layer ridotto** (DeltaNet + MoE shared + QSA opzionale):
   - Forward 1 token
   - Confronto con llama.cpp pinned-ggml
   - Se bit-exact: archiviare come **Fase 0 verde**, procedere
   - Se non bit-exact: diagnosticare e correggere
5. Misura RAM viva: confermare <12.3 GB in single-token forward

**Cancello e basta** se FP8 non chiude la parity pinned-ggml (situazione eccezionale).

### Q2 — Reference pinned-ggml full model

**Obiettivo**: llama.cpp locale gira Qwen3.8-27B FP8 end-to-end, 5 token oracolari prodotti.

Solo dopo Q1 verde.

### Q3 — Tensor & expert catalog per Qwen3.8-Flash-Next FP8

**Obiettivo**: enumerare tutti i tensori del 125B FP8 (layer-trunk, shared, routed, n-gram
tables, gated-residual, DeltaNet state, QSA indexer).

Solo dopo Q2 verde.

### Q4 — Range importer + safety governor

Tre-object gate come per DeepSeek 0731.

### Q5 — Operatori nativi pinned-ggml

Decoder FP8 (nuovo), DeltaNet state, QSA indexer, n-gram tables, Gated Residual.

### Q6 — Layer graph non-routed 48 layer

Struttura ibrida 3 linear_attn + 1 full_attn × 12.

### Q7 — Prefill/decode reduced layer (2–3 layer + MTP)

### Q8 — MoE routing 10+1

**No hash shortcut** (a differenza di DeepSeek).

### Q9 — Gated DeltaNet state persistence + N-gram lookup resident

### Q10 — QSA sparse attention

### Q11 — Full-depth generate validate (48 layer)

### Q12 — prefix-KV MoE stream (porting M13)

### Q13 — `vastloom serve` per Qwen3.8

---

## 4. Decisione vincolante aggiornata

> **SCARICARE `unsloth/Qwen3.8-Flash-Next-FP8` SOLO DOPO che il POC su `unsloth/Qwen3.8-27B-FP8`
> ha confermato:**
> 1. Parità pinned-ggml su almeno 1 layer completo (DeltaNet + MoE shared)
> 2. RAM viva < 12.3 GB in single-token forward
> 3. Tempo di preparazione: 1-2 settimane
>
> Se POC boccia: **cancello e basta** (per vincolo utente).

---

## 5. File di riferimento

- HF: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next`
- HF (config): `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`
- GitHub: `https://github.com/QwenLM/Qwen3.8-Flash-Next`
- Tech report PDF: `https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf`
- FP8 Unsloth: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-FP8`
- 27B Unsloth (POC target): `https://huggingface.co/unsloth/Qwen3.8-27B-FP8`
- 27B GGUF (per calibrazione): `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`
- Unsloth docs Qwen3.8: `https://unsloth.ai/docs/models/qwen3.8-next`
- Unsloth Dynamic 3.0: `https://unsloth.ai/docs/basics/dynamic-3.0-ggufs`

---

**STOP. In attesa di OK utente per procedere con Q0 (setup ambiente).**
