# Marcusina AI Platform — Design & Capability Map

**Audience:** AI/ML team + backend leads. **Status:** proposal for phase-1 build.
**Author:** AI/ML engineering. **Companion doc:** [`BACKEND-INTEGRATION-REFERENCE.md`](./BACKEND-INTEGRATION-REFERENCE.md) (the contract the Fastify team implements against).

---

## 1. Position of the AI service

The AI layer ships as **one standalone microservice**, hosted separately and reached
by the Core API (Fastify, `:3001`) over HTTP. It is a **sibling** of the existing
`videocall-service` / `voicecall-service`, behind the same Nginx gateway.

```
                 Nginx gateway (:8080)
                        │
   ┌──────────────┬─────┴───────┬──────────────┐
   ▼              ▼             ▼              ▼
Core API     Videocall      Voicecall      AI Service   ← we build/host this
(:3001)      (:3002)        (:3003)        (:8001)
 MongoDB      signaling      signaling      ├─ FastAPI (sync + enqueue)
 Redis        Redis          Redis          ├─ Celery workers (async)
                                            ├─ local LLM server (vLLM/llama.cpp)
                                            
                                            ├─ self-hosted models (ASR/NLP/embed)
                                            └─ Postgres + Redis (our state only)
```

**Everything runs on open-weight models we host ourselves. No external API, no
API keys, no per-call fees, no data leaving our infrastructure.** See §4.

**Hard boundaries (non-negotiable):**

- The AI service is **stateless with respect to Marcusina's data.** It receives a
  request, returns a result, and the Core API persists it to **MongoDB**. We do
  **not** read or write their database. Our Postgres holds only our own inference
  metrics / audit and any RAG index — never a copy of patient records.
- **All inference is local.** The LLM runs on a model server inside our own
  deployment (localhost). Nothing is sent to a third-party AI provider — which
  for a regulated health product is a feature, not just a cost decision (§6).
- **PHI minimization:** the backend sends only the fields a task needs (a transcript,
  a claim, a symptom string) — never full patient records. See §6.

---

## 2. Two lanes: sync vs async

Different AI tasks have opposite latency profiles. We expose **both**, and the
backend picks per call site.

| Lane | Used for | Shape | Latency |
|------|----------|-------|---------|
| **Sync** | Anything that must gate a user action: chat moderation, distress detection, fast triage | `POST … → 200` with the result inline | < ~400 ms |
| **Async** | Anything slow or bursty: transcription, SOAP notes, summaries, recommendations, RAG misinfo | `POST … → 202 {task_id}` then **callback** to Fastify (or poll) | seconds–minutes |

The async lane already matches the backend's BullMQ/queue worldview. The callback
contract (`FASTIFY_CALLBACK_URL` + `X-Callback-Secret`) already exists in our config —
the backend just needs to expose the receiving route (see the integration reference).

---

## 3. Capability map — where AI plugs into the backend

Two engines power everything: **Claude (LLM lane)** for language understanding,
generation, and reasoning; **self-hosted models** for speech, fast classification,
and embeddings. Ranked by phase.

### Tier 1 — text-only, no blockers, highest value (build first)

| # | Capability | Backend touchpoint | Engine | Lane | Safety posture |
|---|-----------|--------------------|--------|------|----------------|
| 1 | **Triage / urgency + specialty routing** | `appointments.controller` create (`consultation_reason`) | rules (`red_flags.json`) + ML advisory + local-LLM rationale | sync | Rules net **always** overrides; ML/LLM advisory only |
| 2 | **Distress / self-harm detection** | `msgAndComs/comm.js` `sendMessage`, posts/comments | fast classifier (high-recall) → local-LLM escalation on ambiguous | sync | Escalates to a **human**, never auto-acts |
| 3 | **Content moderation** (toxicity/abuse) | posts, comments, chat, `userReport` | fast classifier → local LLM for nuanced cases | sync | Auto-hold + human review; precision-tuned |
| 4 | **Health misinformation check (RAG)** ✅ | posts/comments, community content | local LLM + retrieval over WHO/CDC/PubMed, **with citations** | async | **Advisory flag for human review**, never auto-removal |
| 5 | **Sentiment** | reviews, support tickets, messages | self-hosted (cardiffnlp) | sync | Analytics/routing only |
| 6 | **Support assist** (draft replies, routing) ✅ | `support/contactUs`, `answers` | local LLM | async | Draft for staff, human sends; no medical advice; distress-net override |

### Tier 2 — high value, needs a media decision

| # | Capability | Backend touchpoint | Engine | Blocker |
|---|-----------|--------------------|--------|---------|
| 7 | **Consultation transcription** | voice/video consultations | Whisper large-v3 (self-hosted) | **No server-side audio** — video/voice services are WebRTC signaling only. Needs client-side recording upload, an SFU that forks audio, or post-call upload. |
| 8 | **SOAP note generation** 🟡 built | post-consultation | local LLM (JSON-constrained) from transcript; folds in entity extraction; ICD grounded in config | Generation done; **input** depends on #7 |
| 9 | **Patient-friendly visit summary** 🟡 built | post-consultation | local LLM | Generation done; **input** depends on #7 |

### Tier 3 — needs data / later

| # | Capability | Touchpoint | Engine |
|---|-----------|-----------|--------|
| 10 | **Content recommendations / feed ranking** ✅ | social feed | embeddings + in-process vector store (self-hosted) |
| 11 | **Semantic search** (meds, providers, content) ✅ | search endpoints | embeddings |
| 12 | **Medication / drug-interaction check** ✅ | `presAndMeds/prescriptions` | curated deterministic rule base (authoritative) + optional LLM advisory, **clinician-confirmed** |
| 13 | **Pre-consultation symptom intake assistant** ✅ | new conversational surface | red-flag triage net (authoritative) + local LLM structuring; advisory, not diagnosis |

---

## 4. Models — fully self-hosted, open weights

**No external API, no keys, no per-call fees, no data egress.** Every model is
open-weight and served from inside our own deployment. Most of the stack is
already this way (Whisper, embeddings, sentiment, NER, ONNX classifiers); this
section is mainly about the one new piece — a **local LLM** for the generative
tasks (SOAP, summaries, misinfo reasoning, support drafts, nuanced moderation).

### 4.1 The non-LLM models (already open + self-hosted)

| Task | Open model | Status |
|------|-----------|--------|
| ASR / transcription | `Systran/faster-whisper-large-v3` | ✅ downloaded |
| Embeddings (RAG, recommend, search) | `all-MiniLM-L6-v2` (→ upgrade to `BAAI/bge-base-en` for quality) | ✅ have MiniLM |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | ✅ eval 0.70 |
| Toxicity / moderation | `unitary/toxic-bert` or `martin-ha/toxic-comment-model` | to add |
| Biomedical NER | `d4data/biomedical-ner-all` → or `scispaCy en_ner_bc5cdr_md` | ⚠️ recall low; fold into SOAP LLM |

### 4.2 The local LLM (the new piece)

**Chosen deployment (2026-06-18): cloud GPU, rented hourly.** Since GPU-hours
cost money, the architecture is built to **minimize them** by splitting lanes:

- **Always-on CPU (free): the sync / safety path** — moderation, distress, triage,
  sentiment, embeddings. Small models; must never depend on a GPU being up.
- **On-demand GPU (vLLM, paid hourly): the generative tasks** — SOAP, summaries,
  RAG-misinfo, support drafts. All **async**, so they tolerate spin-up and **batch**,
  letting us rent the GPU in bursts instead of 24/7.

Develop on CPU (cached Mistral-7B) for free; flip to the cloud vLLM endpoint with a
**single URL change** — both speak the same OpenAI-compatible interface, so the
model/runtime is config, never code.

Served by an **OpenAI-compatible** model server, so our code calls
`http://<host>:<port>/v1/...` — same shape as an API client, but **no key and
no egress**. Two runtimes depending on hardware:

| Runtime | Hardware | When |
|---------|----------|------|
| **Ollama / llama.cpp** (GGUF) | CPU or modest GPU | Dev / low-resource / no GPU. **You already have `Mistral-7B-Instruct` GGUF cached.** |
| **vLLM** (HF/AWQ/GPTQ) | GPU | Production throughput. You also have the Mistral GPTQ cached. |

**Model candidates (pick by hardware — see the question at the end):**

| Tier | Open model | Footprint | Notes |
|------|-----------|-----------|-------|
| Starter (CPU-friendly) | `Mistral-7B-Instruct` (have it), `Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct` | ~4–5 GB Q4 | Runs on CPU (slow) or any GPU. Good enough for advisory drafts. |
| Medical-tuned | `BioMistral-7B`, `OpenBioLLM-8B`, `Meditron-7B`, `Med42-v2-8B` | ~5–8 GB | Better clinical vocabulary; still advisory + human-reviewed. |
| High-quality (needs GPU) | `Llama-3.1-70B`, `OpenBioLLM-70B`, `Med42-70B` (4-bit) | ~40 GB | Closest to frontier quality; needs a 48 GB GPU or two 24 GB. |

Implementation notes:
- **JSON-constrained decoding** (Outlines / llama.cpp grammars / vLLM guided JSON)
  for SOAP and any structured result — the open-model equivalent of structured outputs.
- **RAG over a local corpus** (WHO/CDC/PubMed snapshots in our Postgres+pgvector or
  FAISS) grounds misinfo/medication answers with citations — this matters *more*
  with a smaller model, because retrieval carries the facts the model lacks.
- A single internal `app/llm/` client wraps the local server so swapping models
  (or sizes) is a config change, never a code change.

### 4.3 Honest tradeoffs (so nobody is surprised)

- **Quality:** a 7–8B open model writes weaker clinical prose than a frontier model.
  That's acceptable here **only because every clinical output is advisory +
  human-reviewed** (§6) — never autonomous. We measure each with the `eval/` harness.
- **"Self-hosted" ≠ free:** it trades per-call fees for **compute**. CPU works for
  prototyping (slow, like Whisper today); real-time/scale wants a GPU eventually.
  But the cost is **fixed, owned, and predictable** — no metered key.
- **Build-our-own:** training from scratch is out (millions $, huge data).
  **Fine-tuning** an open model on Marcusina's own data is the realistic path —
  but *later*, once labeled data exists (none yet). Start off-the-shelf + RAG.

---

## 5. What we keep vs reshape (existing service)

The current FastAPI + Celery + ONNX + Postgres skeleton is **sound** — async
enqueue + callback already matches the backend. We **keep** it and **add** a
local-LLM lane. We do **not** start from scratch.

| Keep | Reshape / add |
|------|---------------|
| FastAPI gateway, Celery workers, Redis, callback pattern | Add an **LLM module** (`app/llm/`) wrapping a **local OpenAI-compatible model server** (JSON-constrained decoding, retries) — no external SDK/key |
| Whisper large-v3 (upgraded), sentiment, embeddings | **Replace** misinfo classifier (failed eval: 0.557) with **RAG + local LLM**; **upgrade** NER (failed eval: 0.36) or move entity extraction into the SOAP LLM call |
| Eval harness (`eval/`) | Extend to score LLM outputs (local LLM-as-judge + rubric) and gate each capability |
| Postgres for our metrics/audit | Add a **vector index** (Postgres+pgvector or FAISS) for the RAG trusted-source corpus |

> The two failed evals from prior work drive this: misinfo isn't a classifier
> problem (→ retrieval), and NER recall is too low to trust standalone (→ fold into
> the LLM SOAP extraction, which also has the transcript context).

---

## 6. Safety, compliance, and the go-live gate

This is a **real telehealth product**, likely a regulated medical device
(FDA SaMD / EU MDR / MHRA). The intelligence layer is built around that.

1. **Human-in-the-loop for everything clinical.** SOAP notes, triage, misinfo,
   drug checks are **advisory**. A clinician/moderator confirms before anything
   acts. Distress detection **escalates to a human**, never auto-responds.
2. **Deterministic safety nets win.** `red_flags.json` (triage) and the
   high-recall distress classifier override any ML/LLM score. ML optimizes
   *recall* for safety-critical detection; moderation optimizes *precision*.
3. **PHI handling.** Because **all inference is local**, no patient data ever
   leaves Marcusina's infrastructure — there is no third-party AI provider in the
   data path, and **no BAA to negotiate**. This is a direct benefit of the
   self-hosted design. Still send minimal fields, strip identifiers where
   possible, and keep patient content out of our Postgres (metrics/audit only).
4. **No model ships without a score.** The `eval/` harness gates each capability
   against a per-task target (see its README). Public benchmarks select
   candidates; **Marcusina's own labeled data + clinical sign-off certify
   go-live** — that gate is not yet met and must not be skipped.
5. **Audit everything.** Every inference (model, version, score, latency) is
   logged for compliance forensics.

---

## 7. Recommended build order

1. **Local LLM serving layer** (`app/llm/`) — stand up Ollama/llama.cpp (or vLLM)
   with the model you already have cached, behind one internal client. Unblocks all LLM work.
2. **Triage + distress + moderation (Tier 1, sync)** — highest safety value, no blockers.
3. **RAG misinfo check** — replaces the model that failed eval; real value.
4. **Resolve the audio path** with the backend team → transcription → **SOAP/summaries**.
5. **Recommendations + semantic search** (embeddings/FAISS).

Each step ships with: the endpoint, an eval score, and the backend contract entry.
```
