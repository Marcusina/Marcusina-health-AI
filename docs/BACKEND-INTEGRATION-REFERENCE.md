# Backend Integration Reference — Marcusina AI Service

**Audience:** Fastify / Core API team. **Companion:** [`AI-PLATFORM-DESIGN.md`](./AI-PLATFORM-DESIGN.md).

This is the contract to build against. The AI service is a separate hosted
microservice; you call it over HTTP. Nothing here requires changing your
database model — you call us, we return a result, **you** persist it to MongoDB.

> Endpoint paths and payloads below are the **target contract**. A few already
> exist in the service under slightly different paths; treat this doc as the
> source of truth and we will align the service to it.

---

## 1. Networking & auth

- **Base URL:** `AI_SERVICE_URL` (e.g. `https://ai.marcusina.internal` or, via the
  gateway, `https://api.marcusina.dev/services/ai`). Route `/services/ai/*` →
  `ai-service:8001` in Nginx, same pattern as `/services/video`.
- **All endpoint paths are under `/api/v1`** (e.g. `POST /api/v1/triage`). The
  examples below show the path after the base URL.
- **Auth (backend → AI):** every request carries the header
  `X-AI-Secret: <shared secret>` (matched against the AI service's `API_SECRET_KEY`).
  Requests without it get `401`; wrong value gets `403`.
- **Auth (AI → backend callback):** every callback carries
  `X-Callback-Secret: <FASTIFY_CALLBACK_SECRET>`. Your callback route must verify it.
- **Identity:** forward the gateway's `x-user-id` / `x-user-role` so we can audit
  who triggered an inference. We never use them for authorization — that's yours.

```
# .env additions (Core API)
AI_SERVICE_URL=https://ai.marcusina.internal
AI_SERVICE_KEY=<shared secret, also set on the AI service>
FASTIFY_CALLBACK_URL=https://api.marcusina.dev/internal/ai-callback
FASTIFY_CALLBACK_SECRET=<shared secret, also set on the AI service>
```

---

## 2. Two interaction patterns

### A) Sync — for actions you must gate (moderation, distress, fast triage)

Call, block briefly (< ~400 ms), act on the inline result.

### B) Async — for slow/bursty work (transcription, SOAP, summary, recommend, misinfo)

1. You `POST` the job → we return `202 { task_id }`.
2. We process on our workers.
3. We **POST the result to your callback** `FASTIFY_CALLBACK_URL` (preferred), **or**
   you poll `GET /v1/tasks/{task_id}`.

Use the callback for everything; polling is the fallback if a callback is missed.

---

## 3. Sync endpoints

### `POST /api/v1/moderate/text` — toxicity + distress in one call ✅ implemented

Use on `sendMessage`, post/comment create, `userReport`.

```jsonc
// request
{
  "text": "the user-generated content",
  "context": "chat|post|comment|review|report",  // optional, default "post"
  "entity_id": "msg_123",                          // optional, for your correlation
  "deep_scan": false        // optional: also run the LLM for nuanced toxicity (slower; use for reports/borderline)
}
// 200 response
{
  "action": "allow | flag | block",         // your gate: allow=deliver, flag=deliver+queue review, block=hold
  "toxicity": { "score": 0.02, "label": "clean | toxic | harassment", "matched": [] },
  "distress": {                               // safety-critical
    "detected": false,
    "severity": "none | low | high",
    "escalate_to_human": false,              // if true: route to crisis/clinical workflow
    "matched": []                            // distress phrases that fired (audit)
  },
  "llm_used": false,                          // whether the LLM was consulted this call
  "model_version": "moderation-rules+llm/2026.06"
}
```

**Rules:**
- If `distress.escalate_to_human` is `true`, route to your human crisis workflow
  **regardless** of `action`. Distress detection never auto-replies.
- Toxicity is precision-oriented: curated harmful phrases `block`; otherwise the
  content is `allow`ed unless `deep_scan` + the LLM judge it toxic.
- Distress always uses the LLM when a pattern fires; if the LLM is unreachable it
  **fails safe** (escalates) rather than missing.

### `POST /api/v1/triage` — urgency + specialty routing ✅ implemented

Use on `appointments.controller` create.

```jsonc
// request
{ "symptoms": "chest pain radiating to left arm", "age": 54,
  "medical_history": ["hypertension"], "patient_id": "pat_1", "use_llm": true }
// 200 response
{
  "urgency_level": "emergency | urgent | semi_urgent | non_urgent | self_care",
  "urgency_score": 1.0,
  "red_flag_symptoms": ["chest pain"],       // deterministic safety net — if non-empty, treat as emergency
  "recommended_specialty": "Pulmonologist",
  "reasoning": "Red-flag symptom(s) detected: chest pain. ...",
  "self_care_advice": null,                  // populated only when level == self_care
  "llm_used": false,
  "model_version": "triage-rules+llm/2026.06"
}
```

**Rule:** `red_flag_symptoms` is authoritative for emergency routing — when
non-empty the result is always `emergency` and the LLM is not even consulted. The
LLM urgency opinion is advisory and only used for non-red-flag cases.

### Semantic search & recommendations ✅ implemented (sync, CPU)

The AI service keeps its own content index (embeddings) — you **push** content to
it, it never reads your DB.

```jsonc
// Populate / update the index (call on content create/update):
POST /api/v1/content/index
{ "items": [ { "id": "post_1", "text": "title + body to make searchable",
              "type": "article|guide|post", "metadata": { "title": "...", "topic": "..." } } ] }
// → { "indexed": 1, "total": 153 }

POST /api/v1/content/remove   { "ids": ["post_1"] }    // on delete

// Semantic search:
POST /api/v1/search
{ "query": "how to manage blood sugar", "k": 10, "content_type": "article" }   // type optional
// → { "query": "...", "results": [ { "content_id", "type", "title", "score", "metadata" } ], "count": 10 }

// Recommendations (build from the user's profile):
POST /api/v1/recommend
{ "user_interests": ["diabetes"], "user_conditions": ["hypertension"],
  "context": "after_consultation", "k": 10, "exclude": ["post_9"] }
// → { "recommendations": [ { "content_id", "type", "title", "reason", "score" } ], "strategy": "content_based|trending" }
```

`strategy: "trending"` is the fallback when the profile is empty or the index has
no content yet. `context: "after_consultation"` boosts explanatory articles/guides.
(Note: the index is per-process for now; multi-worker deployments need a periodic
rebuild or a shared store — see the design doc.)

---

## 4. Async endpoints (enqueue → callback)

All return `202 { "task_id": "...", "status": "queued" }`. Pass `callback_url`
(defaults to your configured `FASTIFY_CALLBACK_URL`) and an idempotency key.

| Endpoint | Purpose | Key request fields |
|----------|---------|--------------------|
| `POST /api/v1/consultation/transcribe` | audio → transcript (Whisper); optional stereo speaker-split | `session_id`, **exactly one of** `audio_base64` / `audio_url`, `audio_format`, `language?`, `diarize_stereo?`, `channel_roles?` |
| `POST /api/v1/consultation/soap-note` ✅ | transcript → SOAP + entities + ICD (local LLM) | `transcript`, `session_id`, `patient_id`, `doctor_id`, `specialty?` |
| `POST /api/v1/consultation/summary` ✅ | transcript → patient-friendly summary | `transcript`, `session_id` |
| `POST /api/v1/misinfo/check` ✅ | health claim → verdict + citations (RAG) | `text`, `entity_id`, `k?` |
| `POST /api/v1/support/assist` ✅ | support ticket → routing + draft reply (local LLM) | `ticket_id`, `subject`, `message`, `category_hint?` |

```jsonc
// example: POST /v1/consultations/soap-note
{
  "transcript": "Doctor: what brings you in... ",
  "session_id": "sess_123",
  "patient_id": "pat_456",
  "idempotency_key": "soap:sess_123",        // dedupes retries
  "callback_url": "https://api.marcusina.dev/internal/ai-callback"  // optional override
}
// → 202
{ "task_id": "task_789", "status": "queued" }
```

**Misinfo callback `result` shape** (advisory — flagged claims go to human review,
never auto-removal):

```jsonc
{
  "claim": "vaccines cause autism",
  "verdict": "supported | contradicted | unsupported | not_health_claim | unverified",
  "confidence": 0.95,
  "rationale": "The retrieved WHO/CDC evidence contradicts the claim.",
  "flag": true,                  // convenience gate: true ⇒ route to human review
  "needs_human_review": true,
  "citations": [ { "source": "WHO", "url": "https://...", "snippet": "…", "score": 0.91 } ],
  "model_version": "misinfo-rag/2026.06"
}
```

`unverified` means the LLM was unavailable — the claim was **not** judged; route to
human review (the retrieved evidence is included as candidate citations).

### The consultation pipeline (audio → transcript → notes)

These three are **separate async jobs** chained by your backend, with the
**transcript as the shared hub**. One ASR step produces the transcript; the SOAP
note and the patient summary are two **independent** LLM calls that both take that
transcript — so you can fire them **in parallel** after transcription.

```
audio ──/transcribe──▶ transcript ──┬──/soap-note──▶ SOAP note (clinician signs)
                                     └──/summary────▶ patient summary
```

**Transcribe callback `result`:**

```jsonc
{
  "success": true,
  "session_id": "sess_123",
  "transcript": "what brings you in today ... i've had a cough for three days ...",
  "language_detected": "en",
  "language_confidence": 0.991,
  "confidence": 0.94,            // rough ASR confidence (1 − avg no-speech prob)
  "duration_seconds": 312.5
}
```

**Speaker-labeled transcript (stereo split — we handle it):** if you record in
stereo (practitioner → left channel, patient → right), send
`diarize_stereo: true` + `channel_roles: ["doctor", "patient"]` (= `[left, right]`).
We split the channels, transcribe each, and return a labeled, time-ordered
transcript plus structured segments:

```jsonc
{
  "success": true, "session_id": "sess_123",
  "transcript": "doctor: what brings you in today\npatient: i've had a cough...",
  "segments": [
    { "speaker": "doctor",  "start": 0.0, "end": 3.2, "text": "what brings you in today" },
    { "speaker": "patient", "start": 3.4, "end": 7.1, "text": "i've had a cough..." }
  ],
  "speakers": ["doctor", "patient"],
  "language_detected": "en", "duration_seconds": 312.5, "diarized": true
}
```

If the file turns out to be mono, we transparently fall back to the flat
transcript (`diarized: false`, no `segments`). No diarization model needed — the
channels carry the speaker identity.

Orchestration: on the transcribe callback, persist the transcript, then enqueue
`/soap-note` **and** `/summary` with `transcript = result.transcript`. (You don't
have to run all three — stop after the transcript if that's all you need.)

> **Audio formats:** any common audio/video container decodes — `webm`/opus
> (browser `MediaRecorder` default), `ogg`, `m4a`/`mp4` (AAC), `wav`, `mp3`,
> `flac`. The `audio_format` field is a hint; the decoder auto-detects. Image
> formats (jpg, **avif**, png) are not audio and will fail to decode.

> **Speaker labels (Doctor vs Patient):** Whisper transcribes words, not who
> spoke them. The clean way to get labels — **without any diarization model** — is
> to **record in stereo** (practitioner on the left channel, patient on the right)
> and send `diarize_stereo: true`. **We do the channel split and per-channel
> transcription server-side** and return the labeled transcript above. The client
> still records *one* file from *one* device (see §8); it just keeps the two
> voices on separate channels. A single *mixed* track can't be labeled this way —
> that would need a diarization model (`pyannote`, GPU) — so use stereo.

### Entitlement — who gets SOAP / summary (your decision, not ours)

**The AI service knows nothing about subscriptions or billing** — it just does the
work when you call it. **Whether** to call an endpoint for a given user, and **who
receives** the output, is enforced by your backend (which has the subscription
data). That keeps billing logic where it belongs and keeps the AI service simple.

Concretely, for your tiers:

| Output | Who | How you gate it |
|--------|-----|------------------|
| Transcript + **SOAP note** (professional's record) | **Professionals — automatic** | Always call `/transcribe` + `/soap-note` after a consultation |
| **Patient-facing summary** | **Paid subscribers only** | Only call `/summary` (or only deliver its result to the patient) when the patient's tier allows it |

Cheapest approach: **don't call `/summary` at all** for non-entitled patients — you
save the compute. Or call it and gate delivery; your choice. Either way it's a
one-line `if (patient.tier === "paid")` in your orchestration — no change on our
side.

**SOAP callback `result`** (a draft for the **clinician to review and sign** — never
final): `{ soap_note: {subjective, objective, assessment, plan}, extracted_entities:
{medications, diagnoses, symptoms, procedures, vitals}, icd_suggestions: ["E11", …],
llm_used, degraded, model_version }`. ICD codes are mapped from a config table, not
invented by the model. `degraded: true` means the LLM was down and the draft is a
clearly-marked placeholder for the clinician to complete.

**Summary callback `result`** (for the patient): `{ summary, next_steps[],
when_to_seek_help[], disclaimer, llm_used, degraded, model_version }`.

**Support-assist callback `result`** (a **draft** — a human agent reviews/edits/sends):
`{ category, priority: "low|normal|high|urgent", summary, draft_reply,
suggested_actions[], distress_flag, needs_human_review: true, llm_used, degraded,
model_version }`. The draft never gives medical advice (clinical questions are
routed to a clinician). `distress_flag: true` forces `priority: "urgent"` and an
escalation action — this runs even when the LLM is down.

> **Transcription is blocked until audio is available.** Video/voice services are
> WebRTC signaling only — there is no server-side audio stream, so something has to
> capture the consultation audio first. **See §8 (Audio path) for the recommended
> design.** Transcription → SOAP → summary (#7–#9) can't ship until that's in place.

---

## 5. The callback (AI → your backend)

Expose **one** route. We POST the finished result here.

```jsonc
// POST {FASTIFY_CALLBACK_URL}   header: X-Callback-Secret: <secret>
{
  "task_id": "task_789",
  "task_type": "soap_note",
  "status": "succeeded | failed",
  "result": { /* task-specific, e.g. the SOAP note */ },
  "error": null,
  "entity": { "type": "session", "id": "sess_123" }   // for routing to the right record
}
```

Respond `2xx` to ack. We retry non-2xx with backoff. Dedupe on `task_id`
(callbacks may arrive at-least-once).

---

## 6. Error model

| HTTP | Meaning | You should |
|------|---------|------------|
| `200` | sync result | act on it |
| `202` | async accepted | await callback / poll |
| `400` | bad request (missing field) | fix payload; don't retry blindly |
| `401` | bad/missing `AI_SERVICE_KEY` | check secret |
| `422` | unprocessable (e.g. empty transcript) | surface to caller |
| `429` | rate limited | honor `Retry-After` |
| `503` | model warming / dependency down | retry with backoff |

Async failures come back via the callback with `status: "failed"` + `error`, not
as an HTTP error.

---

## 7. Fastify-side: the two pieces you add

### a) A thin AI client

```js
// src/services/ai.service.js
import axios from "axios";

const ai = axios.create({
  baseURL: process.env.AI_SERVICE_URL,
  timeout: 5000,
  headers: { Authorization: `Bearer ${process.env.AI_SERVICE_KEY}` },
});

// SYNC — gate a chat message before delivering it
export async function moderateText({ text, context, userId }) {
  const { data } = await ai.post(
    "/v1/moderate/text",
    { text, context },
    { headers: { "x-user-id": userId } }
  );
  return data; // { action, toxicity, distress, ... }
}

// ASYNC — enqueue a SOAP note; result arrives at the callback route
export async function requestSoapNote({ transcript, sessionId, patientId }) {
  const { data } = await ai.post("/v1/consultations/soap-note", {
    transcript, session_id: sessionId, patient_id: patientId,
    idempotency_key: `soap:${sessionId}`,
  });
  return data; // { task_id, status: "queued" }
}
```

### b) The callback route (verify secret → persist to Mongo)

```js
// src/routes/internal/aiCallback.routes.js
export default async function (fastify) {
  fastify.post("/internal/ai-callback", async (request, reply) => {
    if (request.headers["x-callback-secret"] !== process.env.FASTIFY_CALLBACK_SECRET) {
      return reply.code(401).send({ error: "bad callback secret" });
    }
    const { task_id, task_type, status, result, error, entity } = request.body;

    // Idempotency: ignore if we've already processed this task_id.
    // Persist to MongoDB based on task_type + entity (your models, your call).
    if (status === "succeeded") {
      switch (task_type) {
        case "soap_note":  await saveSoapNote(entity.id, result); break;
        case "transcribe": await saveTranscript(entity.id, result); break;
        case "misinfo_check": await flagForReview(entity.id, result); break;
        // ...
      }
    } else {
      await recordAiFailure(task_id, error);
    }
    return reply.code(200).send({ ok: true });
  });
}
```

### Wiring it into existing flows (examples)

- **Chat** (`msgAndComs/comm.js` `sendMessage`): `await moderateText(...)` before
  broadcasting; if `block`, drop + notify; if `distress.escalate_to_human`, route
  to crisis workflow.
- **Appointments** (`appointments.controller` create): `await triage(...)`; store
  `urgency_level` + `recommended_specialty`; feed "intelligent provider matching".
- **Posts/comments** (`socialMedia/*`): `moderateText` (sync gate) + enqueue
  `misinfo/check` (async advisory flag).
- **Consultation end** (voice/video `call_ended`): once audio is available,
  enqueue `transcribe` → on its callback, enqueue `soap-note` + `summary`.

---

## 8. Audio path (transcription → SOAP/summary)

This is the **one external blocker** for #7–#9. The SOAP/summary generation is
built and tested; it just needs a transcript, and there's no transcript until
consultation audio is captured. Everything in Tier 1 (triage, moderation,
distress, misinfo, sentiment, support) and Tier 3 (search, recommend) is
**unblocked and text-only** — start integration there while this is decided.

### Recommendation: client-side recording + post-call upload (Phase 1)

SOAP and summary are **post-consultation, async** tasks — they don't need a live
stream. So the cheapest, most privacy-safe option fits: the **client records the
call locally and uploads the audio when the call ends.**

**Why this over an SFU/media server:**
- **No new server infrastructure** → no recurring cost.
- **PHI stays in your infrastructure** — audio goes client → your backend → your
  storage → AI service. No third party, no BAA.
- **Matches the use case** — these tasks run after the call; real-time isn't needed.
- **Unblocks work that's already built.**

### Where the media lives

| Layer | Where | Lifespan |
|-------|-------|----------|
| Temporary buffer | The device, in **app-private storage** (browser IndexedDB / mobile app *cache* dir) — **never** the photo gallery or shared files | Seconds–minutes |
| System of record | **Your backend object storage** (self-hosted MinIO/S3) | Per your retention policy |

The phone is a **courier, not a vault.** As soon as the backend confirms receipt,
the **local copy is deleted** — PHI must not linger on the device.

### Recording lifecycle

```
call ends (voice/video `call_ended`)
  → finalize recording (app-private temp file)
  → upload to backend
  → backend confirms 200  →  DELETE local copy
  → on failure (network drop / app closed / device died):
       keep the temp file in an encrypted retry queue,
       retry on reconnect / next app open, then delete once confirmed
```

**Whole-file at call end** is simplest; **chunked upload during the call**
(every ~10–30 s) is recommended for long consultations — a crash then loses only
the last chunk, the upload finishes near-instantly at call end, and transcription
can even start early.

### End-to-end flow

```
client records → uploads → backend stores in object storage
  → POST /api/v1/consultation/transcribe  (audio)
  → transcript returned via callback
  → POST /api/v1/consultation/soap-note  +  /summary
  → results returned via callback → persist to Mongo
  → clinician reviews & signs the SOAP draft (it is a draft, never final)
```

### Capture scope — ONE device records, ONE upload, both voices

**Only one participant records, and only one upload happens** — exactly like a
phone call-recording app: your phone captures *both* voices (yours + the person
you're talking to) from one device; you don't ask the other person to record
their half too.

Why one device is enough: during the call that device has **two** audio sources —
its **microphone** (this person's voice) **and the other person's voice arriving
over the network** (what lets you hear them). The app records **both together** =
the whole conversation. The microphone alone is only half; the incoming audio is
the other half, and the app grabs it too (via the WebRTC remote track + Web Audio).

- **Pick one recorder** — typically the **practitioner's client** (more stable).
  The other party uploads **nothing**. No duplicate submissions, no missing half.
- **Simplest:** record a **single mixed mono track** (local mic + remote mixed into
  one `MediaRecorder`). Enough for Whisper/SOAP. **No speaker labels.**
- **Want auto "Doctor:/Patient:" labels?** Still one device, one upload — record
  **stereo**: practitioner → **left** channel, patient → **right** channel, then
  call `/transcribe` with `diarize_stereo: true` + `channel_roles: ["doctor",
  "patient"]`. **We split the channels and transcribe each server-side** → a
  labeled transcript, no diarization model. (A single *mixed* track can't be
  labeled this way; only stereo carries the per-speaker identity.)
- **Mobile** uses native recording; **web** uses `MediaRecorder` (webm/opus is fine).

### Safety / PHI rules (bake these in)
1. **Consent first** — recording must not start until both parties agree
   (regulated-data + clinical requirement).
2. **App-private + encrypted at rest** while queued on the device.
3. **Delete locally** once the backend confirms receipt.
4. Define a **retention policy** for the stored audio.

### Decisions the backend/product team owns
Consent UX · object storage choice · retention period · chunked-vs-whole-file ·
mobile native recording.

### Audio sourcing (implemented ✅)
`/transcribe` accepts **exactly one** of:
- `audio_base64` — fine for short clips.
- `audio_url` — a **fetchable URL** (e.g. a presigned object-storage link). The AI
  service downloads it, so long consultations aren't base64-inflated through the
  request body. **Preferred for real recordings.**

Guards: the URL must be `http`/`https`; downloads are streamed with a size cap
(`AUDIO_MAX_MB`, default 100); and an optional host allowlist
(`AUDIO_FETCH_ALLOWED_HOSTS`, e.g. `["storage.marcusina.dev"]`) prevents SSRF — set
it to your storage host in production. Use **presigned/expiring URLs** so the
service can fetch without credentials.

### Phase 2 (only if needed later)
If you later want **live real-time transcription** or **recordings as a product
feature**, that's when an SFU (LiveKit / mediasoup / Janus) becomes worth the
infrastructure. Don't pay that cost now — Phase 1 delivers SOAP/summary without it.
```
