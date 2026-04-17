/**
 * Fastify ↔ Health AI Service Integration (v2)
 * =============================================
 * Async task pattern:
 *   1. POST to AI endpoint → get { task_id } back in <5ms
 *   2. Either:
 *      a) Webhook: AI service POSTs result to FASTIFY_CALLBACK_URL when done
 *      b) Polling: GET /api/v1/task/{task_id} until status === "complete"
 *
 * Install: npm install undici
 */

import { fetch } from "undici";

const AI_URL    = process.env.AI_SERVICE_URL    || "http://localhost:8001/api/v1";
const AI_SECRET = process.env.AI_SECRET_KEY     || "change-me";
const CB_SECRET = process.env.AI_CALLBACK_SECRET || "change-me-callback";

const AI_HEADERS = {
  "Content-Type": "application/json",
  "X-AI-Secret": AI_SECRET,
};

// ── Generic helpers ───────────────────────────────────────────────────────────

async function enqueue(endpoint, body) {
  const res = await fetch(`${AI_URL}${endpoint}`, {
    method: "POST",
    headers: AI_HEADERS,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "AI error" }));
    throw new Error(`[AI ${res.status}] ${err.detail}`);
  }
  return res.json();
  // Returns: { task_id, status: "queued" | "complete", result? }
}

/**
 * Poll until task is complete or timeout.
 * Use this for sync-style flows (e.g. triage needs an answer before continuing).
 */
async function pollUntilComplete(taskId, { timeoutMs = 30000, intervalMs = 500 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await fetch(`${AI_URL}/task/${taskId}`, { headers: AI_HEADERS });
    const data = await res.json();
    if (data.status === "complete") return data.result;
    if (data.status === "failed")   throw new Error(`AI task failed: ${data.error}`);
    await new Promise(r => setTimeout(r, intervalMs));
  }
  throw new Error(`AI task ${taskId} timed out after ${timeoutMs}ms`);
}

// ── If result is cached, it comes back immediately in the enqueue response ───
async function enqueueOrGetCached(endpoint, body) {
  const enqueued = await enqueue(endpoint, body);
  if (enqueued.status === "complete" && enqueued.result) {
    return { taskId: enqueued.task_id, result: enqueued.result, cached: true };
  }
  return { taskId: enqueued.task_id, result: null, cached: false };
}

// ════════════════════════════════════════════════════════════════════════════ //
// E-CONSULTATION                                                               //
// ════════════════════════════════════════════════════════════════════════════ //

/**
 * Transcribe audio — fire and forget, result via webhook.
 * @param {Buffer} audioBuffer
 * @param {string} sessionId
 * @param {string} format  "webm" | "mp3" | "wav"
 * @returns {string} task_id
 */
export async function transcribeAudio(audioBuffer, sessionId, format = "webm") {
  const { taskId } = await enqueueOrGetCached("/consultation/transcribe", {
    audio_base64: audioBuffer.toString("base64"),
    audio_format: format,
    session_id: sessionId,
    speaker: "patient",
  });
  return taskId; // Webhook delivers result when ready
}

/**
 * Triage patient — poll for result (doctor/flow needs the urgency level now).
 * Emergency cases are automatically routed to the highest-priority Celery queue.
 */
export async function triagePatient(patientId, symptoms, { age, medicalHistory } = {}) {
  const { taskId, result, cached } = await enqueueOrGetCached("/consultation/triage", {
    patient_id: patientId,
    symptoms,
    age: age || null,
    medical_history: medicalHistory || [],
  });

  if (cached) return result;
  // Poll — triage should resolve in ~500ms
  const triageResult = await pollUntilComplete(taskId, { timeoutMs: 10000, intervalMs: 300 });

  if (triageResult.urgency_level === "emergency") {
    console.error(`[EMERGENCY] Patient ${patientId}: ${triageResult.red_flag_symptoms}`);
    // TODO: trigger your on-call doctor notification
  }
  return triageResult;
}

/**
 * Generate SOAP note — fire and forget, result via webhook.
 */
export async function generateSOAPNote(sessionId, transcript, patientId, doctorId) {
  const { taskId } = await enqueueOrGetCached("/consultation/soap-note", {
    session_id: sessionId,
    transcript,
    patient_id: patientId,
    doctor_id: doctorId,
  });
  return taskId;
}

// ════════════════════════════════════════════════════════════════════════════ //
// HEALTH SOCIAL MEDIA                                                          //
// ════════════════════════════════════════════════════════════════════════════ //

/**
 * Moderate content — poll because we need the verdict before storing the post.
 * Cache hit = instant return (same content hash seen before).
 */
export async function moderateContent(contentId, contentType, text, authorId) {
  const { taskId, result, cached } = await enqueueOrGetCached("/social/moderate", {
    content_id: contentId,
    content_type: contentType,
    text,
    author_id: authorId,
  });
  if (cached) return result;
  return pollUntilComplete(taskId, { timeoutMs: 8000, intervalMs: 200 });
}

/**
 * Get recommendations — fire and forget OR poll based on context.
 */
export async function getRecommendations(userId, context, { interests = [], conditions = [], limit = 10 } = {}) {
  const { taskId, result, cached } = await enqueueOrGetCached("/social/recommend", {
    user_id: userId,
    context,
    user_interests: interests,
    user_conditions: conditions,
    limit,
  });
  if (cached) return result;
  return pollUntilComplete(taskId, { timeoutMs: 5000, intervalMs: 250 });
}

/**
 * Sentiment analysis — always fire and forget.
 * Mental health alerts are handled by the webhook callback.
 */
export async function analyzeSentiment(contentId, text) {
  const { taskId } = await enqueue("/social/sentiment", { content_id: contentId, text });
  return taskId;
}

// ════════════════════════════════════════════════════════════════════════════ //
// WEBHOOK RECEIVER (register this in your Fastify server)                     //
// ════════════════════════════════════════════════════════════════════════════ //

/**
 * Register the AI callback route in your Fastify instance.
 * The AI service POSTs here when an async task completes.
 *
 * In your Fastify bootstrap:
 *   registerAICallbackRoute(fastify);
 */
export function registerAICallbackRoute(fastify) {
  fastify.post("/internal/ai-callback", {
    config: { rawBody: true },
  }, async (request, reply) => {
    // Verify the callback is from our AI service
    const secret = request.headers["x-callback-secret"];
    if (secret !== CB_SECRET) {
      return reply.status(403).send({ error: "Forbidden" });
    }

    const { task_id, result } = request.body;

    // ── Route result to the right handler ──────────────────────────────
    if (result.task_id?.startsWith("transcr")) {
      // Store transcript, notify doctor's session via WebSocket
      // await storeTranscript(result);
      // fastify.io.to(result.session_id).emit("transcript_ready", result);
    }

    if (result.mental_health_concern === true) {
      console.warn(`[WELLNESS ALERT] content_id=${result.content_id}`);
      // TODO: notify wellness support team
    }

    if (result.urgency_level === "emergency") {
      console.error(`[EMERGENCY] patient=${result.patient_id}`);
      // TODO: alert on-call doctor
    }

    // You can also store result in Redis and let the frontend poll:
    // await redis.setex(`task:${task_id}`, 3600, JSON.stringify(result));

    return reply.send({ received: true });
  });
}

// ════════════════════════════════════════════════════════════════════════════ //
// EXAMPLE ROUTE WIRING                                                         //
// ════════════════════════════════════════════════════════════════════════════ //

export function registerRoutes(fastify) {
  // POST /posts — moderate before storing
  fastify.post("/posts", async (req, reply) => {
    const { text, contentType = "post", userId } = req.body;
    const contentId = `post_${Date.now()}_${userId}`;

    const moderation = await moderateContent(contentId, contentType, text, userId);

    if (moderation.verdict === "rejected") {
      return reply.status(422).send({ error: "Content rejected", reasons: moderation.flagged_reasons });
    }

    const safeText = moderation.pii_detected ? moderation.safe_text : text;
    // await db.posts.create({ id: contentId, text: safeText, userId, status: moderation.verdict });

    // Async sentiment — don't await
    analyzeSentiment(contentId, safeText).catch(console.error);

    return reply.send({ success: true, content_id: contentId, status: moderation.verdict });
  });

  // GET /feed/:userId
  fastify.get("/feed/:userId", async (req, reply) => {
    const { userId } = req.params;
    // const profile = await db.users.getProfile(userId);
    const recs = await getRecommendations(userId, "feed", {
      interests: [],   // profile.interests
      conditions: [],  // profile.conditions
    });
    return reply.send(recs);
  });

  // POST /triage
  fastify.post("/triage", async (req, reply) => {
    const { patientId, symptoms, age, medicalHistory } = req.body;
    const result = await triagePatient(patientId, symptoms, { age, medicalHistory });
    return reply.send(result);
  });

  // Register webhook receiver
  registerAICallbackRoute(fastify);
}
