"""
scripts/test_endpoints.py
=========================
Live endpoint tests against the running server.
Run AFTER: uvicorn app.main:app --reload --port 8001

Usage: python scripts/test_endpoints.py
Exit code: 0 = all passed, 1 = one or more failed
"""

import httpx
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings

settings = get_settings()

BASE_URL = "http://127.0.0.1:8001/api/v1"
HEADERS = {
    "Content-Type": "application/json",
    "X-AI-Secret": settings.API_SECRET_KEY,
}

PASS = "✅ PASS"
FAIL = "❌ FAIL"

_results: list[bool] = []


def print_result(test_name: str, passed: bool, detail: str = "") -> None:
    _results.append(passed)
    status = PASS if passed else FAIL
    print(f"  {status}  {test_name}")
    if detail:
        print(f"         {detail}")


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return f"Connection refused — is the server running on port 8001? ({exc})"
    if isinstance(exc, httpx.TimeoutException):
        return f"Request timed out — server may be overloaded ({exc})"
    return str(exc)


def poll_task(task_id: str, *, max_wait: float = 10.0, interval: float = 0.5) -> dict:
    """Poll GET /task/{task_id} until status != 'pending', or until max_wait seconds."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        r = httpx.get(f"{BASE_URL}/task/{task_id}", headers=HEADERS, timeout=10)
        data = r.json()
        if data.get("status") not in ("pending", "queued"):
            return data
        time.sleep(interval)
    return data  # return whatever we got after timeout


def main() -> int:
    # ── 1. Health check ──────────────────────────────────────────

    section("1. Health Check")
    try:
        r = httpx.get("http://127.0.0.1:8001/health", timeout=10)
        passed = r.status_code == 200 and r.json().get("status") == "ok"
        print_result("GET /health", passed, str(r.json()))
    except Exception as e:
        print_result("GET /health", False, _error_detail(e))


    # ── 2. Auth checks ───────────────────────────────────────────

    section("2. Authentication")

    try:
        r = httpx.post(f"{BASE_URL}/social/sentiment",
                       json={"content_id": "x", "text": "hello"}, timeout=10)
        print_result("No secret → 401", r.status_code == 401, f"Got {r.status_code}")
    except Exception as e:
        print_result("No secret → 401", False, _error_detail(e))

    try:
        r = httpx.post(f"{BASE_URL}/social/sentiment",
                       headers={"X-AI-Secret": "wrong-key"},
                       json={"content_id": "x", "text": "hello"}, timeout=10)
        print_result("Wrong secret → 403", r.status_code == 403, f"Got {r.status_code}")
    except Exception as e:
        print_result("Wrong secret → 403", False, _error_detail(e))


    # ── 3. Triage ────────────────────────────────────────────────

    section("3. Triage")

    try:
        r = httpx.post(f"{BASE_URL}/consultation/triage", headers=HEADERS, json={
            "patient_id": "p001",
            "symptoms": "severe chest pain and difficulty breathing, cannot breathe",
            "age": 55,
            "medical_history": ["hypertension"],
        }, timeout=30)
        data = r.json()
        task_id = data.get("task_id")
        print_result(
            "Emergency triage → routed to emergency queue",
            r.status_code == 200 and data.get("priority") == "emergency",
            f"priority={data.get('priority')} task_id={task_id}"
        )
    except Exception as e:
        print_result("Emergency triage", False, _error_detail(e))

    try:
        r = httpx.post(f"{BASE_URL}/consultation/triage", headers=HEADERS, json={
            "patient_id": "p002",
            "symptoms": "mild headache for two days",
            "age": 28,
        }, timeout=30)
        data = r.json()
        print_result(
            "Non-urgent triage → normal queue",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} priority={data.get('priority')}"
        )
    except Exception as e:
        print_result("Non-urgent triage", False, _error_detail(e))


    # ── 4. Content Moderation ────────────────────────────────────

    section("4. Content Moderation")

    try:
        r = httpx.post(f"{BASE_URL}/social/moderate", headers=HEADERS, json={
            "content_id": "post_clean_001",
            "content_type": "post",
            "text": "Drinking enough water daily helps keep your kidneys healthy and prevents dehydration.",
            "author_id": "user_123",
        }, timeout=30)
        data = r.json()
        print_result(
            "Clean health content → queued/complete",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} task_id={data.get('task_id')}"
        )
    except Exception as e:
        print_result("Clean content moderation", False, _error_detail(e))

    try:
        r = httpx.post(f"{BASE_URL}/social/moderate", headers=HEADERS, json={
            "content_id": "post_misinfo_001",
            "content_type": "post",
            "text": "This miracle cure 100% cures cancer and diabetes. Doctors don't want you to know!",
            "author_id": "user_456",
        }, timeout=30)
        data = r.json()
        print_result(
            "Misinformation content → queued/complete",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} task_id={data.get('task_id')}"
        )
    except Exception as e:
        print_result("Misinformation moderation", False, _error_detail(e))


    # ── 5. Recommendations ───────────────────────────────────────

    section("5. Recommendations")

    try:
        r = httpx.post(f"{BASE_URL}/social/recommend", headers=HEADERS, json={
            "user_id": "user_789",
            "context": "feed",
            "user_interests": ["diabetes", "nutrition"],
            "user_conditions": ["hypertension"],
            "limit": 5,
        }, timeout=30)
        data = r.json()
        print_result(
            "Recommendations → queued/complete",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} task_id={data.get('task_id')}"
        )
    except Exception as e:
        print_result("Recommendations", False, _error_detail(e))


    # ── 6. Sentiment ─────────────────────────────────────────────

    section("6. Sentiment Analysis")

    try:
        r = httpx.post(f"{BASE_URL}/social/sentiment", headers=HEADERS, json={
            "content_id": "post_sent_001",
            "text": "Feeling much better after following my doctor's advice on diet and exercise!",
        }, timeout=30)
        data = r.json()
        print_result(
            "Positive sentiment → queued/complete",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} task_id={data.get('task_id')}"
        )
    except Exception as e:
        print_result("Sentiment analysis", False, _error_detail(e))

    try:
        r = httpx.post(f"{BASE_URL}/social/sentiment", headers=HEADERS, json={
            "content_id": "post_distress_001",
            "text": "I have been struggling so much lately, I don't see any reason to continue.",
        }, timeout=30)
        data = r.json()
        print_result(
            "Distress content → queued/complete",
            r.status_code == 200 and data.get("status") in ("queued", "complete"),
            f"status={data.get('status')} task_id={data.get('task_id')}"
        )
    except Exception as e:
        print_result("Distress sentiment", False, _error_detail(e))


    # ── 7. Task polling ──────────────────────────────────────────

    section("7. Task Result Polling")

    try:
        r = httpx.post(f"{BASE_URL}/social/sentiment", headers=HEADERS, json={
            "content_id": "poll_test_001",
            "text": "Exercise is great for managing blood pressure.",
        }, timeout=15)
        task_id = r.json().get("task_id")

        if task_id and task_id != "cached":
            poll_data = poll_task(task_id, max_wait=10.0)
            final_status = poll_data.get("status")
            print_result(
                f"Poll task/{task_id[:8]}... → completes within 10s",
                final_status in ("complete", "success", "failed"),
                f"final_status={final_status}"
            )
        else:
            print_result("Task polling", True, "Result was cached — polling not needed")
    except Exception as e:
        print_result("Task polling", False, _error_detail(e))


    # ── 8. Validation errors ─────────────────────────────────────

    section("8. Validation (bad requests)")

    try:
        r = httpx.post(f"{BASE_URL}/consultation/triage", headers=HEADERS, json={
            "patient_id": "p999",
            "symptoms": "ok",
        }, timeout=10)
        print_result(
            "Too-short symptoms → 422",
            r.status_code == 422,
            f"Got {r.status_code}"
        )
    except Exception as e:
        print_result("Validation error handling", False, _error_detail(e))


    # ── Summary ──────────────────────────────────────────────────

    total = len(_results)
    passed = sum(_results)
    failed = total - passed

    print(f"\n{'='*55}")
    print(f"  Test run complete.  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print()
        print("  Server is ready for Fastify integration.")
    print(f"{'='*55}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
