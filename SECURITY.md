# Dependency security

## Scanning

```bash
pip install pip-audit
.venv/Scripts/python -m pip_audit          # audits the installed environment (incl. transitive)
```

CI/Dependabot watch `requirements.txt`. Re-run after any dependency change.

## Audit — 2026-06-19

Starting point: 49 advisories across 7 packages. Fixed the web-facing /
request-handling surface and the build tool; the rest are tracked below.

### Fixed (pinned in requirements.txt)

| Package | Was | Now | Why |
|---|---|---|---|
| `fastapi` | 0.111.0 | 0.115.12 | pulls `starlette>=0.40` — clears the 0.37.x multipart DoS (CVE-2024-47874) and others |
| `python-multipart` | 0.0.9 | 0.0.32 | CVE-2024-53981 + later multipart DoS fixes (direct request-parsing path) |
| `sentencepiece` | 0.2.0 | 0.2.1 | upstream security fix |
| `setuptools` | 65.5.0 | `>=78.1.1,<82` | CVE-2024-6345 (RCE) + CVE-2025-47273 (path traversal); `<82` satisfies torch |

All **157 tests pass** on these versions.

### Tracked residuals (deliberately not bumped here)

| Package | Vulns | Why deferred | Plan |
|---|---|---|---|
| `transformers` 4.40.1 | ~26 | A bump to 4.48+ cascades into `tokenizers` (0.19→0.21), `optimum`, and `sentence-transformers`, and touches the inference path. We only load **pinned, trusted** models (no untrusted deserialization at runtime), so live risk is low. | Dedicated upgrade PR with the eval harness re-run to confirm no model regression. |
| `starlette` 0.46.2 | residual | Capped by `fastapi==0.115.12`. The latest fastapi (0.137) **breaks our routing** and forces pydantic 2.13. | Framework migration handled separately (fastapi + pydantic + route/test review). |
| `torch` 2.12.0 | 1 | No patched version published upstream. Transitive (sentence-transformers/transformers). | Monitor; upgrade when a fix ships. |
| `pytest` 8.1.1 | 1 | **Dev/test only** — not shipped in the deployed image. The fix (9.0.3) is a major bump that breaks `pytest-asyncio==0.23.6`. | Bump `pytest` + `pytest-asyncio` together in a tooling PR. |

> Principle: we patch the production request-handling and build surfaces promptly,
> and gate ML-stack upgrades behind an eval re-run so a security bump can't silently
> regress clinical/safety behavior.
