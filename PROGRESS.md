# Car Classification Service Continuous Improvement Progress

This file tracks current status, prioritized opportunities, verification, and
completed autonomous improvement cycles.

## Current state

- FastAPI inference service for a 196-class TensorFlow/Keras model.
- Model and dataset artifacts are intentionally not tracked in Git.
- Baseline after Cycle 6: 8 model-free API tests cover readiness, upload
  validation, safe errors, and successful ranking.

## Opportunity backlog

| Priority | Opportunity | Category | Impact | Effort / risk | Evidence / dependencies | Status |
|---|---|---|---|---|---|---|
| 1 | Unify model artifact discovery in `run.py`, Docker, and the loader | Bug / deploy reliability | Critical: documented `.keras` models are accepted by the loader but rejected or omitted by launch/build paths | Medium / medium | Docker currently requires one absent `.h5` filename | Next |
| 2 | Replace deprecated FastAPI startup events with lifespan | Reliability / maintainability | Medium: current tests emit framework deprecation warnings | Small / low | FastAPI 0.116 lifespan API | Backlog |
| 3 | Add lightweight CI for API contract tests | Test / process | High compounding value: new tests are local-only | Medium / low | TensorFlow import makes a minimal CI environment non-trivial | Backlog |
| 4 | Validate model output width against class mapping at startup | Correctness | High: mismatched artifacts otherwise fail only during a request | Small / low | Requires model output-shape contract | Backlog |
| — | Make API readiness and prediction failures honest and bounded | Correctness / test / security | High: false health, unbounded reads, and exception leakage | Small / low | Reproduced without a model artifact | Completed in Cycle 6 |

## Cycle log

### Cycle 6 — Harden and test the API boundary (2026-08-09)

**Why this won:** This was the only executable service in the workspace without
an automated test baseline. Its readiness endpoint returned HTTP 200 and
`"healthy"` with neither model nor mapping loaded, and prediction responses
included raw internal exception text. Both failures were reproduced before the
change.

**Plan and success criteria**

1. Return 503 until inference dependencies are ready and reject prediction
   attempts during that state.
2. Accept only documented JPEG/PNG types, bound reads to 10 MB, and distinguish
   invalid client images from internal inference failures.
3. Never expose decoder/model exception text to clients.
4. Add model-free endpoint tests for unavailable, invalid, successful, and
   internal-error paths.

**Changes**

- Made `/health` an HTTP readiness check with honest 200/503 semantics.
- Added an early `/predict` readiness guard, strict MIME allowlist, empty-file
  handling, and a bounded 10 MB read.
- Mapped decoder failures to safe 400 responses and internal prediction errors
  to a generic 500 response while retaining server-side trace logging.
- Added `tests/test_api.py` with a fake predictor and documented the portable
  `python3 -m pytest -q` command.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 8 passed.
- `.venv/bin/python -m compileall -q api tests`: passed.
- `python3 -m compileall -q run.py`: passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed (the two existing
  edited files use repository-native CRLF endings).
- Before: unloaded `/health` returned 200/`healthy`; after: 503/`unavailable`.
- Before: a model exception exposed `/private/model/path`; after: the client
  receives only `{"detail": "Prediction failed"}`.
- Pytest reports two existing FastAPI `on_event` deprecation warnings, recorded
  as the lifespan backlog item rather than hidden.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 4/10 | 8/10 | Readiness and client/server failure classes now have explicit contracts |
| Test coverage / verifiability | 1/10 | 8/10 | Eight endpoint tests require no trained model artifact |
| Maintainability | 5/10 | 7/10 | Boundary constants and behavior are named and documented |
| Performance / resources | 4/10 | 8/10 | Request reads stop after the configured maximum plus one byte |
| Security / safety | 3/10 | 8/10 | Internal paths/errors are no longer returned to clients |

**Lesson / process improvement:** Heavy ML services can still have fast API
tests: isolate the inference boundary behind a tiny fake model and test the HTTP
contract without loading weights. Invoke pytest through the active Python
interpreter because copied/moved virtualenv launcher shebangs can be stale.

**Next opportunity:** Repair the model-artifact mismatch across `run.py`, the
Dockerfile, and `load_model` so every documented supported model format can
actually launch and package consistently.
