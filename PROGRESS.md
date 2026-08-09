# Car Classification Service Continuous Improvement Progress

This file tracks current status, prioritized opportunities, verification, and
completed autonomous improvement cycles.

## Current state

- FastAPI inference service for a 196-class TensorFlow/Keras model.
- Model and dataset artifacts are intentionally not tracked in Git.
- Baseline after Cycle 9: 17 model-free tests cover the API boundary, lifecycle,
  and model artifact discovery/build command contract with no warnings.

## Opportunity backlog

| Priority | Opportunity | Category | Impact | Effort / risk | Evidence / dependencies | Status |
|---|---|---|---|---|---|---|
| 1 | Add lightweight CI for API contract tests | Test / process | High compounding value: new tests are local-only | Medium / low | TensorFlow import makes a minimal CI environment non-trivial | Next |
| — | Validate model output width against class mapping at startup | Correctness | High: mismatched artifacts otherwise failed only during a request | Small / low | Atomic lifespan validation | Completed in Cycle 9 |
| — | Replace deprecated FastAPI startup events with lifespan | Reliability / maintainability | Medium: framework lifecycle compatibility | Small / low | Warning-free lifecycle test | Completed in Cycle 8 |
| — | Unify model artifact discovery in `run.py`, Docker, and the loader | Bug / deploy reliability | Critical: supported `.keras` models were rejected or omitted by launch/build paths | Medium / medium | Shared candidate order plus Docker build argument | Completed in Cycle 7 |
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

### Cycle 7 — Unify model artifact discovery (2026-08-09)

**Why this won:** `api.utils.load_model` preferred `best_car_model.keras` and
also supported legacy H5/SavedModel paths, but `run.py` rejected everything
except `car_classification_model.h5` and the Dockerfile always copied that exact
file. The documented preferred artifact therefore could not launch through the
standard deployment path.

**Plan and success criteria**

1. Use the loader's artifact preference in local, automatic, and Docker launch
   checks.
2. Pass the selected relative artifact path into Docker rather than hard-code a
   filename.
3. Test preference, all supported shapes, the generated build command, and
   fail-fast behavior without invoking Docker.

**Changes**

- Added shared model candidate constants and `find_model_artifact` to `run.py`.
- Updated local launch, automatic selection, and Docker build checks to use the
  discovered model plus `class_mapping.json`.
- Added Docker `MODEL_PATH` build argument support.
- Added five runner tests and documented manual artifact selection.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 13 passed (up from 8).
- `.venv/bin/python -m compileall -q api tests run.py`: passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.
- Tests verify `.keras` preference, H5 fallback, SavedModel directory support,
  Docker build-argument propagation, and no Docker call when weights are absent.
- A real Docker image build was not possible because trained weights are
  intentionally absent from this checkout; this is an external artifact
  boundary, not a passing claim.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 3/10 | 9/10 | All loader-supported artifact paths reach standard launch checks |
| Test coverage / verifiability | 4/10 | 8/10 | Five model-free deployment-path tests cover selection and command creation |
| Maintainability | 4/10 | 8/10 | One ordered candidate list replaces three filename checks |
| Developer experience | 3/10 | 8/10 | Preferred `.keras` output now works with documented commands |
| Security / safety | 8/10 | 8/10 | Fixed candidate paths are passed; no new untrusted shell input |

**Lesson / process improvement:** Artifact discovery belongs in one function
shared by every launcher. Test command construction with a captured runner so
deployment logic stays verifiable when large external artifacts are absent.

**Next opportunity:** Migrate startup loading to FastAPI lifespan, eliminate the
two deprecation warnings, and preserve readiness behavior with lifecycle tests.

### Cycle 8 — Adopt FastAPI lifespan startup (2026-08-09)

**Why this won:** Every test emitted two deprecation warnings for
`app.on_event("startup")`. The framework-supported lifespan API removes known
upgrade risk, and a lifecycle-aware test can verify that readiness changes only
after both dependencies load.

**Plan and success criteria**

1. Move model/mapping loading into an async lifespan context.
2. Preserve fail-fast startup and existing readiness semantics.
3. Add a `TestClient` lifecycle test and reach a warning-free suite.

**Changes**

- Replaced the deprecated startup decorator with an `asynccontextmanager`
  lifespan passed to `FastAPI`.
- Added a lifecycle test that injects fake dependencies, enters application
  lifespan, and observes HTTP 200 readiness.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 14 passed, zero warnings (previously 13
  passed with 2 deprecation warnings).
- `.venv/bin/python -m compileall -q api tests run.py`: passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 7/10 | 9/10 | Actual lifecycle now proves dependencies load before readiness |
| Test coverage / verifiability | 7/10 | 9/10 | Startup behavior is tested through ASGI lifespan |
| Maintainability | 5/10 | 9/10 | Uses the supported FastAPI lifecycle API with no warnings |
| Performance | 9/10 | 9/10 | Same one-time startup loading behavior |
| Developer experience | 7/10 | 9/10 | Test output is warning-free and actionable |

**Lesson / process improvement:** Treat deprecation warnings as backlog evidence,
then eliminate them with a behavior-level lifecycle test rather than merely
filtering warning output.

**Next opportunity:** Validate the loaded model's output width against the class
mapping during lifespan so incompatible artifacts fail before serving traffic.

### Cycle 9 — Validate model/mapping compatibility (2026-08-09)

**Why this won:** Startup previously treated any independently loadable model
and mapping as ready. A mismatched class count or broken reverse mapping would
surface only after accepting a prediction request, producing a generic 500.

**Plan and success criteria**

1. Require a single known positive model output width.
2. Require `index_to_class` to exactly cover contiguous output indices and
   `class_to_index` to be its inverse.
3. Publish model/mapping globals atomically only after validation and prove a
   failed lifespan never becomes ready.

**Changes**

- Added `validate_runtime_artifacts` at the inference startup boundary.
- Changed lifespan to load into local variables and assign globals only after
  compatibility succeeds.
- Added width-mismatch, reverse-mapping, and atomic-failure tests; enhanced the
  fake model with an output shape.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 17 passed, zero warnings (up from 14).
- `.venv/bin/python -m compileall -q api tests run.py`: passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.
- A three-output fake model paired with five labels now aborts lifespan, and
  both readiness globals remain `None`.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 4/10 | 9/10 | Incompatible artifacts fail before serving traffic |
| Test coverage / verifiability | 6/10 | 9/10 | Shape, inverse mapping, and atomic publication have direct tests |
| Maintainability | 6/10 | 8/10 | One startup validator owns the artifact compatibility contract |
| Performance | 9/10 | 9/10 | Validation is linear over 196 labels and runs once |
| Security / safety | 8/10 | 8/10 | No external surface change; failure remains server-side |

**Lesson / process improvement:** Related runtime resources should be loaded
into locals, cross-validated, and published atomically. Readiness must reflect a
validated pair, not merely two non-null objects.

**Next opportunity:** Add lightweight GitHub Actions coverage without requiring
TensorFlow/model downloads by separating API contract imports from heavy model
loading dependencies.
