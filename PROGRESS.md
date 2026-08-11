# Car Classification Service Continuous Improvement Progress

This file tracks current status, prioritized opportunities, verification, and
completed autonomous improvement cycles.

## Current state

- FastAPI inference service for a 196-class TensorFlow/Keras model.
- Model and dataset artifacts are intentionally not tracked in Git.
- Baseline after Cycle 28: 95 model-free tests cover the API boundary,
  lifecycle, model input/output compatibility, prediction decoding,
  exact class-mapping metadata, decoded-image policy, lightweight import, and
  model artifact discovery/build command. Request-level concurrency coverage
  proves slow synchronous inference stays off the event loop and shared model
  access is serialized; 19 CI policy assertions keep the complete lightweight
  gate on supported action runtimes without TensorFlow or trained weights.
- Dependency audit: the lightweight test graph has zero known vulnerabilities;
  production resolution has 17 Keras-only findings constrained by real model
  compatibility; the full training workspace now has the same Keras-only set.

## Opportunity backlog

| Priority | Opportunity | Category | Impact | Effort / risk | Evidence / dependencies | Status |
|---|---|---|---|---|---|---|
| 1 | Bound inference queue wait with explicit overload behavior | Reliability / resources | Medium-high: model calls are serialized, but callers can wait indefinitely behind a stalled prediction | Medium / medium | Requires a documented timeout/status policy that does not abandon a running TensorFlow thread | Backlog |
| 2 | Re-export models for a current Keras release | Security / reliability | High: Keras 3.10 retains 17 advisory records, but every tested fixed release breaks real artifact loading | High / high | Requires trusted migration/re-export plus prediction-equivalence evidence | Backlog |
| — | Bound synchronous inference concurrency | Performance / reliability | Medium-high: CPU/GPU prediction ran directly inside an async route with no admission control | Medium / medium | Synchronized request test proves responsive health and one active model call | Completed in Cycle 28 |
| — | Require an exact class-mapping bijection | Correctness / observability | Low-medium: startup checks required pairs but permitted extra reverse entries and weak label/index types | Small / low | Thirteen pure and runtime fixtures cover the exact typed inverse contract | Completed in Cycle 27 |
| — | Clear non-Keras training-workspace audit findings | Security / maintenance | Medium-high: requests, notebook, JupyterLab, and python-dotenv retained 19 advisory records | Medium / medium | Fresh full install, tool imports, entry points, kernel execution, and audit isolate Keras as the only remaining exposure | Completed in Cycle 26 |
| — | Audit and refresh deploy/test dependency pins | Security / maintenance | High: test and production resolution initially exposed 43 and 59 advisory records | Medium / medium | Fresh Python 3.12 tests, audits, and real model load separated safe upgrades from breaking Keras releases | Completed in Cycle 25 |
| — | Modernize and policy-test GitHub Actions | Process / observability | Medium: hosted CI passed with Node 20 deprecation annotations and lacked timeout, permissions, or policy contracts | Small-medium / low | Nineteen policy assertions enforce the bounded v7 workflow | Completed in Cycle 24 |
| — | Validate model output rank and batch metadata at startup | Correctness / reliability | High: rank-three or fixed multi-row outputs could report ready then fail every request | Small / low | Six rejection/acceptance contracts align startup with the runtime decoder | Completed in Cycle 23 |
| — | Verify decoded image format instead of trusting MIME alone | Correctness / security | Medium: renamed unsupported formats passed the JPEG/PNG header check | Small / low | Real GIF/WebP fixtures at utility and endpoint boundaries | Completed in Cycle 22 |
| — | Apply JPEG EXIF orientation before resize | Correctness / UX | Medium-high: phone photos were classified sideways despite valid pixels | Small / low | Asymmetric orientation-6 JPEG | Completed in Cycle 21 |
| — | Validate model input shape against preprocessing | Correctness / reliability | High: an incompatible artifact passed readiness then failed every request | Small / low | Shared `(1, 224, 224, 3)` contract | Completed in Cycle 20 |
| — | Validate prediction tensor shape and finite scores | Correctness / robustness | High: malformed model output produced misleading rankings or generic indexing failures | Small / low | Pure decoder plus fake outputs | Completed in Cycle 19 |
| — | Bound decoded image dimensions before resize | Security / resources | High: compressed images bypassed the byte-size limit after decoding | Small / low | 50-megapixel ceiling | Completed in Cycle 18 |
| — | Distinguish corrupt model artifacts from missing artifacts | Correctness / observability | High: loader falsely returned `FileNotFoundError` after deserialization failures | Small / low | Injected loader covers preference and fallback | Completed in Cycle 17 |
| — | Preserve documented class-mapping error types | Correctness / test | Medium: invalid structure was rewrapped as `RuntimeError` | Small / low | Four isolated mapping fixtures | Completed in Cycle 16 |
| — | Split training and API runtime dependencies | Performance / deploy | High: Docker installed notebook, plotting, dataset, and training packages | Medium / medium | Pinned inference manifest | Completed in Cycle 15 |
| — | Validate CLI ports before launching Uvicorn/Docker | Correctness / UX | Medium: invalid ports reached setup/launch | Small / low | Seven boundary cases | Completed in Cycle 14 |
| — | Replace `shell=True` runner commands with argument vectors | Security / portability | High: shell strings and unconditional `sudo` made workflows brittle | Medium / medium | Interactive processes retain inherited terminal | Completed in Cycle 13 |
| — | Run the container as a non-root user | Security | Medium: inference does not need root inside the image | Small / low | Writable home plus read-only app/model access | Completed in Cycle 12 |
| — | Add a focused `.dockerignore` | Performance / security | High: Docker sent virtualenvs, datasets, Git history, and local artifacts as build context | Small / low | Allowlist retains all supported model shapes | Completed in Cycle 11 |
| — | Add lightweight CI for API contract tests | Test / process | High compounding value: tests were local-only | Medium / low | Lazy TensorFlow import plus minimal dependency file | Completed in Cycle 10 |
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
- A full Docker image build was not run because reinstalling the pinned
  TensorFlow stack is expensive; command construction was tested. Cycle 11
  later confirmed ignored local weights are present and added static Dockerfile
  checks for each supported shape.

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

### Cycle 10 — Add lightweight API CI (2026-08-09)

**Why this won:** Eighteen reliable tests provide little compounding value if
they run only in one local virtualenv. Importing `api.main` eagerly imported
TensorFlow, forcing a large unrelated dependency into every contract test and
making fast CI unnecessarily expensive.

**Plan and success criteria**

1. Keep TensorFlow imports on real model-loading/model-inspection paths only.
2. Define a minimal pinned test dependency set and prove it in a fresh venv.
3. Run tests and syntax compilation on pushes and pull requests.

**Changes**

- Made TensorFlow imports lazy in `api/utils.py` while retaining deferred type
  annotations.
- Added `requirements-test.txt` and a subprocess regression proving API import
  does not touch TensorFlow.
- Added `.github/workflows/ci.yml` for Python 3.12 tests and compilation.
- Documented the runtime/test dependency split.

**Verification evidence**

- Existing environment: 18 tests passed in 1.08s, down from about 5.8s before
  lazy import; compile and CRLF-aware diff checks passed.
- Fresh temporary venv containing only `requirements-test.txt`: 18 tests passed
  in 0.95s and compilation passed.
- The subprocess test installs a TensorFlow import blocker before importing the
  API, providing direct evidence that the lightweight boundary is real.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 8/10 | 9/10 | Every push/PR will execute the API and runner contracts |
| Test coverage / verifiability | 3/10 | 9/10 | Clean-environment CI is now defined and reproduced locally |
| Maintainability | 6/10 | 9/10 | Runtime and contract-test dependencies have explicit boundaries |
| Performance / resources | 4/10 | 9/10 | Contract startup fell to ~1s and skips TensorFlow installation |
| Developer experience | 5/10 | 9/10 | Small pinned requirements reproduce CI without model weights |

**Lesson / process improvement:** A heavy dependency should load at the narrow
runtime path that needs it, not at module import. Validate CI dependency files
in a fresh environment before trusting a workflow definition.

**Next opportunity:** Add `.dockerignore` rules that exclude Git metadata,
virtualenvs, datasets, caches, notebooks, and test-only files while retaining
the selected external model artifact.

### Cycle 11 — Bound the Docker build context (2026-08-09)

**Why this won:** The local checkout occupied 5.1 GB, including a 2.6 GB
virtualenv and 1.9 GB dataset. Without `.dockerignore`, every build could send
all of it to the Docker daemon even though the Dockerfile consumes only API
source, requirements, mapping, and one model artifact.

**Plan and success criteria**

1. Default-deny the context and re-include only Docker inputs.
2. Preserve `.keras`, H5, and SavedModel artifact paths supported by Cycle 7.
3. Add a contract test and run Docker's static build check with each locally
   available artifact argument.

**Changes**

- Added an allowlist-style `.dockerignore` excluding Git data, virtualenvs,
  datasets, notebooks, tests, caches, and unrelated outputs.
- Added a regression test requiring the default-deny rule and every necessary
  runtime inclusion.
- Corrected Cycle 7's inference from tracked files: ignored model weights are
  present locally even though they are intentionally absent from Git.

**Verification evidence**

- Pre-change measurement: repository 5.1 GB; `.venv` 2.6 GB; `data` 1.9 GB;
  `models` 192 MB, plus root `.keras` and H5 artifacts. The allowlisted files
  total about 615 MB, an approximately 88% reduction before Docker's own
  transfer/compression behavior.
- `python -m pytest -q`: 19 tests passed.
- `docker build --check` reported no warnings for default `.keras`, H5, and
  SavedModel `MODEL_PATH` arguments. This is a static Dockerfile check, not a
  claim that full COPY and dependency-install stages were built.
- Python compilation and CRLF-aware diff checks passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 7/10 | 9/10 | Every Docker COPY input is explicit and contract-tested |
| Test coverage / verifiability | 6/10 | 8/10 | Ignore policy and three artifact arguments are statically checked |
| Maintainability | 5/10 | 9/10 | Context contents now follow a short allowlist |
| Performance / resources | 2/10 | 8/10 | Multi-gigabyte unrelated directories are excluded |
| Security / safety | 4/10 | 9/10 | Local secrets/environments and Git metadata cannot enter context |

**Lesson / process improvement:** Inspect ignored filesystem state as well as
tracked files when evaluating packaging. For narrow Dockerfiles, a context
allowlist is safer and easier to audit than accumulating exclusions.

**Next opportunity:** Create and switch to a non-root runtime user in the image,
then verify the Dockerfile and health/prediction read paths need no root access.

### Cycle 12 — Drop container root privileges (2026-08-09)

**Why this won:** The inference process only reads application source, mapping,
and model weights, yet the image ran Uvicorn as root. Removing unnecessary
privilege is a small defense-in-depth improvement with no API behavior change.

**Plan and success criteria**

1. Create a dedicated system user with a writable home for library caches.
2. Copy runtime artifacts with that ownership and switch users before health
   checks and the server command.
3. Contract-test the Dockerfile ordering and keep all API checks green.

**Changes**

- Added system group/user `app`, owned runtime copies, and `HOME=/home/app`.
- Added `USER app` before `HEALTHCHECK` and `CMD`.
- Extended Dockerfile tests to enforce the non-root ordering.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 20 passed (up from 19).
- `docker build --check .`: no warnings.
- Python compilation and CRLF-aware diff checks passed.
- A full TensorFlow image build was not required to verify the Dockerfile
  security contract; runtime ownership paths are explicit.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 8/10 | 8/10 | API/runtime contract is unchanged |
| Test coverage / verifiability | 7/10 | 9/10 | User and command ordering are regression-tested |
| Maintainability | 7/10 | 8/10 | Runtime ownership intent is explicit |
| Performance | 9/10 | 9/10 | No material runtime overhead |
| Security / safety | 4/10 | 9/10 | Uvicorn no longer runs with root privileges |

**Lesson / process improvement:** Express container privilege boundaries in the
Dockerfile and test their ordering; comments alone do not prevent a later COPY
or command from moving execution back across the boundary.

**Next opportunity:** Refactor `run.py` away from `shell=True` command strings
and unconditional `sudo`, using argument vectors for safer cross-platform local
and Docker workflows.

### Cycle 13 — Remove shell command execution (2026-08-09)

**Why this won:** `run.py` constructed every setup, Uvicorn, and Docker command
as a shell string, often prefixed with platform-specific `sudo`. Even currently
typed values passed through unnecessary shell interpretation, and Windows/local
Docker workflows inherited Unix assumptions.

**Plan and success criteria**

1. Require argument sequences in the shared runner and never invoke a shell.
2. Convert setup, requirement, Uvicorn, and Docker operations while preserving
   live output and Ctrl+C for long-running processes.
3. Remove unconditional `sudo` and prove spaced arguments stay single values.

**Changes**

- Changed `run_command` to reject strings, render arguments safely for display,
  and call `subprocess.run` with a list and no shell.
- Converted every command site, including cleanup operations and reload flags,
  to explicit argument vectors.
- Replaced the shell-based Docker availability probe with `shutil.which` and
  invoked Docker directly for cross-platform compatibility.
- Added tests for exact Docker arguments, spaced-argument preservation, and
  shell-string rejection.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 22 passed (up from 20).
- `rg 'shell=True|sudo docker' run.py`: no matches; remaining subprocess calls
  all receive constructed lists.
- Python compilation and CRLF-aware diff checks passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 6/10 | 9/10 | Arguments no longer depend on shell parsing or empty flag strings |
| Test coverage / verifiability | 6/10 | 9/10 | Command shape and string rejection have direct tests |
| Maintainability | 5/10 | 9/10 | Commands are structured data rather than interpolated scripts |
| Developer experience | 4/10 | 8/10 | Docker no longer assumes Unix `sudo`; server output stays interactive |
| Security / safety | 3/10 | 9/10 | Shell interpretation is removed from all launch paths |

**Lesson / process improvement:** Treat subprocess commands as argument data by
default. Keep captured helper commands separate from foreground processes, but
use the same no-shell invariant for both.

**Next opportunity:** Add an argparse port validator covering 1–65535 so invalid
ports fail immediately with a clear CLI error before setup or launch work.

### Cycle 14 — Validate CLI ports early (2026-08-09)

**Why this won:** `argparse` converted ports to integers but accepted `0`,
negative numbers, and values above 65535. Those inputs triggered environment
setup before Uvicorn or Docker eventually failed with less useful errors.

**Plan and success criteria**

1. Parse and range-check ports in the CLI boundary.
2. Accept both numeric strings and integers across the full TCP range.
3. Reject malformed and out-of-range values before invoking any workflow.

**Changes**

- Added `valid_port` using `argparse.ArgumentTypeError` with clear messages.
- Wired `--port` to the validator.
- Added three valid and four invalid boundary cases.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 29 passed (up from 22).
- `.venv/bin/python run.py --help`: passed.
- Python compilation and CRLF-aware diff checks passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 5/10 | 9/10 | All invalid TCP port classes fail at parsing |
| Test coverage / verifiability | 5/10 | 9/10 | Both range boundaries and malformed input are covered |
| Maintainability | 7/10 | 8/10 | Port policy is one named parser function |
| Performance / resources | 7/10 | 8/10 | Invalid runs stop before environment setup |
| Developer experience | 5/10 | 9/10 | argparse reports the constraint immediately |

**Lesson / process improvement:** Validate CLI values at parsing time when the
constraint is intrinsic. This prevents expensive setup from masking a simple
input error.

**Next opportunity:** Separate production inference requirements from notebook,
plotting, dataset, and training dependencies so Docker installs only what the
API needs.

### Cycle 15 — Split production inference dependencies (2026-08-09)

**Why this won:** The Docker image installed the full research workspace:
Jupyter/JupyterLab, plotting, pandas/scikit-learn, dataset download tooling, and
test packages. None are used by the FastAPI inference process, increasing build
time, image size, dependency surface, and vulnerability exposure.

**Plan and success criteria**

1. Define an explicit pinned inference manifest containing only model loading,
   preprocessing, and API-serving dependencies.
2. Make Docker consume that manifest while preserving full training setup.
3. Contract-test required/forbidden packages and validate dependency resolution
   plus Dockerfile structure.

**Changes**

- Added `requirements-api.txt` for TensorFlow/Keras, FastAPI/Uvicorn,
  multipart parsing, Pillow, and NumPy.
- Switched Docker and its context allowlist from the full `requirements.txt` to
  the inference manifest.
- Added tests for runtime inclusions, training/test exclusions, and Dockerfile
  usage; documented all three dependency scopes.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 31 passed (up from 29).
- `pip install --dry-run -r requirements-api.txt`: dependency resolution
  succeeded against the verified environment.
- `docker build --check .`: no warnings.
- Python compilation and CRLF-aware diff checks passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 7/10 | 9/10 | Docker dependencies now correspond directly to imported runtime modules |
| Test coverage / verifiability | 6/10 | 9/10 | Runtime contents and Docker consumption are contract-tested |
| Maintainability | 4/10 | 9/10 | Training, testing, and inference have distinct manifests |
| Performance / resources | 3/10 | 9/10 | Large unused research packages are removed from image installation |
| Security / safety | 5/10 | 8/10 | Production dependency surface is materially smaller |

**Lesson / process improvement:** Dependency manifests should reflect deployment
roles, not mirror a development workstation. Test both what a runtime manifest
contains and what it deliberately excludes.

**Next opportunity:** Correct `load_class_mapping` so malformed JSON/structure
raises its documented `ValueError` rather than being caught and rewrapped as a
generic `RuntimeError`, with isolated path-based fixtures.

### Cycle 16 — Preserve class-mapping error contracts (2026-08-09)

**Why this won:** `load_class_mapping` documented malformed JSON/structure as
`ValueError`, but its broad final handler caught the structure validation error
and changed it to `RuntimeError`. Callers and tests could not reliably
distinguish invalid user artifacts from operational I/O failures.

**Plan and success criteria**

1. Separate pure minimum-structure validation from file loading.
2. Preserve `FileNotFoundError`, use `ValueError` for malformed JSON/shape, and
   reserve `RuntimeError` for other read failures.
3. Test all branches with temporary files and no TensorFlow import.

**Changes**

- Added `validate_class_mapping` and an optional path parameter for isolated
  loader verification.
- Opened mapping JSON explicitly as UTF-8 and preserved validation errors.
- Added valid, invalid-JSON, invalid-structure, and missing-file tests.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 35 passed (up from 31).
- Python compilation and CRLF-aware diff checks passed.
- Missing keys now raise the documented `ValueError`; missing files remain
  `FileNotFoundError`.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 4/10 | 9/10 | Each artifact failure class has a stable exception type |
| Test coverage / verifiability | 2/10 | 9/10 | Four file/format paths are directly tested |
| Maintainability | 5/10 | 8/10 | Pure structure validation is separated from I/O |
| Performance | 9/10 | 9/10 | No material runtime change |
| Developer experience | 4/10 | 8/10 | Startup failures now retain actionable categories |

**Lesson / process improvement:** Do not place deliberate validation exceptions
under a catch-all wrapper. Test documented exception types as part of the public
artifact-loading contract.

**Next opportunity:** Make `load_model` aggregate deserialization failures and
raise `RuntimeError` when candidates exist but are corrupt/incompatible, while
retaining `FileNotFoundError` only for truly absent artifacts.

### Cycle 17 — Distinguish missing and corrupt models (2026-08-09)

**Why this won:** The loader tried every existing model candidate, printed each
deserialization failure, then raised `FileNotFoundError` claiming no model was
found. That diagnosis sent operators toward retraining or path fixes instead of
the actual compatibility/corruption problem.

**Plan and success criteria**

1. Keep preference/fallback behavior across all supported paths.
2. Aggregate failures and raise `RuntimeError` when files exist but none load.
3. Reserve `FileNotFoundError` for an actually empty candidate set and test all
   branches without importing TensorFlow.

**Changes**

- Added optional project-root and deserializer injection to `load_model`.
- Deferred TensorFlow import until no injected loader is supplied.
- Aggregated per-path exception types/messages before raising a load failure.
- Added preference, fallback, corrupt-candidates, and missing-candidates tests.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 39 passed (up from 35).
- Python compilation and CRLF-aware diff checks passed.
- Existing corrupt candidates now name every failed path under `RuntimeError`;
  an empty temporary root still raises `FileNotFoundError`.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 3/10 | 9/10 | Missing and unloadable states now have distinct outcomes |
| Test coverage / verifiability | 2/10 | 9/10 | Preference, fallback, aggregate failure, and absence are covered |
| Maintainability | 5/10 | 8/10 | Model discovery/deserialization is dependency-injectable |
| Performance | 8/10 | 8/10 | Existing fallback loop is unchanged |
| Observability / DX | 3/10 | 9/10 | Startup reports every rejected artifact and cause |

**Lesson / process improvement:** Exhaustive fallback logic must preserve the
difference between “nothing existed” and “everything failed.” Inject heavy
deserializers to test that state machine cheaply.

**Next opportunity:** Enforce a decoded pixel-count ceiling in image
preprocessing so compressed image bombs cannot bypass the byte-size upload
limit and exhaust memory before resizing.

### Cycle 18 — Bound decoded image dimensions (2026-08-09)

**Why this won:** The HTTP boundary limited compressed uploads to 10 MB, but a
small file can declare or decompress into hundreds of millions of pixels.
Pillow opened the image before resize, so conversion/NumPy allocation could
consume disproportionate memory.

**Plan and success criteria**

1. Validate positive dimensions immediately after opening the image.
2. Reject more than 50 megapixels before conversion, resize, or array creation.
3. Cover exact-boundary, zero/negative, and over-limit dimensions cheaply.

**Changes**

- Added `MAX_DECODED_PIXELS` and `validate_image_dimensions`.
- Called the guard immediately after `Image.open` and before decoded-pixel work.
- Added four dimension boundary cases.

**Verification evidence**

- `.venv/bin/python -m pytest -q`: 43 passed (up from 39).
- Python compilation and CRLF-aware diff checks passed.
- Exactly 50,000,000 pixels passes; invalid dimensions and 50,005,000 pixels
  fail before allocation.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 6/10 | 9/10 | Decode resource policy is explicit and deterministic |
| Test coverage / verifiability | 4/10 | 9/10 | Boundary and malformed dimensions have pure tests |
| Maintainability | 6/10 | 8/10 | One named constant/helper owns the policy |
| Performance / resources | 3/10 | 9/10 | Excessive images stop before RGB/NumPy allocation |
| Security / safety | 4/10 | 9/10 | Compressed image bombs cannot rely only on byte size |

**Lesson / process improvement:** Validate both encoded bytes and decoded
dimensions for media uploads. Put dimension checks before operations that force
pixel decoding or large allocations.

**Next opportunity:** Extract prediction decoding and reject non-2D, wrong-width,
non-finite, or empty score vectors before ranking and class lookup.

### Cycle 19 — Validate prediction output before ranking (2026-08-10)

**Why this won:** The endpoint indexed `predictions[0]` and ranked it directly.
A one-dimensional, empty, multi-row, wrong-width, non-numeric, NaN, or infinite
model result could therefore raise an opaque indexing/mapping error or return a
misleading non-finite ranking. This was the workspace's highest-impact
small/low-risk open correctness item after the portfolio cycles.

**Plan and success criteria**

1. Extract score decoding into a pure helper with no TensorFlow dependency.
2. Accept exactly one non-empty numeric row matching contiguous mapping keys.
3. Reject every non-finite score before `argmax`/sorting and preserve the safe
   generic HTTP 500 boundary.
4. Prove valid ranking and every malformed class with model-free tests.

**Changes**

- Added `decode_predictions`, which converts tensor-like output to numeric
  NumPy scores, validates rank/batch/width/mapping/finiteness, bounds top-k to
  the available classes, and only then ranks results.
- Replaced inline endpoint indexing and sorting with the validated helper and
  removed the now-unneeded NumPy import from the API module.
- Added one valid ranking case, nine malformed-output cases, four invalid top-k
  cases, and an endpoint regression proving non-finite diagnostics remain
  server-side.
- Documented the model-output contract in the API response section.

**Verification evidence**

- Test-first evidence: the focused suite initially failed to import the absent
  `decode_predictions` helper.
- `.venv/bin/python -m pytest -q`: 58 passed in 1.10s (up from 43 in 1.42s).
- Invalid fixtures cover one- and three-dimensional arrays, multiple rows,
  empty width, mapping-width/key mismatch, NaN, infinity, and non-numeric
  scores; invalid top-k values also fail explicitly.
- The endpoint returns only `{"detail": "Prediction failed"}` for a NaN model
  output; the word `finite` does not reach the client.
- Lightweight-import/dependency contracts: three focused tests passed without
  importing TensorFlow.
- `.venv/bin/python -m compileall -q api tests run.py` and CRLF-aware
  `git diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 4/10 | 9/10 | Invalid output cannot reach ranking or class lookup |
| Test coverage / verifiability | 3/10 | 10/10 | Every specified malformed class plus HTTP containment is exercised |
| Maintainability | 6/10 | 9/10 | One pure helper owns prediction shape, mapping, and ranking policy |
| Performance / resources | 8/10 | 8/10 | Validation is linear over 196 scores and reuses one converted array |
| Security / robustness | 6/10 | 9/10 | NaN/infinity and internal diagnostics fail closed behind the API boundary |

**Lesson / process improvement:** Validate untrusted model output as strictly as
client input. Extract decoding before testing it so malformed tensor contracts
stay cheap, deterministic, and independent of trained weights.

**Next opportunity:** Validate the loaded model's input shape against the
preprocessor's fixed `(1, 224, 224, 3)` contract during lifespan, so an
incompatible artifact never reports ready and fails every request.

### Cycle 20 — Validate model input compatibility at startup (2026-08-10)

**Why this won:** Lifespan validated only output width and label compatibility.
A model expecting a second input, a different tensor rank/resolution/channel
count, or a fixed batch larger than one could still report healthy and then
fail every request when given the preprocessor's `(1, 224, 224, 3)` array.

**Plan and success criteria**

1. Define one shared preprocessed image shape and use it during resize.
2. Accept single-input model shapes whose fixed dimensions can consume that
   array, including dynamic dimensions and batch size one.
3. Reject missing, multi-input, wrong-rank, and incompatible fixed dimensions
   before readiness globals are published.
4. Test the preprocessor's actual output shape and every startup branch without
   TensorFlow or model weights.

**Changes**

- Added `PREPROCESSED_IMAGE_SHAPE = (1, 224, 224, 3)` and made image resizing
  derive its dimensions from that contract.
- Extended `validate_runtime_artifacts` to require one rank-four input and to
  compare every fixed dimension against the preprocessed array while allowing
  `None` as a compatible dynamic dimension.
- Added five incompatible-shape, three compatible-shape, and one real
  preprocessing-output test.
- Updated readiness documentation to cover both input/output compatibility.

**Verification evidence**

- Test-first evidence: all five incompatible input fixtures initially reached
  readiness validation without raising.
- `.venv/bin/python -m pytest -q`: 67 passed in 1.10s (up from 58).
- Rejected shapes cover absent metadata, multi-input models, rank three,
  299×299 images, and fixed batch 32; accepted shapes cover dynamic batch,
  batch one, and dynamic spatial dimensions with three channels.
- A real in-memory PNG preprocesses to the shared `(1, 224, 224, 3)` shape.
- `.venv/bin/python -m compileall -q api tests run.py`, lightweight import
  contracts, and CRLF-aware `git diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 5/10 | 9/10 | Incompatible inputs abort lifespan before readiness |
| Test coverage / verifiability | 5/10 | 10/10 | Shape compatibility and actual preprocessing output share a tested constant |
| Maintainability | 6/10 | 9/10 | Resize and startup validation cannot silently drift apart |
| Performance / resources | 9/10 | 9/10 | Four constant-time dimension checks run once at startup |
| Security / robustness | 7/10 | 9/10 | Multi-input and malformed shape metadata fail closed |

**Lesson / process improvement:** Validate both sides of an inference boundary:
the preprocessor's actual tensor and the model's declared input. A shared
constant prevents configuration drift more reliably than duplicated literals.

**Next opportunity:** Apply JPEG EXIF orientation before RGB conversion and
resize, with an in-memory rotated-photo fixture proving mobile uploads reach
the model upright.

### Cycle 21 — Honor JPEG EXIF orientation (2026-08-10)

**Why this won:** Phone cameras commonly store landscape pixel arrays with an
EXIF orientation tag instead of rewriting pixels. The preprocessor opened,
converted, and resized stored pixels directly, so a valid portrait upload could
reach the classifier sideways and reduce prediction quality without any error.

**Plan and success criteria**

1. Build an asymmetric in-memory JPEG whose orientation tag moves left/right
   colors to top/bottom when displayed correctly.
2. Keep decoded pixel-count validation before any transpose/decode work.
3. Apply orientation before RGB conversion and resize.
4. Prove spatial color placement, then run every model-free contract.

**Changes**

- Applied `ImageOps.exif_transpose` immediately after the decoded-dimension
  guard and before color conversion or resize.
- Added an orientation-6 JPEG fixture with red stored pixels on the left and
  blue on the right; the displayed/model tensor must have red on top and blue
  on the bottom.
- Documented phone-photo orientation handling in the API response contract.

**Verification evidence**

- Test-first evidence: without EXIF handling, top red and blue channel means
  were both approximately `0.498`, proving the sideways stored layout reached
  resize unchanged.
- After the fix, the focused orientation test passed and verified red-dominant
  top pixels plus blue-dominant bottom pixels.
- `.venv/bin/python -m pytest -q`: 68 passed in 1.26s (up from 67).
- `.venv/bin/python -m compileall -q api tests run.py`, lightweight import
  contracts, and CRLF-aware `git diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 6/10 | 9/10 | Model pixels now match the photo's displayed orientation |
| Test coverage / verifiability | 5/10 | 10/10 | A spatially asymmetric real JPEG proves the transform |
| Maintainability | 8/10 | 9/10 | Standard Pillow normalization replaces custom orientation logic |
| Performance / resources | 8/10 | 8/10 | One bounded transpose occurs before the existing 224×224 resize |
| Security / robustness | 9/10 | 9/10 | Pixel-count validation remains ahead of transpose/decode allocation |

**Lesson / process improvement:** Image validity is not the same as semantic
orientation. Use a spatially asymmetric fixture so a preprocessing test proves
pixel placement rather than merely output shape.

**Next opportunity:** Inspect Pillow's decoded `image.format` and reject
anything except JPEG/PNG even when a client supplies an allowed MIME header,
with renamed WebP/GIF fixtures at the utility and endpoint boundaries.

### Cycle 22 — Enforce decoded image formats (2026-08-11)

**Why this won:** The HTTP boundary allowed only JPEG/PNG MIME claims, but
Pillow decoded any installed image format. Genuine GIF or WebP bytes renamed
and uploaded as an allowed type therefore reached model inference despite the
documented contract.

**Plan and success criteria**

1. Use real in-memory GIF and WebP payloads rather than signature stubs.
2. Reject unsupported formats immediately after Pillow identifies them and
   before EXIF, color conversion, resize, or NumPy allocation.
3. Keep the endpoint's existing generic client error and preserve valid
   JPEG/PNG behavior.
4. Pass the full model-free gate without TensorFlow or trained weights.

**Changes**

- Added an explicit `JPEG`/`PNG` decoded-format allowlist at the preprocessing
  boundary.
- Added two utility regressions proving GIF and WebP bytes fail closed.
- Added two endpoint regressions proving allowed MIME claims cannot disguise
  either payload and decoder details remain behind the generic HTTP 400.
- Documented that both upload MIME and decoded file format are enforced.

**Verification evidence**

- Test-first evidence: all four new cases failed; preprocessing accepted both
  formats and the disguised endpoint uploads returned HTTP 200.
- Focused red/green run: 4 passed after the format boundary was added.
- `.venv/bin/python -m pytest -q`: 72 passed in 1.48s (up from 68).
- `.venv/bin/python -m compileall -q api tests run.py prediction_example.py`:
  passed.
- `.venv/bin/python -m pip check`: no broken requirements found.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 6/10 | 9/10 | Implementation now matches the documented JPEG/PNG input contract |
| Test coverage / verifiability | 6/10 | 10/10 | Real encoded fixtures exercise both utility and HTTP boundaries |
| Maintainability | 8/10 | 9/10 | One named decoded-format policy sits beside decode resource policy |
| Performance / resources | 8/10 | 9/10 | Unsupported formats stop before pixel transforms and array allocation |
| Security / robustness | 6/10 | 9/10 | Client-controlled MIME metadata can no longer widen accepted formats |

**Lesson / process improvement:** Validate media claims at two independent
boundaries: transport metadata controls early rejection, while decoder-derived
metadata enforces the actual file contract. Real alternate-format fixtures
prove the complete path more reliably than handcrafted magic bytes.

**Next opportunity:** Require startup model output metadata to describe exactly
one rank-two score row per preprocessed image, so rank-three or fixed multi-row
artifacts fail before readiness rather than on every prediction.

### Cycle 23 — Validate model output metadata at startup (2026-08-11)

**Why this won:** Runtime decoding required exactly one two-dimensional score
row, but startup inspected only the last output dimension. Rank-one,
rank-three, and fixed multi-row output shapes could therefore report healthy
and then fail every request.

**Plan and success criteria**

1. Reproduce each incompatible metadata shape without TensorFlow or weights.
2. Require rank two plus a dynamic or exactly-one batch dimension before width
   and mapping validation.
3. Explicitly retain dynamic-batch and single-batch compatible shapes.
4. Preserve all existing runtime, lifecycle, image, and deployment contracts.

**Changes**

- Extended `validate_runtime_artifacts` to require rank-two output metadata.
- Required a dynamic or integer batch dimension of one, with a distinct error
  for malformed batch metadata.
- Added four incompatible-shape regressions and two compatible-shape controls.
- Documented the startup output-metadata contract next to runtime score rules.

**Verification evidence**

- Test-first evidence: all three initial incompatible shapes passed startup
  validation without raising.
- Focused validation: seven output-shape, input-shape, and lifecycle checks
  passed after implementation.
- `.venv/bin/python -m pytest -q`: 78 passed in 1.24s (up from 72).
- `.venv/bin/python -m compileall -q api tests run.py prediction_example.py`:
  passed.
- `.venv/bin/python -m pip check`: no broken requirements found.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 5/10 | 9/10 | Output metadata incompatible with the decoder cannot become ready |
| Test coverage / verifiability | 6/10 | 10/10 | Four rejection and two acceptance shapes define the startup boundary |
| Maintainability | 8/10 | 9/10 | Startup and runtime now share the same rank/batch assumptions |
| Performance / resources | 9/10 | 9/10 | Constant-time checks run once during lifespan startup |
| Security / robustness | 8/10 | 9/10 | Malformed artifact metadata fails closed before request traffic |

**Lesson / process improvement:** Startup compatibility must validate the full
tensor contract, not just the dimension used for class count. Pair negative
fixtures with explicit positive controls so stricter readiness checks cannot
silently narrow supported dynamic models.

**Next opportunity:** Modernize GitHub Actions and add repository-owned policy
contracts for least privilege, bounded execution, supported action runtimes,
and the complete lightweight verification gate; the latest hosted run passed
with a Node 20 deprecation annotation that this cycle did not hide.

### Cycle 24 — Modernize and policy-test CI (2026-08-11)

**Why this won:** Two consecutive hosted runs passed with a Node 20 deprecation
annotation because checkout v4 and setup-python v5 were being forced onto a
newer runtime. The workflow also had no explicit permissions, timeout,
concurrency policy, dependency consistency check, or repository-owned guard
against regression.

**Plan and success criteria**

1. Verify current action majors against official upstream release data.
2. Encode supported actions, least privilege, stale-run cancellation, timeout,
   interpreter/cache pins, and every local gate command in a fast policy test.
3. Make the workflow execute its own policy before the full suite.
4. Pass locally, then require a successful hosted run with zero annotations.

**Changes**

- Upgraded `actions/checkout` and `actions/setup-python` from v4/v5 to their
  current v7 Node 24 majors.
- Granted only read access to repository contents, grouped runs by workflow/ref,
  canceled superseded runs, and bounded the job at 10 minutes.
- Added dependency consistency and expanded compilation to include
  `prediction_example.py`.
- Added one model-free policy test with 19 assertions covering triggers,
  permissions, concurrency, timeout, supported runtimes, caching, install,
  self-enforcement, full pytest, dependency, and compilation commands.
- Documented the complete local gate and workflow-policy coverage.

**Verification evidence**

- Official GitHub releases identified checkout v7.0.1 and setup-python v7.0.0
  as the current releases on 2026-07-20.
- Test-first evidence: the policy test failed first on the absent read-only
  permissions block against the original workflow.
- Focused policy run: 1 test / 19 assertions passed after modernization.
- `.venv/bin/python -m pytest -q`: 79 passed in 1.20s (up from 78).
- `.venv/bin/python -m compileall -q api tests run.py prediction_example.py`:
  passed.
- `.venv/bin/python -m pip check`: no broken requirements found.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 8/10 | 9/10 | CI now executes the same complete gate documented for local use |
| Test coverage / verifiability | 6/10 | 10/10 | Nineteen repository-owned assertions prevent workflow-policy drift |
| Maintainability | 7/10 | 9/10 | Supported runtime and gate obligations are explicit and executable |
| Performance / resources | 6/10 | 9/10 | Stale same-ref runs cancel and every job is time-bounded |
| Security / safety | 6/10 | 9/10 | CI token access is explicitly read-only and deprecated runtimes are removed |

**Lesson / process improvement:** A passing hosted workflow can still carry
actionable lifecycle debt. Treat annotations as verification failures in the
improvement rubric, confirm upstream versions from primary sources, and encode
the desired workflow shape locally so future updates cannot silently regress it.

**Next opportunity:** Audit the production/test dependency pins for disclosed
vulnerabilities, then update the smallest affected set in fresh lightweight and
production-resolution environments before changing any version constraint.

### Cycle 25 — Audit and refresh deploy/test dependencies (2026-08-11)

**Why this won:** Exact pins from 2023–2025 made `pip check` pass while hiding
known security advisories. A Python 3.12 audit found 43 vulnerability records in
the lightweight test graph, 59 in production resolution, and 79 in the complete
training workspace.

**Plan and success criteria**

1. Resolve every manifest with the project's Python 3.12 baseline.
2. Update the smallest coherent deploy/test set and keep shared pins aligned.
3. Require warning-free tests in a brand-new environment.
4. Load and readiness-validate the real ignored model before accepting any
   Keras change.
5. Report, rather than conceal, compatibility-blocked or out-of-scope findings.

**Changes**

- Updated FastAPI to 0.139.2, explicitly pinned Starlette 1.6.0, updated
  python-multipart to 0.0.32 and Pillow to 12.3.0 across their applicable
  manifests, and updated pytest to 9.1.1.
- Replaced the deprecated test-client dependency `httpx` with verified
  `httpx2` 2.7.0.
- Added dependency-version parsing and an alignment/security-baseline contract;
  model-free coverage increased from 79 to 80 tests.
- Promoted warnings to errors in the documented and hosted full test command.
- Kept Keras at 3.10.0 after real artifact failures under every tested fixed
  release, and documented the constraint in README, manifests, and AGENTS.md.

**Verification evidence**

- Process failure: an initial isolated audit defaulted to Python 3.14, where
  Pillow 10.0.0 could not build and TensorFlow 2.19 had no distribution. Pinning
  `uvx --python 3.12` produced valid results and is now the reusable method.
- Test-first evidence: dependency policy failed on the missing Starlette pin
  and stale versions before manifest updates.
- First fresh environment: 80 tests passed but exposed Starlette's deprecated
  `httpx` path. A second clean environment with `httpx2` passed 80 tests with
  warnings treated as errors in 1.00s and had no broken requirements.
- Existing environment: 80 tests passed with warnings treated as errors in
  1.28s; compilation and CRLF-aware diff checks passed.
- `requirements-test.txt`: 43 vulnerability records before, zero after.
- `requirements-api.txt`: 59 records across four packages before, 17 Keras-only
  records after; Pillow, multipart, and Starlette findings are eliminated.
- `requirements.txt`: 79 records across nine packages before, 36 across five
  after; the remaining 19 non-Keras tooling records are isolated for Cycle 26.
- Real production smoke: the updated API stack with TensorFlow 2.19/Keras 3.10
  loaded `best_car_model.keras` and validated `(None, 224, 224, 3)` input,
  `(None, 196)` output, and the 196-class mapping.
- Rejected candidates: Keras 3.11.3, 3.12.3, and 3.15.0 each failed both `.keras`
  and H5 artifacts because a dense layer received two tensors; reverting to
  3.10.0 restored successful loading.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 7/10 | 9/10 | Fresh and real-model paths prevent a security update from breaking inference |
| Test coverage / verifiability | 7/10 | 10/10 | Exact pins, fresh installs, warnings-as-errors, audits, and real artifacts agree |
| Maintainability | 6/10 | 9/10 | Shared audited pins and the Keras exception are explicit and executable |
| Performance / resources | 8/10 | 8/10 | Runtime behavior is unchanged; audits/install smokes are development-only |
| Security / safety | 3/10 | 8/10 | Test graph is clean and production findings are reduced to a constrained trusted-artifact dependency |

**Lesson / process improvement:** Dependency resolution is not runtime
compatibility. Match the audit interpreter to production, use fresh installs to
surface deprecations, and require real artifact smoke tests for ML serialization
libraries. When the fixed version breaks the model, preserve the working pin,
document the exposure, and create a migration dependency instead of shipping a
nominally clean but unusable service.

**Next opportunity:** Upgrade and audit the remaining training-workspace tools
(`requests`, `notebook`, `jupyterlab`, and `python-dotenv`) in an isolated Python
3.12 environment, preserving notebook kernel startup and keeping the Keras
compatibility constraint separate.

### Cycle 26 — Clear training-workspace tooling advisories (2026-08-11)

**Why this won:** Cycle 25 isolated 19 non-Keras vulnerability records to the
training/notebook manifest. Updating that separate compatibility domain could
remove the remaining fixable exposure without touching the working model or
the Keras migration constraint.

**Plan and success criteria**

1. Pin the audited requests, Notebook, JupyterLab, and dotenv releases plus a
   pytest-asyncio release compatible with pytest 9.
2. Resolve and install the complete Python 3.12 workspace from scratch.
3. Run warning-free tests, compilation, imports, Jupyter entry points, kernel
   discovery, and actual kernel execution.
4. Re-audit the full graph and require Keras to be the only remaining finding.

**Changes**

- Updated requests to 2.34.2, Notebook to 7.6.1, JupyterLab to 4.6.3,
  python-dotenv to 1.2.2, and pytest-asyncio to 1.4.0.
- Added `httpx2` 2.7.0 to the full workspace after the fresh install exposed
  that FastAPI tests could not use Starlette's non-deprecated client path.
- Added a training-tool audit baseline and extended shared `httpx2` alignment;
  model-free coverage increased from 80 to 81 tests.

**Verification evidence**

- Test-first evidence: the new tooling contract failed first on requests 2.31.0.
- Audit: `requirements.txt` fell from 36 records across five packages to the 17
  documented Keras-only records; all 19 tooling records are eliminated.
- First full-environment attempt: package resolution and Jupyter imports passed,
  but warnings-as-errors stopped test collection because `httpx2` existed only
  in `requirements-test.txt`. Adding it to the workspace fixed the drift.
- Fresh full Python 3.12 environment: 81 tests passed warning-free in 1.02s,
  compilation passed, and `pip check` reported no broken requirements.
- Import smoke reported requests 2.34.2, Notebook 7.6.1, JupyterLab 4.6.3, and
  pytest-asyncio 1.4.0.
- `jupyter --version`, Notebook/Lab version entry points, and kernel discovery
  succeeded; a real kernel started, became ready, executed `assert 1 + 1 == 2`,
  returned `status: ok`, and shut down.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 7/10 | 9/10 | Full-manifest tests and a live kernel prove more than resolver compatibility |
| Test coverage / verifiability | 7/10 | 10/10 | Pin contracts, fresh install, imports, entry points, execution, and audit agree |
| Maintainability | 7/10 | 9/10 | Shared test-client and tooling pins are explicit across their applicable manifests |
| Performance / resources | 8/10 | 8/10 | Runtime inference is unchanged; verification cost is development-only |
| Security / safety | 6/10 | 9/10 | Every fixable tooling advisory is removed; only the documented Keras constraint remains |

**Lesson / process improvement:** A complete environment can fail even when its
individual tools import: run the project's own warning-strict suite inside that
environment. For notebook stacks, validate discovery and execute code in a real
kernel; version commands alone do not prove the kernel path works.

**Next opportunity:** Locally, require an exact typed bijection between
`index_to_class` and `class_to_index` so malformed mapping metadata fails at
startup. At workspace scope, rotate to another repository before returning to
this service, avoiding diminishing returns after five consecutive car-service
cycles.

### Cycle 27 — Require an exact typed class mapping (2026-08-11)

**Why this won:** Startup verified required mapping keys, output width,
contiguous forward indices, and the pairs it happened to visit, but it still
accepted extra reverse entries and labels or reverse indices with weak types.
Python equality made this especially subtle: `False == 0` and `0.0 == 0` let
invalid JSON-like metadata masquerade as an integer inverse.

**Plan and success criteria**

1. Specify the full persisted mapping contract with failing utility and
   readiness fixtures.
2. Enforce one shared exact bijection at both file-load and runtime-artifact
   boundaries without duplicating policy.
3. Keep the complete warning-strict model-free suite, dependency graph,
   compilation, and whitespace checks green.

**Changes**

- Expanded `validate_class_mapping` to require a non-empty object with
  contiguous string indices, unique non-blank string labels, identical reverse
  keys, strict integer reverse values, and exact inverse pairs.
- Reused that validator in `validate_runtime_artifacts` and removed its weaker,
  duplicated mapping checks; model shape/width compatibility remains local to
  the runtime boundary.
- Added ten pure mapping cases and three runtime-artifact cases for malformed
  containers, sparse indices, label types, blanks, duplicates, extra entries,
  booleans, and floats.
- Documented the readiness mapping contract in the API README.

**Verification evidence**

- Test-first evidence: all 13 new malformed-mapping cases failed against the
  prior validators because no `ValueError` was raised.
- Focused regression: 14 mapping-format/inverse tests passed after the change.
- `.venv/bin/python -m pytest -q -W error`: 94 passed in 1.32s (up from 81).
- `.venv/bin/python -m pip check`: no broken requirements found.
- `.venv/bin/python -m compileall -q api tests run.py prediction_example.py`:
  passed.
- `git -c core.whitespace=cr-at-eol diff --check`: passed.

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 6/10 | 10/10 | Every mapping accepted at load is now a typed bijection accepted at readiness |
| Test coverage / verifiability | 8/10 | 10/10 | Thirteen adversarial fixtures cover both validation boundaries |
| Maintainability | 7/10 | 9/10 | One pure helper owns mapping metadata policy |
| Performance / resources | 9/10 | 9/10 | Linear validation remains startup-only over 196 entries |
| Security / robustness | 7/10 | 9/10 | Extra and coercion-like metadata can no longer enter inference state |

**Lesson / process improvement:** Validate inverse metadata as an exact key-set
relationship before checking individual pairs, and use exact type checks where
Python's numeric equality would admit booleans or floats. A load-time validator
should also be reused at readiness so injected or alternate loading paths
cannot bypass the persisted-data contract.

**Next opportunity:** At workspace scope, rotate to the least recently improved
repository. When returning here, bound synchronous model inference with a
measured, model-safe admission policy before attempting the higher-risk Keras
artifact migration.

### Cycle 28 — Isolate and serialize synchronous inference (2026-08-11)

**Why this won:** `/predict` was declared async but ran image preprocessing and
TensorFlow directly on the event-loop thread. One slow model call therefore
stalled readiness traffic, while deployment patterns with concurrent request
loops could enter the shared Keras model simultaneously. This was the highest
ranked remaining reliability opportunity and could be fixed without changing
the response contract.

**Plan and success criteria**

1. Reproduce event-loop starvation with a synchronized request-level test.
2. Move CPU/model work off-loop and allow at most one active prediction per
   service process.
3. Prove health remains responsive during a blocked model call, concurrent
   predictions do not overlap, and both predictions still complete normally.
4. Run the real ignored artifact through the new threaded path plus the full
   warning-strict lightweight gate.

**Changes**

- Offloaded Pillow/NumPy preprocessing and TensorFlow prediction through
  Starlette's worker-thread boundary.
- Added a one-token asynchronous semaphore around model prediction and decode,
  reconstructed for each successful application lifespan.
- Extracted `run_model_inference` as the synchronous unit executed by the
  worker and documented the per-process concurrency policy.
- Added a synchronized two-request test with a deliberately blocked model,
  concurrent health probe, and peak-active-call measurement.

**Verification evidence**

- Test-first evidence: the new request contract failed because `/health` could
  not return until the blocked model was released.
- Focused regression passed in 0.24s: health returned while inference remained
  blocked, two prediction responses succeeded, and peak model concurrency was
  exactly one.
- `.venv/bin/python -m pytest -q -W error`: 95 passed in 1.21s (up from 94).
- `.venv/bin/python -m pip check`: no broken requirements found.
- `.venv/bin/python -m compileall -q api tests run.py prediction_example.py`:
  passed; CRLF-aware whitespace validation passed.
- Real artifact smoke loaded the 196-class `best_car_model.keras`, validated
  its input/output metadata, and completed prediction and top-five decode from
  a worker thread (`Dodge Challenger SRT8 2011`, confidence `0.890301`).

**Scores (change-specific)**

| Dimension | Before | After | Evidence |
|---|---:|---:|---|
| Correctness / reliability | 5/10 | 9/10 | Readiness no longer shares a blocked event loop and model calls cannot overlap |
| Test coverage / verifiability | 7/10 | 10/10 | Synchronization-driven request coverage distinguishes both failure modes |
| Maintainability | 7/10 | 9/10 | One named worker function and one explicit capacity constant own the policy |
| Performance / resources | 4/10 | 9/10 | Event-loop availability is preserved without increasing model concurrency |
| Security / robustness | 7/10 | 9/10 | Concurrent access to a mutable native ML runtime is eliminated per process |
| Developer / user experience | 6/10 | 9/10 | Health and unrelated async traffic stay responsive during slow inference |

**Lesson / process improvement:** Concurrency tests should coordinate on events
inside the slow operation, not infer overlap from benchmark timings. Combining
an in-flight health probe with a measured peak-call counter distinguishes an
offload-only change from a serialization-only change. A real artifact smoke is
still necessary because fake models cannot prove TensorFlow works from the
selected worker context.

**Next opportunity:** Locally, bound how long a request may wait to acquire the
prediction lane and define explicit overload behavior without abandoning an
already-running TensorFlow thread. At workspace scope, rotate to the portfolio
repository before returning here so improvements continue across projects.
