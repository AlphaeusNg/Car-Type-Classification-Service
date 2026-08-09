# Car Classification Service Continuous Improvement Progress

This file tracks current status, prioritized opportunities, verification, and
completed autonomous improvement cycles.

## Current state

- FastAPI inference service for a 196-class TensorFlow/Keras model.
- Model and dataset artifacts are intentionally not tracked in Git.
- Baseline after Cycle 16: 35 model-free tests cover the API boundary,
  lifecycle, lightweight import, and model artifact discovery/build command;
  GitHub Actions runs them without TensorFlow or trained weights.

## Opportunity backlog

| Priority | Opportunity | Category | Impact | Effort / risk | Evidence / dependencies | Status |
|---|---|---|---|---|---|---|
| 1 | Distinguish corrupt model artifacts from missing artifacts | Correctness / observability | High: loader currently ends with `FileNotFoundError` after every existing candidate fails to deserialize | Small / low | Inject a model loader to test failure aggregation without TensorFlow | Next |
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
