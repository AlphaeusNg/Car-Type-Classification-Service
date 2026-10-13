# Classifier feature roadmap

Researched 2026-10-14. Repository baseline: 3b551b1.

## Needs and evidence

The standalone prediction CLI already validates images, authenticates selected
artifacts and decodes probability scores. Its only successful output is human
text; shared loaders also print progress to stdout. Scripts cannot consume a
prediction without parsing changing text or starting the HTTP service. This is
repository evidence, not a claim that user interviews have been conducted.

Python's [JSON documentation](https://docs.python.org/3.12/library/json.html)
provides serialization of a single object and strict finite-number encoding via
`allow_nan=False`. Its [argparse documentation](https://docs.python.org/3.12/library/argparse.html)
supports validating integer choices before running inference. The
[contextlib documentation](https://docs.python.org/3.12/library/contextlib.html#contextlib.redirect_stdout)
shows redirection of Python stdout and warns it changes global state and does
not capture subprocess output. This is suitable for the single-process CLI,
not a change to the concurrent HTTP API.

## Selected feature: structured local predictions

Add `--json` for one versioned JSON object on stdout and `--top-k` for the
number of ranked candidates (1–196, default 5). Reuse the existing decoder's
probability validation, stable tie order and truncation to the mapping size.
JSON v1 fields: `schema_version`, `status`, `predicted_class`, `class_index`,
`confidence`, `top_predictions` (objects with `class` and `confidence`). Scores
remain softmax scores, not calibrated correctness probabilities.

Acceptance criteria:

- Successful JSON invocation emits exactly one complete document and exits 0.
- Python loader/inference diagnostics go to stderr; no progress text contaminates
  stdout. Native runtime diagnostics are outside Python stdout redirection.
- Missing/invalid images, model errors and serialization errors emit no result
  to stdout, explain failure on stderr and exit 1. Invalid CLI options exit 2.
- JSON is serialized before output and rejects NaN/infinity.
- Top-k is validated before model loading; actual inference reuses the shared
  ranking. Stable ties and a mapping smaller than top-k work correctly.
- Existing human mode, Python return keys and HTTP response remain compatible.
- Model-free regressions cover success, diagnostics, failure, finite-number
  policy, Unicode labels, ranking and argument validation. Full warning-strict
  pytest, dependency consistency and Python compilation pass.
- No serving weights, manifest, accuracy claims or dependency pins change.

## Next priorities

1. Stream local batch predictions as independently framed JSON Lines with one
   model load; first define per-image errors, cancellation and resource bounds.
2. Add an HTTP client with bounded timeouts and overload retries; first specify
   retry limits and distinguish lane saturation from readiness failures.
3. Complete the trusted Keras migration with actual artifact equivalence and
   readiness evidence. A pin-only update does not satisfy the existing gate.

This increment implements only the selected local CLI feature. Batch serving,
new public API surfaces and artifact promotion require separate design work.
