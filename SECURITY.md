# Keras security triage — 2026-10-07

The current GitHub Dependabot report contains 32 open alerts: 16 distinct Keras advisories repeated in `requirements.txt` and `requirements-api.txt`. Both files deliberately retain Keras 3.10.0 because the trusted serving artifact fails real deserialization under newer tested releases. This report records exposure; it does not dismiss or resolve the alerts.

## Boundaries verified in the code

- `/predict` accepts JPEG/PNG image uploads, not model archives. API callers cannot choose the model path.
- `api/utils.py` checks the selected artifact size, SHA-256 and class mapping against the local manifest before deserialization. The manifest and artifact must come from a trusted source; a malicious pair would pass those integrity checks.
- The legacy loader accepts `.keras` and `.h5` candidates when no manifest is present. Keras `safe_mode` cannot be treated as protection for every format or every listed advisory.
- Model loading during startup, training and operator workflows remains a security boundary. Run these workflows only with trusted artifacts and trusted local paths.

## Open advisories

| Advisory | Severity | First patched Keras | Summary |
|---|---|---|---|
| [GHSA-26c4-7vv6-867j](https://github.com/advisories/GHSA-26c4-7vv6-867j) | medium | 3.12.3 | Keras: HDF5 virtual datasets can disclose local files |
| [GHSA-36fq-jgmw-4r9c](https://github.com/advisories/GHSA-36fq-jgmw-4r9c) | high | 3.11.0 | Keras is vulnerable to Deserialization of Untrusted Data |
| [GHSA-36rr-ww3j-vrjv](https://github.com/advisories/GHSA-36rr-ww3j-vrjv) | high | 3.11.3 | The Keras `Model.load_model` method **silently** ignores `safe_mode=True` and allows arbitrary code execution when a `.h5`/`.hdf5` file is loaded. |
| [GHSA-3m4q-jmj6-r34q](https://github.com/advisories/GHSA-3m4q-jmj6-r34q) | high | 3.12.1 | Keras has a Local File Disclosure via HDF5 External Storage During Keras Weight Loading |
| [GHSA-4f3f-g24h-fr8m](https://github.com/advisories/GHSA-4f3f-g24h-fr8m) | high | 3.13.2 | Keras has an untrusted deserialization vulnerability |
| [GHSA-58hv-7753-xmfq](https://github.com/advisories/GHSA-58hv-7753-xmfq) | low | 3.12.3 | Keras: tar extraction permits symlink-based path traversal |
| [GHSA-5gwj-m78q-7pq3](https://github.com/advisories/GHSA-5gwj-m78q-7pq3) | high | 3.12.3 | Keras: Lambda deserialization can bypass safe mode and execute code |
| [GHSA-74m6-m3xx-3vmj](https://github.com/advisories/GHSA-74m6-m3xx-3vmj) | medium | 3.15.0 | Keras model loading is vulnerable to denial of service through HDF5 shape bombs |
| [GHSA-c9rc-mg46-23w3](https://github.com/advisories/GHSA-c9rc-mg46-23w3) | high | 3.11.0 | Keras vulnerable to CVE-2025-1550 bypass via reuse of internal functionality |
| [GHSA-gh82-f9x8-5frx](https://github.com/advisories/GHSA-gh82-f9x8-5frx) | medium | 3.12.3 | Keras: DiskIOStore permits path traversal through crafted layer names |
| [GHSA-hjqc-jx6g-rwp9](https://github.com/advisories/GHSA-hjqc-jx6g-rwp9) | high | 3.12.0 | Keras Directory Traversal Vulnerability |
| [GHSA-hqp4-2352-xf5r](https://github.com/advisories/GHSA-hqp4-2352-xf5r) | high | 3.14.0 | Keras archive extraction utilities allow path traversal and arbitrary file writes |
| [GHSA-m8wh-29wm-52mv](https://github.com/advisories/GHSA-m8wh-29wm-52mv) | medium | 3.12.3 | Keras: HDF5 links can disclose local file contents |
| [GHSA-mgx6-5cf9-rr43](https://github.com/advisories/GHSA-mgx6-5cf9-rr43) | high | 3.12.1 | Keras vulnerable to DoS via Malicious .keras Model (HDF5 Shape Bomb Causes Petabyte Allocation in KerasFileEditor) |
| [GHSA-mq84-hjqx-cwf2](https://github.com/advisories/GHSA-mq84-hjqx-cwf2) | medium | 3.12.0 | Keras is vulnerable to arbitrary local file loading and Server-Side Request Forgery |
| [GHSA-v2w2-w228-c444](https://github.com/advisories/GHSA-v2w2-w228-c444) | high | 3.12.3 | Keras: TorchModuleWrapper can deserialize unsafe PyTorch pickle data |

## Remediation gate

Do not upgrade the serving pin merely to clear alerts. Produce a candidate artifact using a patched Keras release, independently verify its provenance, and compare predictions against the current artifact on representative held-out images. Require a real load, API readiness/prediction smoke, class order and tensor-shape verification, then update artifact and manifest together. Keep the current artifact available for rollback. Re-export equivalence tooling is in `tools/check_reexport_equivalence.py`.

The public image endpoint does not directly expose artifact upload, but that alone does not establish that every advisory is unreachable. No model weights, manifest, dependency pins or Dependabot dismissal state changed in this triage.
