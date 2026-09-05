# RUNBOOK — Regenerating the Fischer 2023 Fig. 6 C0–C7 certification chain

**Repo:** `B:/AEinstein/Einstein/Documents/Soren/qpsim` · **Branch/HEAD verified now:** `fix/review-2026-08-03` @ `fca9389`, tree clean (only untracked `scratchpad/`).

I re-verified the characterisation against the live tree while writing this. **Five of its claims are wrong and are corrected inline below** — two of them would have produced a silently false certificate. They are marked ⚠️ **CORRECTION**.

---

## 0. Scope and the one-paragraph summary

Regenerate **C1 → C2 → C3 → C4 → C5 → C6 → C7**, in that order, atomically, in one sitting. **Do not regenerate C0.** C0's committed score is still valid (all 8 of its source bindings match live bytes), and its raw bundle was produced under **numpy 2.4.2** (confirmed from its manifest `runtime` block) — it is not byte-reproducible on the pinned numpy 2.5.1 environment, and nothing downstream requires rebuilding it. C0 is read-only input.

The chain is stale for two independent reasons: (a) source-closure digests moved under commits `c51a21c`, `da5aa74`, `6c78454`, `0e4b7ed`, `734ae50`, `90ccad3`; (b) the `c51a21c` factored-radicand / `arccosh`→`arcsinh` change genuinely moves kernel numbers from C3 downward. Re-binding digests is **structurally impossible** — every raw manifest embeds its own producer source closure that `build_cX_score` re-derives and bit-compares.

**Scope, measured at `c2470a0` (the parent of `b851add`):** every baseline source fingerprint in the tree was already invalid there — of the recorded source files, **3 are absent** from the tree (among them `qpsim/backends/t3_spatial_1d.py`), **38 files present in the tree are unrecorded**, and **32 recorded digests have drifted**. The recertification scope is therefore *every fingerprint*, not this chain alone. Two consequences for reading test results: (a) the slow-lane failures raise inside the baseline readers — each producer's `read_baseline` (`read_artifact` underneath the Fischer 2024 family) compares the recorded source fingerprint against the live one before any numeric assertion runs — so the reproduction gates are currently asserting provenance only, and a red there says nothing about the numbers; (b) with that provenance gate bypassed, the Fischer 2024 Fig. 8 producer (`validation/fischer_2024/fig8_paper.py`) output is digit-for-digit identical at `c2470a0` and `b851add`, and its worst column drift against the frozen CSV is **5.689e-12**, against the test's `rtol=1e-6`.

---

## 1. PRECONDITIONS

### 1.1 Interpreter (verified today)

```
C:/Users/Einstein2/Quasiparticle-Physics-Simulation/.venv/Scripts/python.exe
```
Verify — all four lines must match exactly:
```powershell
& C:/Users/Einstein2/Quasiparticle-Physics-Simulation/.venv/Scripts/python.exe -c `
  "import sys,platform,numpy,scipy;print(sys.version.split()[0]);print(numpy.__version__);print(scipy.__version__);print(platform.platform())"
```
Expected: `3.14.3` / `2.5.1` / `1.18.0` / `Windows-11-10.0.26100-SP0`.

**Why it must be exactly this:** the C3/C4/C5/C6/C7 raw manifests all already record `numpy_version: 2.5.1` (I read them). So this is not a re-pin — it is the environment those bundles were made in. The repo's own `.venv` (numpy 2.4.6, no pydantic) is wrong. `scipy` is a hard import of the C6 verifier (`from scipy.special import ellipe`).

### 1.2 Environment variables

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH       = "B:/AEinstein/Einstein/Documents/Soren/qpsim"
$env:BLIS_NUM_THREADS       = "1"
$env:MKL_DYNAMIC            = "FALSE"
$env:MKL_NUM_THREADS        = "1"
$env:NUMEXPR_NUM_THREADS    = "1"
$env:OMP_DYNAMIC            = "FALSE"
$env:OMP_NUM_THREADS        = "1"
$env:OPENBLAS_NUM_THREADS   = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
Set-Location B:/AEinstein/Einstein/Documents/Soren/qpsim
```

Verify:
```powershell
"PYTHONPATH","PYTHONIOENCODING","BLIS_NUM_THREADS","MKL_DYNAMIC","MKL_NUM_THREADS",
"NUMEXPR_NUM_THREADS","OMP_DYNAMIC","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS",
"VECLIB_MAXIMUM_THREADS" | ForEach-Object { "{0} = {1}" -f $_, [Environment]::GetEnvironmentVariable($_) }
```

- Only **three** (`MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` = the literal string `"1"`) are *hard-enforced* — `_runtime_record()` in the C5/C6/C7 bundle producers raises `CxBundleError` otherwise. The other five are harmless and should be set anyway for reduction-order determinism.
- **`PYTHONPATH` is safety-critical, not cosmetic.** With the pinned venv and no `PYTHONPATH`, `import qpsim` / `import validation` resolve into `C:/Users/Einstein2/Quasiparticle-Physics-Simulation/` — a *different tree*. `REPOSITORY_ROOT` derives from `__file__`, so an unset `PYTHONPATH` silently certifies the wrong repository.

### 1.3 Author archive — NOT required

`QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE` is unset and **must stay irrelevant**. Only C0 (via `fig6_author_adapter.py`) and the separate `author-output-score.json` producer consume the archive; neither is in this runbook's scope. Verify no stage reaches for it:
```powershell
Select-String -Path validation/fischer_2023/fig6_author_c[1-7]_*.py -Pattern "getenv|environ|AUTHOR_ARCHIVE" | Select-Object -First 5
```
Expect only the three thread-var reads in the C5/C6/C7 bundle producers. Zero archive hits.

### 1.4 External raw bundles (gitignored, irreplaceable)

All seven exist. **Back up `tmp/author-runs/` before touching anything** — if lost, the chain cannot be rebuilt at all, and C0 could only be recovered from the archive.

| Directory (under `tmp/author-runs/`) | files | size | manifest sha256 (verified today) |
|---|---|---|---|
| `fig6-T020-sweep049-C0-author-equivalent-v1` | 25 | 0.5 MB | `ebe32e48…56ac` |
| `fig6-T020-sweep049-C2-parameters-v1` | 125 | 1.7 MB | `ae6e0693…607d` |
| `fig6-T020-sweep049-C3-grid-regen-v1` | 106 | 82.4 MB | `80b5f82a…13b8` |
| `fig6-T020-sweep049-C4-photon-regen-v1` | 31 | 0.4 MB | `f81b392f…101e` |
| `fig6-T020-sweep049-C5-qp-phonon-regen-v1` | 59 | 146.9 MB | `0999dda1…0121` |
| `fig6-T020-sweep049-C6-phonon-balance-v1` | 87 | 86.3 MB | `1533688e…be08` |
| `fig6-T020-sweep049-C7-solver-v1` | 45 | 0.8 MB | `d6b8733c…f0c3f` |

Free disk: budget **~350 MB** (new C3 + C5 + C6 alongside the archived old copies).

### 1.5 ⚠️ CORRECTION — bundle directory names are load-bearing, and the characterisation's C5 advice is a trap

The C5 characterisation recommends writing the new C5 bundle to `fig6-T020-sweep049-C5-qp-phonon-v1` because "that name does not exist yet and is the FIRST candidate both C5 test files probe." **Do not do this.** I grepped every hardcoded literal:

- `test_fig6_author_c5_bundle.py:78` and `c5_evidence.py:72` probe `C5-qp-phonon-v1` **first**, falling back to `-regen-v1`.
- `test_fig6_author_c6_bundle.py:56`, `c6_evidence.py:37`, `c7_bundle.py:215`, `c7_evidence.py:20` hardcode `C5-qp-phonon-regen-v1` with **no fallback**.

Writing to `-qp-phonon-v1` therefore binds the C5 suite to the new bundle while C6 and C7 keep scoring the **old, stale** one — both lanes green, one certifying dead evidence. That is precisely the inert-switch pattern.

**Rule: every regenerated bundle must land at the exact directory name that exists today.**

| Stage | Required output directory name |
|---|---|
| C2 | `fig6-T020-sweep049-C2-parameters-v1` |
| C3 | `fig6-T020-sweep049-C3-grid-regen-v1` |
| C4 | `fig6-T020-sweep049-C4-photon-regen-v1` |
| C5 | `fig6-T020-sweep049-C5-qp-phonon-regen-v1` |
| C6 | `fig6-T020-sweep049-C6-phonon-balance-v1` |
| C7 | `fig6-T020-sweep049-C7-solver-v1` |

**Corollary for archiving the old copies:** never move an old bundle to a name the test probes. The reserved names you must not create are `C3-grid-v1`, `C4-photon-v1`, `C5-qp-phonon-v1`, `C5-qp-phonon-producer-dev-v5`, `C5-qp-phonon-producer-dev-v6`. Archive **outside** `tmp/author-runs/`:

```powershell
$STAMP = Get-Date -Format "yyyyMMdd-HHmm"
New-Item -ItemType Directory -Force "B:/AEinstein/Einstein/Documents/Soren/qpsim/tmp/preregen-$STAMP" | Out-Null
```

### 1.6 Pytest is opt-in — a plain run proves nothing

`conftest.py:13-33` auto-marks every `tests/validation/test_fig6_author_*` node `paper_validation`; `pyproject.toml:62` sets `addopts = -m "not slow and not paper_validation"`. Plain `pytest` reports *deselected*. **Always use `-m paper_validation`.** Confirm the lane is live before starting:
```powershell
& $PY -m pytest tests/validation/test_fig6_author_c1_score.py -m paper_validation -q
```
Expect **4 failed, 1 skipped** (the four `load_c1_score()` failures are the staleness you are about to fix; the skip is the regen test). If you see `deselected`, your marker flag is wrong.

### 1.7 Set the two test env vars

```powershell
$env:QPSIM_FISCHER2023_FIG6_C0_BUNDLE = "B:/AEinstein/Einstein/Documents/Soren/qpsim/tmp/author-runs/fig6-T020-sweep049-C0-author-equivalent-v1"
$env:QPSIM_FISCHER2023_FIG6_A1_BUNDLE = "C:/tmp/qpsim-round7-fixes/tmp/author-runs/fig6-T020-sweep049-A1-exact-final-v6"
```
Without these, the C0 byte-exact regen test and the C1 regen test **skip**, and their green is meaningless.

### 1.8 Do not grab the wrong driver

`scripts/regenerate_fischer_fig6_parallel.py` exists but is the **300-point curve** producer (A-ladder, `fig6_solve` sweep across three bath temperatures). It has nothing to do with C0–C7. **There is no driver script for this chain**; the module CLIs below are the only build path.

### 1.9 Pre-flight gate

Run all of these and require the stated result before proceeding:

```powershell
& $PY -c "import sys;sys.path.insert(0,'B:/AEinstein/Einstein/Documents/Soren/qpsim');from validation.fischer_2023.fig6_author_c0_summary import load_c0_summary;s=load_c0_summary();print('C0 OK', s['acceptance']['accepted'])"
```
Must print `C0 OK True`. **If C0 fails to load, STOP** — the whole plan assumes C0 is the fixed point (see §5).

```powershell
git status --short   # must show only ?? scratchpad/
```

---

## 2. ORDERED STEPS

Shorthand used below:
```powershell
$PY   = "C:/Users/Einstein2/Quasiparticle-Physics-Simulation/.venv/Scripts/python.exe"
$RUNS = "tmp/author-runs"
$PD   = "validation/paper_data/fischer_2023/fig6"
$ARCH = "tmp/preregen-$STAMP"
function Sha($p) { (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }
```

Every writer is **exclusive-create** (`open('xb')` for JSON, `mkdir(exist_ok=False)` for directories). There is no `--force`. Move the old artifact aside first, every time.

---

### STEP 1 — C1 (observable substitution) · pure digest re-bind

**Nothing numerical moves. I verified this by building the score in memory.**

```powershell
Move-Item "$PD/c1-observable-score.json" "$ARCH/c1-observable-score.json"

& $PY -m validation.fischer_2023.fig6_author_c1_score `
  --c0-bundle "$RUNS/fig6-T020-sweep049-C0-author-equivalent-v1"
```
(`--c0-summary` and `--output` default correctly; only `--c0-bundle` is required.)

**Check before proceeding:**
```powershell
Sha "$PD/c1-observable-score.json"
```
Must be exactly `72a10d95d8c38b8cf36a31b7519bce801374c0eb2dde52c314c6f014a7d346d7`
(old was `e695d2d976158f19a3dd8ac9106313d55d2d38f43a4ee5599642c08bb17773fc`).

Then diff — **it must be exactly one changed line**, the `gap_suppression.py` digest `90d591b9…244dc` → `95769ac7…6cbcc`:
```powershell
& $PY -c "import json;a=json.load(open(r'$ARCH/c1-observable-score.json'));b=json.load(open(r'$PD/c1-observable-score.json'));print([k for k in a if a[k]!=b[k]])"
```
Must print `['sources']`.

**Verified expectations (I ran `build_c1_score` in memory):** `driven_gap_eV = 0.00017999646581118777`, `thermal_gap_eV = 0.00017999597972317586`, `figure6_ordinate = 0.12090908988993258`, `acceptance.accepted = True`, all bit-identical to the committed file. If any of those move, **abort** (§5).

---

### STEP 2 — C2 (parameter plumbing)

⚠️ **CORRECTION — the C2 raw bundle MUST be rebuilt, not reused.** The C2 characterisation treats the retained `C2-parameters-v1` bundle as reusable. It is not, once C1 moves. I read the manifest: it records

```
metadata.parent_bindings.c1_score_sha256 = e695d2d9…73fc   (the OLD C1 score)
```

written by `fig6_author_c2_bundle.py:348`, and `fig6_author_c2_score.py:1306-1310` compares that binding against the live C1 file. Reusing the old bundle raises `C2ScoreError: Checked C2 C1-score binding is stale.`

Good news, also read from the manifest: **the C2 bundle has no `runtime` block at all** (unlike every other stage). Its rebuild is therefore host-independent, and the *only* thing that should change is the parent binding.

```powershell
Move-Item "$RUNS/fig6-T020-sweep049-C2-parameters-v1" "$ARCH/C2-parameters-v1"
Move-Item "$PD/c2-parameter-score.json"               "$ARCH/c2-parameter-score.json"

& $PY -m validation.fischer_2023.fig6_author_c2_bundle `
  --c0-bundle "$RUNS/fig6-T020-sweep049-C0-author-equivalent-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C2-parameters-v1"
```

**Check — this is the single most informative gate in the whole run.** Every `.npy` must be byte-identical to the archived copy; only `manifest.json` may differ:
```powershell
$new = "$RUNS/fig6-T020-sweep049-C2-parameters-v1"; $old = "$ARCH/C2-parameters-v1"
Get-ChildItem $new -Filter *.npy | ForEach-Object {
  if ((Sha $_.FullName) -ne (Sha "$old/$($_.Name)")) { "ARRAY MOVED: $($_.Name)" }
}
```
**Any output here means abort** (§5) — C2's source closure is clean, so no array has any business moving.

Then the score:
```powershell
& $PY -m validation.fischer_2023.fig6_author_c2_score `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --c0-bundle "$RUNS/fig6-T020-sweep049-C0-author-equivalent-v1"
```

**Then hand-edit the receipt.** ⚠️ There is **no** `build_c2_receipt` / `write_c2_receipt` anywhere in the repo — I confirmed receipt builders exist only for C3, C4, C5, C6, C7. `c2-raw-manifest-receipt.json` is hand-maintained. Change exactly two hex strings, preserving sorted keys, 2-space indent, trailing newline, and the `qualification` string verbatim:

```json
{
  "checked_score": { "file_sha256": "<sha256 of the new c2-parameter-score.json>", "schema": "…c2-parameter-score.v1" },
  "qualification": "…leave untouched…",
  "raw_bundle":    { "manifest_sha256": "<sha256 of the new C2 manifest.json>", "schema": "…c2-frozen-bundle.v1" },
  "schema": "…c2-raw-manifest-receipt.v1"
}
```
(Old values, for the diff: score `47e3bbba…9cdb`, raw manifest `ae6e0693…607d`.)

**Check:** `& $PY -c "...load_c2_score(...)"` must load clean, and `parameter_axis.qpsim_fixed_nbar_t_star_over_delta` must still be exactly `0.3399503360830364`, `len(steps) == 6`. Diff the score against the archived copy: the only changed leaves should be `parent_bindings.c1_score_sha256` and `raw_bundle.manifest_sha256`.

---

### STEP 3 — C3 (grid embedding) · **first stage where numbers really move**

```powershell
Move-Item "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" "$ARCH/C3-grid-regen-v1"
Move-Item "$PD/c3-grid-score.json"                    "$ARCH/c3-grid-score.json"
Move-Item "$PD/c3-raw-manifest-receipt.json"          "$ARCH/c3-raw-manifest-receipt.json"

& $PY -m validation.fischer_2023.fig6_author_c3_bundle `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1"

& $PY -m validation.fischer_2023.fig6_author_c3_score score `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output "$PD/c3-grid-score.json"

& $PY -m validation.fischer_2023.fig6_author_c3_score receipt `
  --score "$PD/c3-grid-score.json" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output "$PD/c3-raw-manifest-receipt.json"
```

The `receipt` step independently re-runs the whole score and refuses to anchor bytes that do not reproduce — **it is the real self-check, not a formality.**

**Check:** `acceptance.raw_array_max_absolute_error == 0.0` (the `_derive_grid` mirror is bit-exact against `spectral.py`) and `observable_integral_max_absolute_error <= 4e-18`. Then gate:
```powershell
& $PY -m pytest tests/validation/test_fig6_author_c3_bundle.py tests/validation/test_fig6_author_c3_evidence.py -m paper_validation -q
```
C3 has **no** hard-coded `_EXPECTED_*` pin block — I checked; nothing to hand-edit here.

---

### STEP 4 — C4 (photon operator) · requires a verifier source edit

Build the bundle first:
```powershell
Move-Item "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" "$ARCH/C4-photon-regen-v1"
Move-Item "$PD/c4-photon-score.json"                    "$ARCH/c4-photon-score.json"
Move-Item "$PD/c4-raw-manifest-receipt.json"            "$ARCH/c4-raw-manifest-receipt.json"

& $PY -m validation.fischer_2023.fig6_author_c4_bundle `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1"
```

**Now the pin.** `_EXPECTED_OPERATOR_COMPARISON` (`fig6_author_c4_score.py:96-112`) is nine hard-coded floats compared bit-exactly inside `_operator_comparison` (line 1136) — **during the build**, and the exception does **not** report the measured values. So `build_c4_score` cannot tell you the new numbers. Compute them yourself with the identical expressions from lines 1121-1131:

```python
# scratch script, reads only
from validation.fischer_2023.fig6_author_c4_score import load_c4_raw_bundle
import numpy as np
_, arrays = load_c4_raw_bundle("tmp/author-runs/fig6-T020-sweep049-C4-photon-regen-v1")
for field in ("gain", "loss", "net"):
    parent    = np.asarray(arrays[f"parent_qp_photon_{field}_s_inv"])
    candidate = np.asarray(arrays[f"qpsim_{field}_s_inv"])
    absolute  = np.abs(candidate - parent)
    denom     = float(np.sum(np.abs(parent))) + float(np.sum(np.abs(candidate)))
    print(field,
          repr(float(np.sum(absolute))),
          repr(float(np.max(absolute, initial=0.0))),
          repr(float(np.sum(absolute)) / denom))
```

Record the before/after table explicitly in the commit message. Current pinned values:

| field | `l1_absolute_s_inv` | `linf_absolute_s_inv` | `symmetric_relative_l1` |
|---|---|---|---|
| gain | `2.9390985920687567e-13` | `2.842170943040401e-14` | `2.987877778643742e-17` |
| loss | `6.358292238341556e-13` | `2.2737367544323206e-13` | `6.564511414882551e-17` |
| net  | `6.529623265709012e-13` | `2.942091015256665e-13` | `2.017243083178416e-15` |

Edit those nine literals to the measured values (Python `repr` of a float round-trips exactly; `float.fromhex` is safer still). **You are editing the independent verifier to agree with the producer — this is the single most review-sensitive edit in the run. Do not let it pass silently.** Sanity floor: these are roundoff metrics; they should stay in the `1e-17…1e-13` band. Anything larger means a semantic change, not reordering (§5).

Editing `fig6_author_c4_score.py` is safe *at this point* because it is deliberately excluded from the C4 bundle's `_RAW_SOURCE_RELATIVES` (verified). But it **is** inside C5's closure, so the edit must land before Step 5.

Then:
```powershell
& $PY -m validation.fischer_2023.fig6_author_c4_score score `
  --c4-bundle "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output "$PD/c4-photon-score.json"

& $PY -m validation.fischer_2023.fig6_author_c4_score receipt `
  --score "$PD/c4-photon-score.json" `
  --c4-bundle "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output "$PD/c4-raw-manifest-receipt.json"
```

**Check:** the semantic-delta support must be **exactly `{1619, 1639}`**. The bundle asserts this. If a regeneration makes it anything else, something structural moved — do not paper over it. Gate with `-m paper_validation` on both C4 test files.

---

### STEP 5 — C5 (QP–phonon operator) · five pin blocks, one bootstrap pass

⚠️ **CORRECTION (applies to C5, C6 and C7):** the C7 characterisation says the canonical pins are "checked in `load_c7_score`, **NOT** in `write_c7_score`." That is wrong, and the error is load-bearing. I traced it: `write_cN_score` → `canonical_score_bytes` → `_validate_score_structure` → `_validate_canonical_pins` (`c5:2461`, `c6:2874`, `c7:1986`). **The pins block the write.** You cannot produce the score first and re-pin afterwards; the bootstrap is mandatory.

⚠️ **CORRECTION:** the C6 characterisation lists no pin blocks. **C6 has them too** — `fig6_author_c6_score.py:177` `_EXPECTED_RAW_MANIFEST_SHA256`, `:180` `_EXPECTED_EVIDENCE_DIGEST`, `:183` `_EXPECTED_CANONICAL_METRICS` (7 hex floats), `:204` `_EXPECTED_RESIDUAL_NPY_SHA256` (7 digests). Budget the same bootstrap for C6 as for C5 and C7.

**One bootstrap pass suffices**, and here is why: the pins are *module constants*, not score fields, so editing them does not change the score dict; and `_evidence_digest` covers every score key **except `sources`** (see the comment at `c5_score.py:2048-2052`), so editing the verifier's own source — which only moves `sources` — does not move the digest. Compute all five pins from one in-memory build, edit, then write.

```powershell
Move-Item "$RUNS/fig6-T020-sweep049-C5-qp-phonon-regen-v1" "$ARCH/C5-qp-phonon-regen-v1"
Move-Item "$PD/c5-qp-phonon-score.json"                    "$ARCH/c5-qp-phonon-score.json"
Move-Item "$PD/c5-raw-manifest-receipt.json"               "$ARCH/c5-raw-manifest-receipt.json"

& $PY -m validation.fischer_2023.fig6_author_c5_bundle `
  --c4-bundle "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C5-qp-phonon-regen-v1"
```

Bootstrap the pins (writes nothing):
```python
import validation.fischer_2023.fig6_author_c5_score as m
score = m.build_c5_score(C5DIR, c4_bundle_dir=C4DIR, c3_bundle_dir=C3DIR, c2_bundle_dir=C2DIR)
print("RAW_MANIFEST :", score["raw_bundle"]["manifest_sha256"])
print("EVIDENCE     :", m._evidence_digest(score))
import json; print(json.dumps(score["channel_comparison"], indent=2))   # -> _EXPECTED_CANONICAL_METRICS
print(json.dumps(score["conservation"], indent=2))
# _EXPECTED_RESIDUAL_NPY_SHA256 and _EXPECTED_RAW_BOOKKEEPING_METRICS come from the
# new bundle's manifest.json array_descriptors / bookkeeping block.
```
Write the metrics back as `float.fromhex(...)` literals, matching the existing style exactly. **Compute `_EXPECTED_EVIDENCE_DIGEST` last and re-verify it after the other four edits** — cheap insurance against the self-reference.

Then `score`, then `receipt`, using the CLI forms in the characterisation (`--c5-bundle --c4-bundle --c3-bundle --c2-bundle`, `--output` explicit).

**Check:** the producer's own gates must have fired clean — scattering weighted-QP-number relative drift `<= 1e-12`, and scattering turnover `> 0` ("the public C5 scattering comparison is vacuous" means abort). Expect `kernel_formula.maximum_rounding_bound_fraction` to return to `0.0` for both channels now that C3 and C5 are mutually fresh.

---

### STEP 6 — C6 (phonon balance)

Same shape as C5. Archive `C6-phonon-balance-v1` + both JSONs, rebuild the bundle into the **same** directory name, bootstrap the four C6 pin blocks (§5 correction above), write score, write receipt.

```powershell
& $PY -m validation.fischer_2023.fig6_author_c6_bundle `
  --c5-bundle "$RUNS/fig6-T020-sweep049-C5-qp-phonon-regen-v1" `
  --c4-bundle "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C6-phonon-balance-v1"
```
then `... c6_score score ...` and `... c6_score receipt --score ...`, each repeating all four `--cN-bundle` flags.

**Check:** exactly **86 `.npy` + `manifest.json`**, nothing else; detailed-balance relative limits `<= 1e-12`; the `pair_control` channel present and distinct from `pair` (that control is what separates the Kaplan S₊ correction from the endpoint policy — if it collapses, the stage has lost its point).

Note `build_c6_bundle` internally re-runs `build_c5_score` and demands byte-identity with the C5 score you just wrote. If that fails, your C5 step is not actually settled — go back, do not force forward.

---

### STEP 7 — C7 (nonlinear solver) · the expensive one

```powershell
Move-Item "$RUNS/fig6-T020-sweep049-C7-solver-v1" "$ARCH/C7-solver-v1"
Move-Item "$PD/c7-nonlinear-solver-score.json"    "$ARCH/c7-nonlinear-solver-score.json"
Move-Item "$PD/c7-raw-manifest-receipt.json"      "$ARCH/c7-raw-manifest-receipt.json"

& $PY -m validation.fischer_2023.fig6_author_c7_bundle `
  --c6-bundle "$RUNS/fig6-T020-sweep049-C6-phonon-balance-v1" `
  --c5-bundle "$RUNS/fig6-T020-sweep049-C5-qp-phonon-regen-v1" `
  --c4-bundle "$RUNS/fig6-T020-sweep049-C4-photon-regen-v1" `
  --c3-bundle "$RUNS/fig6-T020-sweep049-C3-grid-regen-v1" `
  --c2-bundle "$RUNS/fig6-T020-sweep049-C2-parameters-v1" `
  --c0-bundle "$RUNS/fig6-T020-sweep049-C0-author-equivalent-v1" `
  --output-dir "$RUNS/fig6-T020-sweep049-C7-solver-v1"
```
Then bootstrap C7's five pins (`:178` raw manifest, `:181` evidence digest, `:184` six hex floats, `:196` `_EXPECTED_CANONICAL_INTS` with `accepted_max_iter=4`, `:199` `_EXPECTED_ROOT_NPY_SHA256`), then `score`, then `receipt` (same six parent flags each time).

**Regression comparison — record these against the new run:**

| quantity | committed value |
|---|---|
| `accepted_max_iter` | `4` |
| `root_ordinate_authors` | `0x1.29d3f1263dc67p-3` ≈ `0.14542377851441587` |
| `root_ordinate_centers` | `0x1.3e2aefabad5f4p-4` ≈ `0.07767766591390296` |
| `AUTHOR_CONTROL_ORDINATE` | `0.12090908988993258` |
| `root_qp_balance_l1_ratio` | `≈0x1.4e1dc1819f830p-45` |
| `root_ph_balance_l1_ratio` | `≈0x1.951455cf0d1bap-38` |

**Fix the known prose defect while you are here** (`docs/CURRENT-STATUS.md:89-98`): `seed_binding.source` in the bundle metadata calls the A1 seed a *"(continuation)"* state. It is not — the author program constructs each sweep point independently from a thermal initializer. The docstring of `fig6_author_c7_bundle.py` ("captured author continuation seed") and the test name `test_c7_seed_is_the_authenticated_a1_continuation_state` carry the same error. This was deliberately deferred "to the next C7 regeneration." It moves `_EXPECTED_EVIDENCE_DIGEST`, which you are re-pinning anyway, so it costs nothing now and costs a full C7 regen later.

---

### STEP 8 — Rebind `reproduction-ladder.json`

There is **no generator** for this file; it is hand-maintained. I audited it against live bytes just now: **26 rows match, 17 are stale.** Every stale row is C3-or-later plus one D0 row:

```
D0 ev[1] validation/fischer_2023/fig6_cleanroom_parity.py        (source_canonical)
C1 ev[0] qpsim/observables/gap_suppression.py                    (source_canonical)
C3 ev[1] validation/fischer_2023/fig6_author_c3_score.py
C3 ev[2] c3-raw-manifest-receipt.json
C3 ev[3] c3-grid-score.json
C3 ev[4] staged-resolve-pilot.json
C4 ev[2] c4-raw-manifest-receipt.json
C4 ev[3] c4-photon-score.json
C5 ev[1] validation/fischer_2023/fig6_author_c5_score.py
C5 ev[2] c5-raw-manifest-receipt.json
C5 ev[3] c5-qp-phonon-score.json
C6 ev[1] validation/fischer_2023/fig6_author_c6_score.py
C6 ev[2] c6-raw-manifest-receipt.json
C6 ev[3] c6-phonon-balance-score.json
C7 ev[1] validation/fischer_2023/fig6_author_c7_score.py
C7 ev[2] c7-raw-manifest-receipt.json
C7 ev[3] c7-nonlinear-solver-score.json
```

Note C0's and C2's rows are currently **clean**; C1 has exactly one stale row (which is *why* C1 must regen). After the run, all C1–C7 rows change. Rebind mechanically — `hash_kind: "source_canonical"` rows use `validation.source_provenance.source_sha256`, `hash_kind: "file_bytes"` rows use plain `sha256` of the file.

**Do not run a global rebind.** Rebind only the stages you just regenerated. A blanket pass would silently bless `D0 ev[1]` and `C3 ev[4] staged-resolve-pilot.json`, which are stale for reasons this runbook did not establish (§6).

Final gate:
```powershell
& $PY -m pytest tests/validation/ -m paper_validation -q
```

---

## 3. EXPECTED COST

**Nothing in the numerical core is hours.** I measured the dominant components directly on the pinned interpreter with all eight thread vars set:

| operation | measured |
|---|---|
| `import` + `fig6_solve._build_grid_and_spectral()` | 1.8 s + 0.04 s |
| `build_phonon_frequency_map` (→ 3600 ω bins) | 0.09 s |
| `build_scattering_kernel_base` (1640×1640) | 0.09 s |
| `build_recombination_kernel_base` (1640×1640) | 0.11 s |
| `phonon_collision_rates` (full) | 1.01 s |
| `math.fsum` over 2.69M floats (verifier reduction) | 0.04 s |
| 1640 row-wise `math.fsum` | 0.04 s |
| writing 6 × 21.5 MB `.npy` into repo `tmp/` | 0.86 s |
| `build_c1_score` end-to-end, in memory | **0.04 s** |

Estimates, flagged by confidence:

| stage | estimate | confidence |
|---|---|---|
| C1 | **< 1 s** | measured |
| C2 | seconds (bundle ≈ 6 frozen operator rebuilds, no Newton; score re-runs `build_c1_score`) | high |
| C3 | **minutes** — 83 MB written, 1640×1640 slicing, verifier re-derives the whole grid | medium |
| C4 | seconds to a minute — 0.4 MB, one 1640-iteration Python loop | medium |
| C5 | **minutes** — 147 MB written, six 21.5 MB matrices | medium |
| C6 | **minutes** — 86 MB written, plus `scipy.special.ellipe` | medium |
| C7 | **the only hours-risk. Unknown.** | low |

**Why C7 is the risk, concretely:** `build_c7_bundle` re-runs the *entire* `build_c6_score` **twice** (once before the solve, once after, to prove parent evidence did not change mid-run), and each of those internally replays C5→C4→C3→C2. Then the iteration probe calls `coupled_newton_solve` up to 10 times with increasing caps plus once more at `max_iter=10` demanding bit-identical output. Then `build_c7_receipt` re-runs the whole `build_c7_score` again. That is **≈4 complete chain replays + ~11 Newton solves**.

**Determine it rather than guess:** time Step 3 (C3) with `Measure-Command`. Call that `T_C3`. A single chain replay is roughly `T_C2+T_C3+T_C4+T_C5+T_C6`; C7 is ≈4× that plus the solves. If `T_C3` comes in under ~3 minutes, C7 lands in tens of minutes and the whole run is a half-day. If `T_C3` exceeds ~15 minutes, C7 is plausibly multi-hour and should be started with a wall-clock budget and `run_in_background`.

**Do not run any of this concurrently with an editor autosave, formatter, linter, or a second build.** Every producer re-hashes its source closure at import, before the work, and again immediately before renaming the temp directory into place. A stray save aborts the build with `source closure changed during execution` — after you have paid the compute.

---

## 4. WHAT MOVES (so the diff can be reviewed, not rubber-stamped)

### Moves — expected and benign

- **C1:** one line. `sources["qpsim/observables/gap_suppression.py"]` `90d591b9…244dc` → `95769ac7…6cbcc`. File sha `e695d2d9…73fc` → `72a10d95…d346d7`. **Verified by me: no number moves.** The two commits that touched the observable since the last C1 regen (`231d949`, `90ccad3`) provably did not perturb the `samples="authors"` path.
- **C2:** `parent_bindings.c1_score_sha256` and `raw_bundle.manifest_sha256`. Every `.npy` must be byte-identical (Step 2 gate). No physics.
- **Every stage:** its `sources` map, its raw manifest sha, its receipt's two digests, and the corresponding ladder rows.

### Moves — real, and the reason regeneration is required

Attributable to **`c51a21c` (batch R2)** — the `arccosh` → factored `arcsinh` anomalous-weight reformulation in `qpsim/physics/spectral.py:435`, mirrored at `fig6_author_c3_score.py:626-632`:

| quantity | measured movement |
|---|---|
| `anomalous_weight` | 509 of 1640 cells move, max rel **1.412511e-12** |
| `K_plus` / `K_minus` | max abs **1.587619e-13**; 1.39M of 2.69M entries (51.6%) move |
| C5 scattering kernel | max abs **6.878179e-16** = **138.8×** the 32-eps gate |
| C5 pair kernel | max abs **1.081600e-15** = **20.1×** the 32-eps gate |

⚠️ **CORRECTION — the numbers in the user's memory note are wrong and must not be carried forward.** "9.9e-13" and "69×" appear **nowhere** in the repo (`git log --all -S"9.9e-13"` is empty) and are mutually inconsistent (9.9e-13 / 32ε = 139, not 69). They also mis-attribute the cause: `da5aa74` ("stop cancelling away the gap edge in σ₂") touches `qpsim/observables/ac_conductivity.py` and a **comment-only** hunk in `spectral.py`; it is not in the Fig. 6 numerical path and contributes only a digest advance. Use the measured table above, and say **`c51a21c` factored-radicand**, not "σ₂".

### Moves — needs a human eye

- **C4's nine `_EXPECTED_OPERATOR_COMPARISON` constants** will very likely shift (they are roundoff metrics downstream of `bcs_quadrature.py` and `spectral.py`, both changed). Record before/after explicitly; this is an edit to the *independent verifier*.
- **C5/C6/C7 `_EXPECTED_CANONICAL_METRICS` and `_EXPECTED_RESIDUAL_NPY_SHA256` will move**, not just the manifest sha and evidence digest. The previous two regenerations (`c269af2`, `a9ead85`) were 2-line re-pins because only `metadata.sources` had moved. **This one is different.** Anyone assuming a 2-line re-pin produces a false certificate.
- **C7 `accepted_max_iter = 4`.** If the regenerated root needs a different iteration count, that is a **semantic change to the stage's headline claim**, not a digest refresh — escalate (§5).

### Does not move

No published physics number is at stake in C0–C7. This chain is one point (T_B = 0.20 K, sweep index 49); every stage's own `limitations` block disclaims the 300-point curve, the plotted ordinate, and paper parity. The 300-point curve and `author-output-score.json` are a **separate** artifact family with a **separate** producer (`scripts/regenerate_fischer_fig6_parallel.py` / `fig6_author_output_parity.py`) and are out of scope here.

---

## 5. ABORT CONDITIONS

Stop and escalate — do not work around, do not relax a limit, do not hand-edit a digest to make a gate pass.

1. **Any `.npy` in the rebuilt C2 bundle differs from the archived copy** (Step 2 gate). C2's ten pinned sources all match disk, so no array has a mechanism to move. If one does, something changed that nobody has accounted for.
2. **`load_c0_summary()` fails at pre-flight.** The entire plan rests on C0 being the fixed point. C0's raw bundle was made under numpy 2.4.2 and cannot be reproduced here — if C0 is broken, this is a much larger problem and needs the archive plus a human decision.
3. **A `C3ScoreError: raw_array_max_absolute_error != 0.0`** — the verifier's `_derive_grid` mirror has fallen out of lockstep with `spectral.py`. That mirror is an explicit contract (`c51a21c`: "Sequence identity is part of the contract, not style"). Fixing it means re-mirroring the arithmetic operation-for-operation, which is a physics-review task, not a runbook task.
4. **C4's semantic-delta support is anything other than `{1619, 1639}`.** Structural change in the photon endpoint policy.
5. **Any new `_EXPECTED_*` value lands outside its plausible band** — C4's roundoff metrics outside `1e-17…1e-13`; C5's detailed-balance or net-parity relatives above `1e-12`; C6's escape budget above 16 eps. These gates are stated bounds on *roundoff*. A violation means the substitution changed the physics, and the correct response is to find out why.
6. **C7's `accepted_max_iter` is not 4.** Semantic change to the published claim.
7. **`"the public C5 scattering comparison is vacuous"`** or any conservation gate firing. A vacuous comparison certifies nothing.
8. **Any producer aborts with `source closure changed during execution`.** Something wrote to the tree mid-build. Do not restart blindly — establish what wrote, and re-run from the start of that stage.
9. **A test suite reports `deselected`, or `skipped` where you expected it to run.** Green-by-skip is the dominant failure mode in this area (`_require_external_bundles()` skips rather than fails). If the C2/C3/C4/C5/C6/C7 evidence tests report skips, the bundle directory names are wrong — see §1.5.
10. **You find yourself creating a bundle directory under a name in the reserved list** (`C3-grid-v1`, `C4-photon-v1`, `C5-qp-phonon-v1`, `C5-qp-phonon-producer-dev-v5/v6`). Stop; you are about to split the test suite across two bundles.
11. **You are tempted to hand-edit a `sha256` field inside a score JSON to make a load pass.** Never. The recorded identity would decouple from the numbers it certifies. The standing project policy is *"regen, do not rebind"* (`docs/CURRENT-STATUS.md:47`).

---

## 6. OPEN UNKNOWNS

Honest gaps. None of these are guesses dressed up as facts.

1. **Actual wall-clock for C3, C5, C6 and especially C7.** I measured components (§3), not stages — I did not execute any producer, since the parent chain is stale and any real timing run *is* the regeneration. **Determine by:** timing C3 with `Measure-Command` and extrapolating as described in §3.

2. **Whether C4/C5/C6/C7's `_EXPECTED_*` values actually move, and by how much.** I established that the inputs feeding them changed and that the drift at the C5 kernel level is 138.8× / 20.1× the 32-eps gate — but the *pinned metrics* are second-order functionals of those kernels. **Determine by:** the bootstrap pass in Steps 4–7. There is no way to know without running.

3. **Whether the regenerated C7 root still converges at `max_iter = 4`.** Flagged as a §5 abort precisely because it cannot be predicted.

4. **Why the ladder was left pinned to pre-`a9ead85` C5/C6/C7 values across two later commits**, including `788569e` which touched the ladder. There is a track record here — `788569e`'s own message is *"correct the A0 ladder binding, which never matched any commit."* I did not establish whether this is process drift or something deliberate.

5. **`D0 ev[1]` (`fig6_cleanroom_parity.py`) and `C3 ev[4]` (`staged-resolve-pilot.json`).** Both are stale at HEAD and outside this chain. `test_reproduction_ladder.py` currently dies at **D0** — *before* it ever reaches the C stages — so the ladder test will stay red after a perfect C0–C7 run unless D0 is also addressed. I did not establish whether `cleanroom-analytic-score.json` internally embeds the drifted adapter digest (which would make a plain ladder rebind illegitimate and force a D0 regeneration). **Determine by:** reading that score's `producer.sources` / `extraction.script_sha256` block and comparing to live `source_sha256`.

6. **The separate `author-output-score.json` debt.** Independent of this chain, two of its four producer digests (`fig6_author_output_parity.py`, `paper_parity.py`) are stale, and regenerating it **does** require `QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE=C:/tmp/PhysApplPaper_Figure_6.zip`. It is a numerical no-op except ~1e-12 numpy-2.5.1 wobble. Out of scope here, but it blocks a meaningfully green `-m paper_validation` lane, and the P05 extraction fix (a real behaviour change) must **not** be silently bundled into it — that patch as filed does not even run (`ACCEPTED_Y_VALUE_WINDOW` floor `0.08` rejects a genuine trace at y=0.0796; ~0.075 is the correct floor).

7. **`_atomic_exclusive_write` uses `os.link()` (hard link) into the target's parent directory**, and the bundle writers use `mkdtemp` + `rename` in the output parent. Both need same-volume NTFS. Everything here stays on `B:`, so it should be fine — but I did not test hard-link creation on this specific `B:` volume. **Determine by:** `New-Item -ItemType HardLink` into `validation/paper_data/fischer_2023/fig6/` before Step 3, then delete it.

8. **Whether the two byte-identical C0 raw bundle copies** (`tmp/author-runs/...` in-repo and `C:/tmp/qpsim-round7-fixes/tmp/author-runs/...`) can diverge. I used the in-repo copy throughout because `test_fig6_author_c2_evidence.py:37` hardcodes the repo-relative path. I verified the in-repo copy's manifest sha (`ebe32e48…56ac`) matches the accepted C0 score; I did **not** re-hash the `C:/tmp` copy's 24 `.npy` files against it.

---

**Files written during this analysis:** only under the session scratchpad (`.../scratchpad/inspect_manifests.py`, `ladder_audit.py`, `cost_probe.py`, `cost_probe2.py`, `c1_preflight.py`). The repo tree is unchanged — `git status --short` still reports only the pre-existing untracked `scratchpad/`, and the temporary `.npy` write test cleaned itself up.