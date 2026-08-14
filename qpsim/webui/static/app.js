/* qpsim frontend — no-build vanilla JS single page app. */
"use strict";

/* ---------- tiny helpers ---------- */

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

function getByPath(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setByPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k];
  o[keys[keys.length - 1]] = value;
}
async function api(path, opts) {
  const resp = await fetch(path, opts);
  const body = await resp.json().catch(() => ({}));
  return { ok: resp.ok, status: resp.status, body };
}
const postJSON = (path, data) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});
function fmt(v) {
  if (v == null) return "—";  // sanitized non-finite values arrive as null
  if (typeof v === "number") {
    if (v === 0) return "0";
    const a = Math.abs(v);
    return (a >= 1e5 || a < 1e-3) ? v.toExponential(4) : String(Math.round(v * 1e6) / 1e6);
  }
  return String(v);
}

/* ---------- form schema ---------- */

const F = (path, label, type = "number", opts = {}) => ({ path, label, type, ...opts });

const MATERIAL_FIELDS = (withTransport) => ({
  title: "Material",
  hint: "Pick a database material to autofill, then edit freely. Energies in μeV, times in ns.",
  fields: [
    F("material.name", "Material", "material"),
    F("material.Delta_0", "Δ₀ (μeV)"),
    F("material.T_c", "T_c (K)"),
    F("material.tau_0", "τ₀ e-ph (ns)"),
    F("material.tau_0_pb_ns", "τ₀^PB phonon-side (ns)", "number", { nullable: true }),
    ...(withTransport ? [F("material.D_0", "D₀ (μm²/ns)")] : []),
    F("material.rho_F", "ρ_F (eV⁻¹m⁻³)"),
    F("material.dynes_gamma", "Dynes Γ (μeV)"),
  ],
});
const GRID_FIELDS = {
  title: "Energy grid",
  hint: "Uniform cell-centered grid over [min, max]×Δ. Photon energies should be near-commensurate with the spacing (Validate checks this).",
  fields: [
    F("grid.min_factor", "E min (×Δ)"),
    F("grid.max_factor", "E max (×Δ)"),
    F("grid.num_bins", "Energy bins", "int"),
  ],
};
const BATH_FIELD = F("T_bath", "Bath temperature (K)");
const SUBGAP_FIELDS = {
  title: "Sub-gap photon drive (ω₀ < 2Δ)",
  fields: [
    F("subgap_drive.enabled", "Enabled", "check"),
    F("subgap_drive.omega_0", "ω₀ (μeV)"),
    F("subgap_drive.n_bar", "n̄ photons"),
    F("subgap_drive.c_phot", "c_phot (1/ns)"),
  ],
};
const PB_FIELDS = {
  title: "Pair-breaking photon drive (ω_PB > 2Δ)",
  fields: [
    F("pb_drive.enabled", "Enabled", "check"),
    F("pb_drive.omega_PB", "ω_PB (μeV)"),
    F("pb_drive.n_bar_PB", "n̄ photons"),
    F("pb_drive.c_phot_PB", "c_phot (1/ns)"),
  ],
};
const PROBE_FIELDS = {
  title: "Readout probe (Mattis–Bardeen observables)",
  hint: "σ₁/σ₂, Q_i and δω/ω at a sub-gap probe (needs ω₀ < Δ and Dynes Γ = 0).",
  fields: [
    F("probe.enabled", "Enabled", "check"),
    F("probe.omega_0", "Probe ω₀ (μeV)"),
    F("probe.alpha", "Kinetic-inductance α"),
    F("probe.Q_ext", "Q_ext cap", "number", { nullable: true }),
  ],
};

const FORMS = {
  steady_state_0d: [
    MATERIAL_FIELDS(false),
    { title: "Bath & grid", fields: [BATH_FIELD, ...GRID_FIELDS.fields], hint: GRID_FIELDS.hint },
    {
      title: "Phonon sector",
      hint: "thermal_bath: n_ph pinned at the bath (Newton). dynamic_escape: n_ph solved with finite escape τ_l. dynamic_closed: no substrate escape (τ_l → ∞).",
      fields: [
        F("phonons.mode", "Mode", "select", { options: ["thermal_bath", "dynamic_escape", "dynamic_closed"] }),
        F("phonons.tau_l_ns", "Escape τ_l (ns)"),
        F("phonons.use_phonon_side_kernel", "Phonon-side kernel (F&C Eq. 12)", "check"),
      ],
    },
    SUBGAP_FIELDS, PB_FIELDS,
    {
      title: "Phonon sector",
      hint: "thermal_bath pins n_ph at the bath. The dynamic modes solve the phonon population in every cell, which is far more expensive: the occupation matrices stop being shared, so each cell builds its own.",
      fields: [
        F("phonons.mode", "Mode", "select", { options: ["thermal_bath", "dynamic_escape", "dynamic_closed"] }),
        F("phonons.tau_l_ns", "Escape τ_l (ns)"),
      ],
    },
    {
      title: "Solver",
      fields: [
        F("solver.method", "Method", "select", { options: ["auto", "picard", "coupled_newton"] }),
        F("solver.self_consistent_gap", "Self-consistent gap", "check"),
        F("solver.picard_tol", "Picard tol"),
        F("solver.picard_max_iter", "Picard max iter", "int"),
        F("solver.picard_mixing", "Picard mixing"),
        F("solver.anderson_depth", "Anderson depth", "int"),
        F("solver.newton_tol", "Newton tol"),
        F("solver.newton_max_iter", "Newton max iter", "int"),
      ],
    },
    PROBE_FIELDS,
  ],
  transient_0d: [
    MATERIAL_FIELDS(false),
    { title: "Bath & grid", fields: [BATH_FIELD, ...GRID_FIELDS.fields], hint: GRID_FIELDS.hint },
    SUBGAP_FIELDS, PB_FIELDS,
    {
      title: "Time stepping (ETD2, frozen n_ph and Δ)",
      hint: "dt ≲ τ₀/10 keeps ETD2 well-behaved. Snapshots default to total/50.",
      fields: [
        F("dt", "dt (ns)"),
        F("total_time", "Total time (ns)"),
        F("snapshot_interval", "Snapshot interval (ns)", "number", { nullable: true }),
        F("stop_tol", "Early-stop tol (1/ns)", "number", { nullable: true }),
      ],
    },
    PROBE_FIELDS,
  ],
  spatial_1d: [
    MATERIAL_FIELDS(true),
    { title: "Bath & grid", fields: [BATH_FIELD, ...GRID_FIELDS.fields], hint: "Spatial transport needs Dynes Γ = 0." },
    {
      title: "Strip",
      fields: [
        F("length_um", "Length (μm)"),
        F("num_cells", "Spatial cells", "int"),
        F("diffusion_model", "Diffusion operator", "select", { options: ["A1", "A1P", "A2", "C", "B"] }),
      ],
    },
    {
      title: "Gap profile",
      hint: "step: left fraction at gap_left, rest at gap_right; a finite G_N makes the step a Kupriyanov–Lukichev interface.",
      fields: [
        F("gap_profile.kind", "Kind", "select", { options: ["uniform", "step"] }),
        F("gap_profile.gap_left", "Gap left (μeV)"),
        F("gap_profile.gap_right", "Gap right (μeV)"),
        F("gap_profile.step_position_fraction", "Step position (0–1)"),
        F("gap_profile.interface_G_N", "Interface G_N", "number", { nullable: true }),
      ],
    },
    {
      title: "Injection",
      hint: "Continuous Gaussian-in-energy QP source.",
      fields: [
        F("injection.enabled", "Enabled", "check"),
        F("injection.center_over_delta", "Line center (×Δ)"),
        F("injection.sigma_over_delta", "Line width σ (×Δ)"),
        F("injection.rate_per_ns", "Peak rate (1/ns)"),
        F("injection.where", "Where", "select", { options: ["left_end", "uniform"] }),
      ],
    },
    {
      title: "Time stepping",
      fields: [
        F("dt", "dt (ns)"),
        F("max_time", "Max time (ns)"),
        F("stop_tol", "Stop tol max|df/dt| (1/ns)"),
        F("snapshot_interval", "Snapshot interval (ns)", "number", { nullable: true }),
      ],
    },
  ],
  spatial_2d: [
    MATERIAL_FIELDS(true),
    { title: "Bath & grid", fields: [BATH_FIELD, ...GRID_FIELDS.fields], hint: "Spatial transport needs Dynes Γ = 0." },
    {
      title: "Geometry",
      hint: "The mask sets the dimensionality: rows = 1 is a 1D strip, 1×1 is a single 0-D cell. Cells are square, so mesh size fixes the resolution in both directions.",
      fields: [
        F("geometry.kind", "Source", "select", { options: ["rectangle", "gds"] }),
        F("geometry.rows", "Rows", "int"),
        F("geometry.cols", "Columns", "int"),
        F("geometry.mesh_size_um", "Mesh size (μm)"),
        F("geometry.gds_path", "GDS file", "text"),
        F("geometry.gds_layer", "GDS layer", "int"),
        F("geometry.require_connected", "Require one connected region", "check"),
      ],
    },
    {
      title: "Boundary",
      hint: "Applied to the whole device rim. reflective keeps everything in; absorbing and dirichlet let quasiparticles leave.",
      fields: [
        F("boundary.kind", "Condition", "select", { options: ["reflective", "absorbing", "dirichlet", "neumann"] }),
        F("boundary.value", "Value"),
      ],
    },
    {
      title: "Transport",
      fields: [
        F("diffusion_model", "Diffusion operator", "select", { options: ["A1", "A1P", "A2", "C", "B"] }),
      ],
    },
    SUBGAP_FIELDS, PB_FIELDS,
    {
      title: "Gap regions",
      hint: "column_step puts gap_left before the step fraction and gap_right after. In 2D that boundary is a curve of faces; a finite G_N makes every face along it a Kupriyanov–Lukichev barrier. It only shows when something drives a flux across the step.",
      fields: [
        F("gap_regions.kind", "Kind", "select", { options: ["uniform", "column_step"] }),
        F("gap_regions.gap_left", "Gap left (μeV)"),
        F("gap_regions.gap_right", "Gap right (μeV)"),
        F("gap_regions.step_fraction", "Step position (0–1)"),
        F("gap_regions.interface_G_N", "Interface G_N", "number", { nullable: true }),
      ],
    },
    {
      title: "Injection",
      hint: "Continuous Gaussian-in-energy source.",
      fields: [
        F("injection.enabled", "Enabled", "check"),
        F("injection.center_over_delta", "Line center (×Δ)"),
        F("injection.sigma_over_delta", "Line width σ (×Δ)"),
        F("injection.rate_per_ns", "Peak rate (1/ns)"),
        F("injection.where", "Where", "select", { options: ["left_edge", "uniform", "centre_cell"] }),
      ],
    },
    {
      title: "Gap closure",
      hint: "Re-solves Δ in every cell from that cell's own occupation, so a hot region digs its own gap well. The quantum bounds the collision grouping: a tenth of a grid spacing costs about 2e-3 in the kernel, while a full spacing costs 3.3e-1 because a bin enters or leaves the above-gap support.",
      fields: [
        F("self_consistent_gap", "Solve the gap", "check"),
        F("gap_quantum_over_dE", "Gap quantum (× dE)"),
      ],
    },
    {
      title: "Time stepping",
      fields: [
        F("dt", "dt (ns)"),
        F("max_time", "Max time (ns)"),
        F("stop_tol", "Stop tol max|df/dt| (1/ns)"),
      ],
    },
  ],
  m25_junction: [
    {
      title: "Junction (energies in GHz ÷ h)",
      hint: "Marchegiani 2025 gap-asymmetric JJ; defaults follow the Fig. 3 reproduction.",
      fields: [
        F("Delta_R_over_h_GHz", "Δ_R (GHz)"),
        F("omega_LR_over_h_GHz", "Gap asymmetry ω_LR (GHz)"),
        F("omega_10_over_h_GHz", "Qubit ω₁₀ (GHz)"),
        F("E_J_over_h_GHz", "E_J (GHz)"),
        F("E_C_over_h_GHz", "E_C (GHz)"),
        F("r_L_Hz", "Recombination r_L (Hz)"),
        F("r_Rlt_Hz", "Recombination r_R< (Hz)"),
        F("Gamma_ee_10_Hz", "Γ^ee₁₀ (Hz)"),
      ],
    },
    {
      title: "Photon drive",
      fields: [
        F("drive.omega_nu_GHz", "Drive ω_ν (GHz)"),
        F("drive.Gamma_ph_00_Hz", "Γ^ph₀₀ target (Hz)"),
        F("drive.nu_0_per_J_per_m3", "ν₀ (J⁻¹m⁻³)"),
        F("drive.volume_m3", "Volume (m³)"),
      ],
    },
    {
      title: "Temperature sweep",
      hint: "Points solve with continuation seeding from low T upward; for M25-like parameters the root is unique, so non-converged points usually want a denser T grid rather than a different branch picker.",
      fields: [
        F("T_start_mK", "T start (mK)"),
        F("T_stop_mK", "T stop (mK)"),
        F("T_points", "Points", "int"),
        F("branch_picker_mode", "Branch picker", "select", { options: ["lock_to_preferred", "min_residual"] }),
      ],
    },
  ],
};

/* ---------- state ---------- */

const state = {
  mode: "steady_state_0d",
  setup: null,          // current setup object (matches pydantic model)
  materials: [],
  modeLabels: {},
  pollTimer: null,
  detailRunId: null,
  lastRunsKey: null,    // change-detection: skip DOM rebuilds on identical polls
  lastDetailKey: null,
  catalogue: null,      // the catalogue is static; fetch it once
  catId: null,
  benchmarks: {},       // name -> declared closed form, from /api/benchmarks
};

/* ---------- new-run view ---------- */

async function switchMode(mode, presetSetup) {
  state.mode = mode;
  if (presetSetup) {
    state.setup = presetSetup;
  } else {
    const { body } = await api(`/api/defaults/${mode}`);
    state.setup = body;
  }
  renderForm();
  $("#feedback").innerHTML = "";
}

function renderForm() {
  renderStepForm("#setup-form", FORMS[state.mode]);
}

function renderStepForm(selector, sections) {
  const form = $(selector);
  form.innerHTML = "";
  for (const section of sections) {
    const det = document.createElement("details");
    det.open = true;
    const sum = document.createElement("summary");
    sum.textContent = section.title;
    det.appendChild(sum);
    if (section.hint) {
      const hint = document.createElement("div");
      hint.className = "hint";
      hint.textContent = section.hint;
      det.appendChild(hint);
    }
    const grid = document.createElement("div");
    grid.className = "field-grid";
    for (const field of section.fields) grid.appendChild(renderField(field));
    det.appendChild(grid);
    form.appendChild(det);
  }
}

function renderField(field) {
  const label = document.createElement("label");
  const name = document.createElement("span");
  name.className = "fname";
  name.textContent = field.label;
  label.appendChild(name);
  const value = getByPath(state.setup, field.path);

  let input;
  if (field.type === "check") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
    input.addEventListener("change", () => setByPath(state.setup, field.path, input.checked));
  } else if (field.type === "select") {
    input = document.createElement("select");
    for (const opt of field.options) {
      const o = document.createElement("option");
      o.value = opt; o.textContent = opt;
      input.appendChild(o);
    }
    input.value = value;
    input.addEventListener("change", () => setByPath(state.setup, field.path, input.value));
  } else if (field.type === "material") {
    input = document.createElement("select");
    const custom = document.createElement("option");
    custom.value = ""; custom.textContent = "(custom)";
    input.appendChild(custom);
    for (const mat of state.materials) {
      const o = document.createElement("option");
      o.value = mat.name; o.textContent = mat.name;
      input.appendChild(o);
    }
    input.value = state.materials.some((m) => m.name === value) ? value : "";
    input.addEventListener("change", () => {
      const mat = state.materials.find((m) => m.name === input.value);
      if (mat) {
        Object.assign(state.setup.material, mat.params);
        renderForm();
      }
    });
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = value == null ? "" : String(value);
    input.addEventListener("change", () => {
      const raw = input.value.trim();
      if (raw === "" && field.nullable) { setByPath(state.setup, field.path, null); return; }
      const num = field.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
      if (Number.isFinite(num)) {
        setByPath(state.setup, field.path, num);
      } else if (field.path.endsWith(".name")) {
        setByPath(state.setup, field.path, raw);
      } else {
        // Unparseable (or emptied non-nullable) input: resync the box
        // to the model so the screen never shows a value that won't
        // be submitted.
        const current = getByPath(state.setup, field.path);
        input.value = current == null ? "" : String(current);
      }
    });
  }
  input.dataset.path = field.path;
  label.appendChild(input);
  return label;
}

function envelope() {
  return { name: $("#run-name").value || "Untitled run", setup: state.setup };
}

function renderFeedback(body, okMessage) {
  const parts = [];
  for (const e of body.errors || []) parts.push(`<div class="error">✗ ${esc(e)}</div>`);
  for (const w of body.warnings || []) parts.push(`<div class="warning">⚠ ${esc(w)}</div>`);
  if (body.detail) {
    const msgs = Array.isArray(body.detail)
      ? body.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`)
      : [String(body.detail)];
    for (const m of msgs) parts.push(`<div class="error">✗ ${esc(m)}</div>`);
  }
  if (!parts.length && okMessage) parts.push(`<div class="ok">✓ ${esc(okMessage)}</div>`);
  $("#feedback").innerHTML = parts.join("");
}

async function doValidate() {
  const { body } = await postJSON("/api/validate", envelope());
  renderFeedback(body, "Setup is valid.");
}
async function doSaveSetup() {
  const { ok, body } = await postJSON("/api/setups", envelope());
  renderFeedback(body, ok ? `Saved as "${body.slug}".` : "");
}
async function doRun() {
  const { ok, body } = await postJSON("/api/runs", envelope());
  if (!ok) { renderFeedback(body, ""); return; }
  renderFeedback({ warnings: body.warnings }, "");
  showView("runs");
  state.detailRunId = body.id;
}

/* ---------- runs view ---------- */

async function awaitRun(runId, timeoutMs = 120000) {
  // Poll rather than block: a solve can outlast any single request, and the
  // run store is the only thing that knows when it is finished.
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { ok, body } = await api(`/api/runs/${runId}`);
    if (!ok) return null;
    if (body.status !== "queued" && body.status !== "running") return body;
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  return null;
}

async function refreshRuns() {
  const { body: runs } = await api("/api/runs");
  const runsKey = JSON.stringify(runs);
  if (runsKey === state.lastRunsKey) {
    // Nothing changed — leave the DOM (and its listeners) alone; the
    // detail pane below has its own change check.
    if (state.detailRunId) showRunDetail();
    return;
  }
  state.lastRunsKey = runsKey;
  const rows = runs.map((r) => {
    const progress = r.status === "running"
      ? `<div class="progress"><div style="width:${Math.round((r.progress || 0) * 100)}%"></div></div>
         <span class="hint">${esc(r.progress_message || "")}</span>`
      : "";
    const actions = (r.status === "running" || r.status === "queued")
      ? `<button class="btn" data-cancel="${esc(r.id)}">Cancel</button>`
      : `<button class="btn danger" data-delete="${esc(r.id)}">Delete</button>`;
    return `<tr class="clickable" data-run="${esc(r.id)}">
      <td>${esc(r.name)}</td><td>${esc(state.modeLabels[r.mode] || r.mode)}</td>
      <td><span class="status ${esc(r.status)}">${esc(r.status)}</span></td>
      <td>${progress}</td>
      <td>${r.elapsed_s != null ? fmt(r.elapsed_s) + " s" : ""}</td>
      <td>${esc(r.created || "")}</td><td>${actions}</td></tr>`;
  });
  $("#runs-list").innerHTML = runs.length
    ? `<table class="list"><tr><th>Name</th><th>Mode</th><th>Status</th><th>Progress</th>
       <th>Elapsed</th><th>Created</th><th></th></tr>${rows.join("")}</table>`
    : `<p class="hint">No runs yet — configure one under “New run”.</p>`;

  document.querySelectorAll("#runs-list [data-cancel]").forEach((b) =>
    b.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await postJSON(`/api/runs/${b.dataset.cancel}/cancel`, {});
      refreshRuns();
    }));
  document.querySelectorAll("#runs-list [data-delete]").forEach((b) =>
    b.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await api(`/api/runs/${b.dataset.delete}`, { method: "DELETE" });
      if (state.detailRunId === b.dataset.delete) { state.detailRunId = null; $("#run-detail").innerHTML = ""; }
      refreshRuns();
    }));
  document.querySelectorAll("#runs-list tr.clickable").forEach((tr) =>
    tr.addEventListener("click", () => { state.detailRunId = tr.dataset.run; showRunDetail(); }));

  if (state.detailRunId) showRunDetail();
}

async function showRunDetail() {
  const id = state.detailRunId;
  if (!id) return;
  const { ok, body: r } = await api(`/api/runs/${id}`);
  if (!ok) { $("#run-detail").innerHTML = ""; state.lastDetailKey = null; return; }
  const detailKey = JSON.stringify(r);
  if (detailKey === state.lastDetailKey) return;  // unchanged — keep DOM (and imgs) intact
  state.lastDetailKey = detailKey;
  // Preserve the "Setup used" expansion across progress-driven rebuilds.
  const setupWasOpen = !!document.querySelector("#run-detail details")?.open;
  let html = `<h2>${esc(r.name)} <span class="status ${esc(r.status)}">${esc(r.status)}</span></h2>
    <div class="hint">${esc(state.modeLabels[r.mode] || r.mode)} · ${esc(r.created || "")}
    ${r.elapsed_s != null ? " · " + fmt(r.elapsed_s) + " s" : ""}</div>`;
  for (const n of r.notes || []) html += `<div class="note">⚠ ${esc(n)}</div>`;
  if (r.error) html += `<div class="note">✗ ${esc(r.error)}</div>`;

  if (r.status === "done") {
    const entries = Object.entries(r.summary || {});
    if (entries.length) {
      html += `<table class="list summary-table">${entries.map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td>${esc(fmt(v))}</td></tr>`).join("")}</table>`;
    }
    html += `<div class="plot-grid">${(r.plots || []).map((p) =>
      `<img src="/api/runs/${esc(id)}/plots/${esc(p)}.png" alt="${esc(p)}" loading="lazy">`).join("")}</div>`;
    html += `<div class="downloads">${(r.csvs || []).map((c) =>
      `<a class="btn" href="/api/runs/${esc(id)}/csv/${esc(c)}.csv">⬇ ${esc(c)}.csv</a>`).join("")}</div>`;
  }
  html += `<details><summary>Setup used</summary><pre class="json">${esc(JSON.stringify(r.setup, null, 2))}</pre></details>`;
  $("#run-detail").innerHTML = html;
  if (setupWasOpen) {
    const details = document.querySelector("#run-detail details");
    if (details) details.open = true;
  }
}

/* ---------- setups view ---------- */

async function refreshSetups() {
  const { body: setups } = await api("/api/setups");
  $("#setups-list").innerHTML = setups.length
    ? `<table class="list"><tr><th>Name</th><th>Mode</th><th>Saved</th><th></th></tr>` +
      setups.map((s) => `<tr>
        <td>${esc(s.name)}</td><td>${esc(state.modeLabels[s.mode] || s.mode)}</td>
        <td>${esc(s.saved_at || "")}</td>
        <td><button class="btn" data-load="${esc(s.slug)}">Load</button>
            <button class="btn danger" data-del="${esc(s.slug)}">Delete</button></td></tr>`).join("") +
      `</table>`
    : `<p class="hint">No saved setups.</p>`;
  document.querySelectorAll("#setups-list [data-load]").forEach((b) =>
    b.addEventListener("click", async () => {
      const { body } = await api(`/api/setups/${b.dataset.load}`);
      $("#run-name").value = body.name;
      showView("new-run");
      switchMode(body.setup.mode, body.setup);
    }));
  document.querySelectorAll("#setups-list [data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      await api(`/api/setups/${b.dataset.del}`, { method: "DELETE" });
      refreshSetups();
    }));
}

/* ---------- materials view ---------- */

async function refreshMaterials() {
  const cols = [
    ["Delta_0", "Δ₀ (μeV)"], ["T_c", "T_c (K)"], ["tau_0", "τ₀ (ns)"],
    ["tau_0_phonon", "τ₀^ph (ns)"], ["tau_0_pb_ns", "τ₀^PB (ns)"], ["D_0", "D₀ (μm²/ns)"],
    ["v_F", "v_F (m/s)"], ["rho_F", "ρ_F (eV⁻¹m⁻³)"], ["film_thickness", "d (nm)"],
    ["substrate_transmission_eta", "η"],
  ];
  // D_0 is shown against the band its own references support, because it
  // varies by a factor of five or more across film qualities of the same
  // material -- so the stored scalar is not by itself something to choose
  // with, and where it sits outside its band the reader should see that.
  const outOfBand = (m) => Array.isArray(m.D_0_range) && m.D_0 != null
    && (m.D_0 < m.D_0_range[0] || m.D_0 > m.D_0_range[1]);

  $("#materials-list").innerHTML =
    `<table class="list"><tr><th>Material</th>${cols.map(([, l]) => `<th>${esc(l)}</th>`).join("")}` +
    `<th>D₀ band</th><th>Substrate</th></tr>` +
    state.materials.map((m) => `<tr><td><b>${esc(m.name)}</b></td>` +
      cols.map(([k]) => k === "D_0" && outOfBand(m)
        ? `<td class="out-of-band" title="Outside the band this entry's own references support">${esc(fmt(m[k]))} ⚠</td>`
        : `<td>${m[k] == null ? "—" : esc(fmt(m[k]))}</td>`).join("") +
      `<td>${Array.isArray(m.D_0_range)
        ? `${esc(fmt(m.D_0_range[0]))}–${esc(fmt(m.D_0_range[1]))}` : "—"}</td>` +
      `<td>${m.substrate ? esc(m.substrate.name) : "—"}</td></tr>`).join("") +
    `</table>
    <p class="hint">From the YAML database at qpsim/materials/data/. Picking a material in the
    setup editor copies these values; edits there don't touch the database.
    A ⚠ marks a stored D₀ outside the band its own references support — the
    value is used as stored, and the sources below say what it should be.</p>` +
    `<div class="sources">` + state.materials.map((m) => `
      <details${outOfBand(m) ? " open" : ""}>
        <summary><b>${esc(m.name)}</b>${outOfBand(m)
          ? ` <span class="out-of-band">D₀ outside its sourced band</span>` : ""}</summary>
        ${m.notes ? `<p>${esc(m.notes)}</p>` : ""}
        <ul>${(m.references || []).map((r) => `<li>${esc(r)}</li>`).join("")
          || "<li>no references recorded</li>"}</ul>
      </details>`).join("") + `</div>`;
}

/* ---------- view switching & init ---------- */

/* ---------- test simulations ---------- */

/* Each case is defaults-for-its-mode plus a small override map, so a case only
   states what it changes and picks up any later change to the defaults. An
   override naming a path the model does not have is reported rather than
   dropped: a silently ignored override would make a case claim to test
   something it is not testing. */
function applyOverrides(setup, overrides) {
  const unknown = [];
  for (const [path, value] of Object.entries(overrides || {})) {
    const parts = path.split(".");
    let node = setup;
    for (const key of parts.slice(0, -1)) {
      if (node === null || typeof node !== "object" || !(key in node)) { node = null; break; }
      node = node[key];
    }
    const leaf = parts[parts.length - 1];
    if (node === null || typeof node !== "object" || !(leaf in node)) { unknown.push(path); continue; }
    node[leaf] = value;
  }
  return unknown;
}

async function buildCaseSetup(tc) {
  const { body } = await api(`/api/defaults/${tc.mode}`);
  const unknown = applyOverrides(body, tc.overrides);
  return { setup: body, unknown };
}

async function openCase(tc) {
  const { setup, unknown } = await buildCaseSetup(tc);
  $("#run-name").value = tc.title;
  await switchMode(tc.mode, setup);
  showView("new-run");
  if (unknown.length) {
    $("#feedback").innerHTML =
      `<div class="err">This case names setup fields that do not exist: ${
        unknown.map(esc).join(", ")}. It was loaded without them.</div>`;
  }
}

async function runCase(tc, btn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const { setup, unknown } = await buildCaseSetup(tc);
    if (unknown.length) {
      throw new Error(`unknown setup fields: ${unknown.join(", ")}`);
    }
    const { ok, status, body } = await postJSON("/api/runs", {
      name: tc.title, setup, benchmark: tc.benchmark ?? null,
    });
    if (!ok) {
      const problems = (body.errors || [String(status)]).join("; ");
      throw new Error(problems);
    }
    if (!tc.expect && !tc.benchmark) {
      showView("runs");
      return;
    }
    // Stay put and report the verdict: the point of a case with a closed form
    // is the comparison, and sending the user to the runs list hides it.
    btn.textContent = "Checking…";
    const holder = document.getElementById(`case-msg-${tc.id}`);
    const finished = await awaitRun(body.id);
    btn.textContent = label;
    btn.disabled = false;
    if (holder) {
      holder.innerHTML = "";
      if (finished === null || finished.status !== "done") {
        holder.innerHTML = `<div class="err">The run did not finish`
          + `${finished && finished.error ? `: ${esc(finished.error)}` : "."}</div>`;
        return;
      }
      const summary = finished.summary || {};
      // A curve comparison supersedes the scalar one where both exist: it is
      // the same statement checked at every point rather than at one.
      if (summary.benchmark) {
        renderBenchmark(holder, summary.benchmark, body.id);
      } else if (tc.benchmark) {
        holder.innerHTML = `<div class="err">This case asks for the `
          + `<code>${esc(tc.benchmark)}</code> comparison, but the run reported `
          + `none. The benchmark may not apply to this mode.</div>`;
      } else {
        renderExpectation(holder, tc.expect, summary);
      }
    }
    return;
  } catch (exc) {
    btn.textContent = label;
    btn.disabled = false;
    const holder = document.getElementById(`case-msg-${tc.id}`);
    if (holder) holder.innerHTML = `<div class="err">Could not start: ${esc(exc.message)}</div>`;
  }
}

/* ---------- catalogue: category -> item -> case ---------- */

async function loadCatalogue() {
  if (state.catalogue) return state.catalogue;
  const resp = await fetch("/static/catalogue.json");
  if (!resp.ok) throw new Error(`catalogue unavailable (${resp.status})`);
  state.catalogue = await resp.json();
  return state.catalogue;
}

/* Level 1: the categories, reached from Home > Test simulations. */
async function openTestSimulations() {
  let cat;
  try {
    cat = await loadCatalogue();
  } catch (exc) {
    $("#cat-body").innerHTML = `<div class="err">${esc(exc.message)}</div>`;
    showView("catalogue");
    return;
  }
  crumbs([
    { label: "Home", go: () => showView("home") },
    { label: "Test simulations" },
  ]);
  $("#cat-title").textContent = "Test simulations";
  $("#cat-blurb").textContent =
    "Reductions of the kinetic equations, grouped by how many terms are on, " +
    "plus reproductions of prior literature results.";

  const host = $("#cat-body");
  host.innerHTML = "";
  const sec = document.createElement("section");
  sec.className = "group";
  for (const c of cat.categories || []) {
    const n = (c.items || []).reduce(
      (acc, it) => acc + (it.cases || it.links || []).length, 0);
    const b = document.createElement("button");
    b.className = "action";
    b.innerHTML =
      `<span class="name">${esc(c.title)}</span>` +
      `<span class="go">${n} ${n === 1 ? "case" : "cases"}</span>` +
      `<span class="desc">${esc(c.blurb)}</span>`;
    b.addEventListener("click", () => {
      if (document.body.classList.contains("editing")) return;
      openCategory(c.id);
    });
    sec.appendChild(b);
  }
  host.appendChild(sec);
  showView("catalogue");
  window.scrollTo(0, 0);
}

function crumbs(trail) {
  crumbsInto("#crumbs", trail);
}

function crumbsInto(selector, trail) {
  const host = $(selector);
  host.innerHTML = "";
  trail.forEach((step, i) => {
    if (i) host.appendChild(document.createTextNode(" / "));
    if (step.go) {
      const a = document.createElement("button");
      a.className = "crumb";
      a.textContent = step.label;
      a.addEventListener("click", step.go);
      host.appendChild(a);
    } else {
      const s = document.createElement("span");
      s.className = "crumb current";
      s.textContent = step.label;
      host.appendChild(s);
    }
  });
}

/* Level 2: the items of one category. */
async function openCategory(catId) {
  const cat = (await loadCatalogue()).categories.find((c) => c.id === catId);
  if (!cat) return;
  state.catId = catId;
  crumbs([
    { label: "Home", go: () => showView("home") },
    { label: "Test simulations", go: () => openTestSimulations() },
    { label: cat.title },
  ]);
  $("#cat-title").textContent = cat.title;
  $("#cat-blurb").textContent = cat.blurb || "";

  const host = $("#cat-body");
  host.innerHTML = "";
  const sec = document.createElement("section");
  sec.className = "group";
  for (const it of cat.items || []) {
    const count = (it.cases || it.links || []).length;
    const b = document.createElement("button");
    b.className = "action" + (it.available === false ? " unavailable" : "");
    b.innerHTML =
      `<span class="name">${esc(it.title)}</span>` +
      `<span class="go">${it.available === false ? "No setup" : count}</span>` +
      `<span class="desc">${esc(it.blurb || "")}</span>`;
    b.addEventListener("click", () => {
      if (document.body.classList.contains("editing")) return;
      openItem(catId, it.id);
    });
    sec.appendChild(b);
  }
  host.appendChild(sec);
  showView("catalogue");
  window.scrollTo(0, 0);
}

/* Level 3: the cases of one item. */
async function openItem(catId, itemId) {
  const cat = (await loadCatalogue()).categories.find((c) => c.id === catId);
  const it = cat && (cat.items || []).find((x) => x.id === itemId);
  if (!it) return;
  crumbs([
    { label: "Home", go: () => showView("home") },
    { label: "Test simulations", go: () => openTestSimulations() },
    { label: cat.title, go: () => openCategory(catId) },
    { label: it.title },
  ]);
  $("#cat-title").textContent = it.title;
  $("#cat-blurb").textContent = it.blurb || "";

  const host = $("#cat-body");
  host.innerHTML = "";

  if (it.available === false) {
    const note = document.createElement("div");
    note.className = "note";
    note.innerHTML =
      `<b>Not reachable from this interface.</b> ${esc(it.why || "")}` +
      (it.evidence ? ` Covered by <code>${esc(it.evidence)}</code>.` : "");
    host.appendChild(note);
    return;
  }

  /* Plain navigation rows (the Miscellaneous "rest of the interface" item). */
  if (it.links) {
    const sec = document.createElement("section");
    sec.className = "group";
    for (const l of it.links) {
      const b = document.createElement("button");
      b.className = "action";
      b.innerHTML =
        `<span class="name">${esc(l.title)}</span>` +
        `<span class="go">Open</span>` +
        `<span class="desc">${esc(l.desc || "")}</span>`;
      b.addEventListener("click", () => {
        if (document.body.classList.contains("editing")) return;
        showView(l.go);
      });
      sec.appendChild(b);
    }
    host.appendChild(sec);
    return;
  }

  for (const tc of it.cases || []) {
    const card = document.createElement("article");
    card.className = "test-case";
    card.innerHTML =
      `<div class="tc-head">
         <span class="tc-title">${esc(tc.title)}</span>
         <span class="tc-mode">${esc(state.modeLabels[tc.mode] || tc.mode)}</span>
       </div>
       <p class="tc-summary">${esc(tc.summary)}</p>
       <div class="tc-actions"></div>
       <div id="case-msg-${esc(tc.id)}"></div>`;
    // The claim is stated BEFORE the run, from the server's registry, so the
    // form shown here is the identical string the verdict is computed against.
    const declared = (tc.benchmark && state.benchmarks[tc.benchmark]) || tc.expect;
    if (declared) {
      const stated = document.createElement("div");
      stated.className = "expectation";
      const tier = declared.tier
        ? `<span class="ex-tier" title="${esc(TIER_NOTE[declared.tier] || "")}">`
          + `${esc(declared.tier)}</span>`
        : "";
      stated.innerHTML =
        `<div class="ex-head"><span class="ex-verdict">Expected</span>${tier}`
        + `<span class="ex-formula">${esc(declared.formula_latex)}</span></div>`
        + `<p class="ex-reason">${esc(declared.reason)}</p>`;
      card.insertBefore(stated, card.querySelector(".tc-actions"));
    }
    const actions = card.querySelector(".tc-actions");
    const open = document.createElement("button");
    open.textContent = "Open in editor";
    open.addEventListener("click", () => openCase(tc));
    const run = document.createElement("button");
    run.className = "primary";
    run.textContent = "Run";
    run.addEventListener("click", () => runCase(tc, run));
    actions.append(open, run);
    host.appendChild(card);
  }
}

function showView(name) {
  const target = document.getElementById(`view-${name}`);
  if (target === null) {
    // Hiding every view and matching none renders a blank page with nothing
    // in the console. Fall back to Home and say so, so a routing mistake is
    // visible instead of looking like a broken app.
    console.error(`showView: no section for "${name}"; falling back to home.`);
    name = "home";
  }
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("hidden", v.id !== `view-${name}`));
  clearInterval(state.pollTimer);
  if (name === "runs") {
    refreshRuns();
    state.pollTimer = setInterval(refreshRuns, 2000);
  } else if (name === "setups") {
    refreshSetups();
  } else if (name === "materials") {
    refreshMaterials();
  }
}

async function init() {
  const [{ body: meta }, { body: mats }, { body: bench }] = await Promise.all([
    api("/api/meta"), api("/api/materials"), api("/api/benchmarks"),
  ]);
  state.modeLabels = meta.modes || {};
  state.materials = mats || [];
  state.benchmarks = bench || {};
  $("#meta").textContent = `qpsim ${meta.qpsim_version} · workspace ${meta.workspace}`;
  $("#home-workspace").textContent = meta.workspace;

  $("#go-home").addEventListener("click", () => {
    if (document.body.classList.contains("editing")) return;
    showView("home");
    window.scrollTo(0, 0);
  });
  // Home rows are shortcuts into the same views the nav reaches; rows with no
  // data-go are the not-yet-built ones and stay inert.
  document.querySelectorAll(".action[data-go]").forEach((b) =>
    b.addEventListener("click", () => {
      // While the wording is being edited, a row is a text field, not a link.
      if (document.body.classList.contains("editing")) return;
      // "tests" is not a view of its own: it is the catalogue opened at its
      // top level, so route it rather than letting showView hunt for a
      // section that does not exist.
      if (b.dataset.go === "tests") {
        openTestSimulations();
        return;
      }
      showView(b.dataset.go);
      window.scrollTo(0, 0);
    }));
  $("#btn-validate").addEventListener("click", doValidate);
  $("#btn-save-setup").addEventListener("click", doSaveSetup);
  $("#btn-run").addEventListener("click", doRun);

  // One mode: the geometry decides the dimensionality, so there is nothing
  // for the user to pick between.
  await switchMode("spatial_2d");
  initWizard();
  showWizardStep(0);
}

init();

/* =====================================================================
   Copy edit mode
   Lets the wording of this interface be edited in the browser, so text
   can be judged in place instead of in a file.

   Deliberately front-end only. Writing edits back to disk would need a
   new route in server.py, which is a .py under qpsim/ and therefore
   inside the source digest -- a convenience feature would cost a full
   recertification. So edits live in localStorage and leave as a JSON
   file that carries the ORIGINAL string alongside the new one, which is
   what makes them applicable to the source afterwards.

   Nodes are keyed by a hash of their original text rather than by a
   markup attribute, so nothing has to be tagged and dynamically
   rendered content is picked up for free. Two nodes with identical
   original text share a key and therefore edit together; that is the
   one known limitation.
   ===================================================================== */

const COPY_STORE_KEY = "qpsim.copyEdits.v1";
const COPY_SELECTOR = [
  ".home h1", ".home .lede", ".eyebrow", ".group > h2",
  ".action .name", ".action .go", ".action .desc",
  ".workspace b", ".home .foot",
  "#cat-title", "#cat-blurb",
  ".tc-title", ".tc-summary",
  "header .subtitle", ".crumb",
].join(",");

const copyState = { edits: {}, on: false, applying: false };

function copyHash(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

function loadCopyEdits() {
  try {
    copyState.edits = JSON.parse(localStorage.getItem(COPY_STORE_KEY) || "{}");
  } catch {
    copyState.edits = {};
  }
}

function saveCopyEdits() {
  localStorage.setItem(COPY_STORE_KEY, JSON.stringify(copyState.edits));
  const n = Object.keys(copyState.edits).length;
  const el = $("#eb-count");
  if (el) el.textContent = n === 0 ? "no changes" : `${n} change${n === 1 ? "" : "s"}`;
}

/* Index every copy node we have not seen, then push any stored override
   onto it. Only pure-text elements qualify: an element with element
   children would need innerHTML editing, which is not worth the risk. */
function applyCopyOverrides() {
  if (copyState.applying) return;
  copyState.applying = true;
  try {
    for (const el of document.querySelectorAll(COPY_SELECTOR)) {
      if (el.children.length > 0) continue;
      // Some nodes (#cat-title, #cat-blurb) are reused as the drill-down
      // navigates, so a key cached on first sight would go stale. Re-key
      // whenever the live text matches neither the remembered original nor
      // the edit stored against it.
      const seen = el.dataset.copyOrig;
      const known = seen && copyState.edits[el.dataset.copyKey];
      const live = el.textContent.trim();
      if (!live) continue;
      if (!seen || (live !== seen && (!known || live !== known.edited))) {
        el.dataset.copyOrig = live;
        el.dataset.copyKey = copyHash(live);
      }
      const edit = copyState.edits[el.dataset.copyKey];
      // Never rewrite the node being typed into: replacing textContent would
      // collapse the selection and throw the caret to the start.
      if (edit && el.textContent !== edit.edited && el !== document.activeElement) {
        el.textContent = edit.edited;
      }
      el.classList.toggle("edited", Boolean(edit));
      if (copyState.on) el.setAttribute("contenteditable", "plaintext-only");
      else el.removeAttribute("contenteditable");
    }
  } finally {
    copyState.applying = false;
  }
}

function setCopyEditMode(on) {
  copyState.on = on;
  document.body.classList.toggle("editing", on);
  $("#edit-bar").classList.toggle("hidden", !on);
  $("#btn-edit-copy").classList.toggle("active", on);
  applyCopyOverrides();
  saveCopyEdits();
}

function exportCopyEdits() {
  const rows = Object.entries(copyState.edits).map(([key, v]) => ({
    key, original: v.original, edited: v.edited,
  }));
  const blob = new Blob([JSON.stringify({ edits: rows }, null, 2)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "qpsim-ui-copy-edits.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

function revertCopyEdits() {
  for (const el of document.querySelectorAll("[data-copy-orig]")) {
    el.textContent = el.dataset.copyOrig;
  }
  copyState.edits = {};
  saveCopyEdits();
}

function initCopyEditing() {
  loadCopyEdits();

  document.addEventListener("input", (ev) => {
    const el = ev.target;
    if (!copyState.on || !el.dataset || !el.dataset.copyKey) return;
    const original = el.dataset.copyOrig;
    const edited = el.textContent.trim();
    if (edited === original) delete copyState.edits[el.dataset.copyKey];
    else copyState.edits[el.dataset.copyKey] = { original, edited };
    el.classList.toggle("edited", edited !== original);
    saveCopyEdits();
  });

  $("#btn-edit-copy").addEventListener("click", () => setCopyEditMode(!copyState.on));
  $("#btn-copy-done").addEventListener("click", () => setCopyEditMode(false));
  $("#btn-copy-export").addEventListener("click", exportCopyEdits);
  $("#btn-copy-revert").addEventListener("click", () => {
    if (confirm("Discard every text edit and restore the shipped wording?")) revertCopyEdits();
  });

  // Dynamically rendered views (runs, setups, test cases) replace whole
  // subtrees, so re-index on any change rather than patching each renderer.
  new MutationObserver(() => applyCopyOverrides())
    .observe(document.querySelector("main"), { childList: true, subtree: true });

  applyCopyOverrides();
}

initCopyEditing();

/* =====================================================================
   New-run wizard: equations -> geometry -> conditions
   ===================================================================== */

/* Which setup field each clickable term maps onto.
   `path` null means the unified 2-D mode cannot express that term yet, and
   `why` says what is missing. Showing the term greyed with a reason is
   honest; hiding it would misrepresent the model, and wiring it to nothing
   would be worse. */
const TERM_FIELDS = {
  diff:    { path: "material.D_0", kind: "zeroable", label: "Diffusion" },
  scat:    { path: "collisions.scattering", kind: "flag",
             label: "Quasiparticle–phonon scattering" },
  recomb:  { path: "collisions.recombination", kind: "flag",
             label: "Recombination and pair breaking" },
  src:     { path: "injection.enabled", kind: "flag", label: "External injection" },
  psc:     { path: "collisions.phonon_scattering_source", kind: "flag",
             label: "Phonon scattering source" },
  prc:     { path: "collisions.phonon_recombination_source", kind: "flag",
             label: "Phonon recombination source" },
  photsg:  { path: "subgap_drive.enabled", kind: "flag",
             label: "Sub-gap photon drive" },
  photpb:  { path: "pb_drive.enabled", kind: "flag",
             label: "Pair-breaking photon drive" },
  pesc:    { path: "phonons.mode", kind: "mode",
             label: "Phonon-bath coupling",
             on: "dynamic_escape", off: "dynamic_closed" },
  gapeq:   { path: "self_consistent_gap", kind: "flag",
             label: "Self-consistent gap" },
};

const GEOMETRY_SECTIONS = FORMS.spatial_2d.filter(
  (section) => section.title === "Geometry" || section.title === "Boundary",
);
const CONDITION_SECTIONS = FORMS.spatial_2d.filter(
  (section) => section.title !== "Geometry" && section.title !== "Boundary",
);

const WIZARD_STEPS = ["equations", "geometry", "conditions"];
const wizard = { index: 0, offD0: null };

function termIsOn(id) {
  const spec = TERM_FIELDS[id];
  if (!spec || spec.path === null) return false;
  const value = getByPath(state.setup, spec.path);
  if (spec.kind === "zeroable") return Number(value) > 0;
  if (spec.kind === "mode") return value === spec.on;
  return Boolean(value);
}

function setTerm(id, on) {
  const spec = TERM_FIELDS[id];
  if (!spec || spec.path === null) return;
  if (spec.kind === "zeroable") {
    // D_0 = 0 IS the transport off switch, so remember the value being
    // switched away from rather than forcing the user to retype it.
    if (on) {
      setByPath(state.setup, spec.path, wizard.offD0 ?? 6.0);
    } else {
      wizard.offD0 = Number(getByPath(state.setup, spec.path)) || wizard.offD0;
      setByPath(state.setup, spec.path, 0.0);
    }
  } else if (spec.kind === "mode") {
    // Off is a CLOSED sector, not a pinned bath: the term being dropped is
    // the coupling to the substrate, and the phonons still evolve.
    setByPath(state.setup, spec.path, on ? spec.on : spec.off);
  } else {
    setByPath(state.setup, spec.path, on);
  }
}

function renderTermPanel() {
  const cons = [];
  for (const button of document.querySelectorAll(".term[data-term]")) {
    const id = button.dataset.term;
    const spec = TERM_FIELDS[id];
    if (button.classList.contains("locked")) continue;
    const unavailable = !spec || spec.path === null;
    const on = termIsOn(id);
    button.classList.toggle("off", unavailable || !on);
    button.classList.toggle("clamped", unavailable);
    button.title = unavailable
      ? `${spec ? spec.label : id}: ${spec ? spec.why : "not available"}.`
      : `${spec.label} — click to ${on ? "drop" : "restore"} it.`;
  }
  // A sign belongs to the term it introduces and fades with it.
  for (const op of document.querySelectorAll(".op[data-op-for]")) {
    const target = document.querySelector(
      `.term[data-term="${op.dataset.opFor}"]`);
    op.classList.toggle("dim", target !== null && target.classList.contains("off"));
  }

  const live = Object.keys(TERM_FIELDS).filter(
    (id) => TERM_FIELDS[id].path !== null && termIsOn(id));
  if (!live.includes("scat") && !live.includes("recomb")) {
    cons.push(["bad", "No electron-phonon collisions: nothing relaxes the "
      + "quasiparticle energies or changes their number."]);
  }
  if (live.includes("src") && !live.includes("recomb")) {
    cons.push(["bad", "Quasiparticles are injected and nothing removes them, "
      + "so there is no steady state to find."]);
  }
  if (!live.includes("diff")) {
    cons.push(["", "No spatial transport: every cell evolves independently, "
      + "whatever the geometry."]);
  }
  const split = ["scat", "recomb"].filter((qp) => {
    const ph = qp === "scat" ? "psc" : "prc";
    return termIsOn(qp) !== termIsOn(ph);
  });
  if (split.length) {
    cons.push(["bad", `Energy conservation is not being tracked: the `
      + `${split.join(" and ")} channel is on for one population and off for `
      + `the other.`]);
  }
  const unavailable = Object.values(TERM_FIELDS).filter((t) => t.path === null);
  if (unavailable.length) {
    cons.push(["", `${unavailable.length} terms are greyed because this solver `
      + `cannot express them yet; hover one for the reason.`]);
  }

  const host = $("#cons");
  host.innerHTML = "";
  for (const [cls, text] of cons) {
    const li = document.createElement("li");
    li.className = cls;
    li.innerHTML = `<span class="mk"></span><span>${esc(text)}</span>`;
    host.appendChild(li);
  }
}

function showWizardStep(index) {
  wizard.index = Math.max(0, Math.min(WIZARD_STEPS.length - 1, index));
  const name = WIZARD_STEPS[wizard.index];
  document.querySelectorAll(".step-panel").forEach((panel) =>
    panel.classList.toggle("hidden", panel.id !== `step-${name}`));
  document.querySelectorAll("#wizard-steps li").forEach((li, i) => {
    li.classList.toggle("current", i === wizard.index);
    li.classList.toggle("done", i < wizard.index);
  });
  $("#btn-back").classList.toggle("hidden", wizard.index === 0);
  $("#btn-next").classList.toggle("hidden", wizard.index === WIZARD_STEPS.length - 1);
  const last = wizard.index === WIZARD_STEPS.length - 1;
  for (const id of ["#btn-validate", "#btn-save-setup", "#btn-run"]) {
    $(id).classList.toggle("hidden", !last);
  }
  crumbsInto("#wizard-crumbs", [
    { label: "Home", go: () => showView("home") },
    { label: "New run" },
  ]);
  if (name === "equations") renderTermPanel();
  if (name === "geometry") renderStepForm("#form-geometry", GEOMETRY_SECTIONS);
  if (name === "conditions") renderStepForm("#setup-form", CONDITION_SECTIONS);
  window.scrollTo(0, 0);
}

function initWizard() {
  for (const button of document.querySelectorAll(".term[data-term]")) {
    button.addEventListener("click", () => {
      if (document.body.classList.contains("editing")) return;
      const id = button.dataset.term;
      if (button.classList.contains("locked")) {
        button.classList.remove("nudge");
        void button.offsetWidth;          // restart the animation
        button.classList.add("nudge");
        return;
      }
      if (!TERM_FIELDS[id] || TERM_FIELDS[id].path === null) return;
      setTerm(id, !termIsOn(id));
      renderTermPanel();
    });
  }
  $("#btn-next").addEventListener("click", () => showWizardStep(wizard.index + 1));
  $("#btn-back").addEventListener("click", () => showWizardStep(wizard.index - 1));
}

/* =====================================================================
   Analytic expectations
   A case may state a closed form and how to check it. The reference is
   another summary field wherever possible, so the target is COMPUTED by
   the run rather than typed into a description by hand -- a number in
   prose stops being true the moment the engine moves, and says nothing
   when it does.
   ===================================================================== */

function evaluateExpectation(expect, summary) {
  const got = Number(summary[expect.observable]);
  const want = Number(summary[expect.reference]);
  if (!Number.isFinite(got) || !Number.isFinite(want)) {
    return {
      verdict: "unknown",
      detail: `the run reported no ${!Number.isFinite(got)
        ? expect.observable : expect.reference}`,
    };
  }
  if (expect.comparison === "greater") {
    const factor = expect.factor ?? 1.0;
    return {
      verdict: got > want * factor ? "pass" : "fail",
      got, want,
      detail: `${fmt(got)} vs ${fmt(want)} × ${factor} = ${fmt(want * factor)}`,
    };
  }
  const scale = Math.abs(want) > 0 ? Math.abs(want) : 1.0;
  const rel = Math.abs(got - want) / scale;
  return {
    verdict: rel <= (expect.rel_tol ?? 1e-6) ? "pass" : "fail",
    got, want, rel,
    detail: `relative difference ${rel.toExponential(2)} `
          + `against a tolerance of ${(expect.rel_tol ?? 1e-6).toExponential(0)}`,
  };
}

function renderExpectation(host, expect, summary) {
  const result = evaluateExpectation(expect, summary);
  const box = document.createElement("div");
  box.className = `expectation ${result.verdict}`;
  const label = { pass: "Matches", fail: "Does not match", unknown: "Not checked" };
  box.innerHTML =
    `<div class="ex-head"><span class="ex-verdict">${label[result.verdict]}</span>`
    + `<span class="ex-formula">${esc(expect.formula_latex)}</span></div>`
    + `<p class="ex-reason">${esc(expect.reason)}</p>`
    + `<p class="ex-detail">${esc(expect.observable)} = `
    + `${result.got === undefined ? "—" : fmt(result.got)}; ${esc(result.detail)}</p>`;
  host.appendChild(box);
  return result;
}

/* ---------- full-curve analytic benchmarks ----------
   A scalar check can be satisfied by a solution that is wrong everywhere
   except the moment it is read. These compare the WHOLE curve -- the server
   builds the closed form on the run's own grid and scores it pointwise -- so
   what is shown here is the trajectory, not one number off the end of it. */

const TIER_NOTE = {
  T1: "Closed form, written from the physics. The analytic side never consults "
     + "the engine's kernels, so agreement is evidence about the physics.",
  T2: "Exact solution of the reduced problem, assembled from the engine's own "
     + "kernel arrays. It checks operator assembly and time integration, but it "
     + "reuses the kernel under test and so cannot detect a wrong kernel.",
  T3: "An independently written quadrature of the same physics.",
};

function renderBenchmark(host, bench, runId) {
  const verdict = bench.verdict || "unknown";
  const box = document.createElement("div");
  box.className = `expectation benchmark ${verdict}`;
  const label = { pass: "Matches", fail: "Does not match", unknown: "Not checked" };
  const tier = bench.tier || "";
  const err = Number(bench.error);
  const tol = Number(bench.rel_tol);

  let html =
    `<div class="ex-head"><span class="ex-verdict">${label[verdict] || verdict}</span>`
    + `<span class="ex-tier" title="${esc(TIER_NOTE[tier] || "")}">${esc(tier)}</span>`
    + `<span class="ex-formula">${esc(bench.formula_latex || "")}</span></div>`
    + `<p class="ex-reason">${esc(bench.reason || "")}</p>`;

  if (runId) {
    html += `<div class="bench-fig"><img src="/api/runs/${esc(runId)}`
          + `/plots/analytic_comparison.png" alt="simulated against the closed form"`
          + ` loading="lazy"></div>`;
  }

  // The metric is named, not just the number: "relative to the curve's peak"
  // and "relative at each point" differ by orders of magnitude on a curve that
  // decays, and a reader who assumes the wrong one misreads the result.
  const metric = bench.metric === "scale"
    ? "relative to the curve's peak"
    : "pointwise relative";
  html += `<p class="ex-detail">Largest disagreement `
        + `<strong>${Number.isFinite(err) ? err.toExponential(2) : "—"}</strong> `
        + `(${metric}) over ${bench.n_points ?? "?"} points`
        + `${bench.n_series > 1 ? ` in ${bench.n_series} series` : ""}, `
        + `against a tolerance of ${Number.isFinite(tol) ? tol.toExponential(0) : "—"}.</p>`;

  if (bench.convergence) {
    html += `<p class="ex-detail ex-quiet">Tolerance from refinement: `
          + `${esc(bench.convergence)}</p>`;
  }
  if (bench.activity) {
    html += `<p class="ex-detail ex-quiet">Term is active: ${esc(bench.activity)}</p>`;
  }
  if (bench.caveat) {
    html += `<p class="ex-detail ex-caveat">${esc(bench.caveat)}</p>`;
  }
  box.innerHTML = html;
  host.appendChild(box);
  return bench;
}
