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
  testsLoaded: false,   // the catalogue is static; fetch it once
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
  document.querySelectorAll("#mode-row button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  renderForm();
  $("#feedback").innerHTML = "";
}

function renderForm() {
  const form = $("#setup-form");
  form.innerHTML = "";
  for (const section of FORMS[state.mode]) {
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
  $("#materials-list").innerHTML =
    `<table class="list"><tr><th>Material</th>${cols.map(([, l]) => `<th>${esc(l)}</th>`).join("")}<th>Substrate</th></tr>` +
    state.materials.map((m) => `<tr><td><b>${esc(m.name)}</b></td>` +
      cols.map(([k]) => `<td>${m[k] == null ? "—" : esc(fmt(m[k]))}</td>`).join("") +
      `<td>${m.substrate ? esc(m.substrate.name) : "—"}</td></tr>`).join("") +
    `</table>
    <p class="hint">From the YAML database at qpsim/materials/data/. Picking a material in the
    setup editor copies these values; edits there don't touch the database.</p>`;
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
    const { resp, body } = await postJSON("/api/runs", { name: tc.title, setup });
    if (!resp.ok) {
      const problems = (body.errors || [String(resp.status)]).join("; ");
      throw new Error(problems);
    }
    showView("runs");
  } catch (exc) {
    btn.textContent = label;
    btn.disabled = false;
    const holder = document.getElementById(`case-msg-${tc.id}`);
    if (holder) holder.innerHTML = `<div class="err">Could not start: ${esc(exc.message)}</div>`;
  }
}

async function refreshTests() {
  const host = $("#tests-list");
  if (state.testsLoaded) return;
  let cat;
  try {
    const resp = await fetch("/static/test-simulations.json");
    if (!resp.ok) throw new Error(`catalogue unavailable (${resp.status})`);
    cat = await resp.json();
  } catch (exc) {
    host.innerHTML = `<div class="err">${esc(exc.message)}</div>`;
    return;
  }
  host.innerHTML = "";
  for (const group of cat.groups || []) {
    const sec = document.createElement("section");
    sec.className = "test-group";
    sec.innerHTML = `<h3>${esc(group.title)}</h3>` +
      (group.hint ? `<p class="hint">${esc(group.hint)}</p>` : "");
    for (const tc of group.cases || []) {
      const card = document.createElement("article");
      card.className = "test-case";
      card.innerHTML =
        `<div class="tc-head">
           <span class="tc-title">${esc(tc.title)}</span>
           <span class="tc-mode">${esc(state.modeLabels[tc.mode] || tc.mode)}</span>
         </div>
         <p class="tc-summary">${esc(tc.summary)}</p>
         <p class="tc-expect"><b>Expect</b> ${esc(tc.expect)}</p>
         <div class="tc-actions"></div>
         <div id="case-msg-${esc(tc.id)}"></div>`;
      const actions = card.querySelector(".tc-actions");
      const open = document.createElement("button");
      open.textContent = "Open in editor";
      open.addEventListener("click", () => openCase(tc));
      const run = document.createElement("button");
      run.className = "primary";
      run.textContent = "Run";
      run.addEventListener("click", () => runCase(tc, run));
      actions.append(open, run);
      sec.appendChild(card);
    }
    host.appendChild(sec);
  }
  state.testsLoaded = true;
}

function showView(name) {
  document.querySelectorAll("nav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name));
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
  } else if (name === "tests") {
    refreshTests();
  }
}

async function init() {
  const [{ body: meta }, { body: mats }] = await Promise.all([
    api("/api/meta"), api("/api/materials"),
  ]);
  state.modeLabels = meta.modes || {};
  state.materials = mats || [];
  $("#meta").textContent = `qpsim ${meta.qpsim_version} · workspace ${meta.workspace}`;
  $("#home-workspace").textContent = meta.workspace;

  const row = $("#mode-row");
  for (const [mode, label] of Object.entries(state.modeLabels)) {
    const b = document.createElement("button");
    b.dataset.mode = mode;
    b.textContent = label;
    b.addEventListener("click", () => switchMode(mode));
    row.appendChild(b);
  }
  document.querySelectorAll("nav button").forEach((b) =>
    b.addEventListener("click", () => showView(b.dataset.view)));
  // Home rows are shortcuts into the same views the nav reaches; rows with no
  // data-go are the not-yet-built ones and stay inert.
  document.querySelectorAll(".action[data-go]").forEach((b) =>
    b.addEventListener("click", () => {
      showView(b.dataset.go);
      window.scrollTo(0, 0);
    }));
  $("#btn-validate").addEventListener("click", doValidate);
  $("#btn-save-setup").addEventListener("click", doSaveSetup);
  $("#btn-run").addEventListener("click", doRun);

  await switchMode("steady_state_0d");
}

init();
