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
  kinetics: [
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
      title: "Solve strategy",
      hint: "time_march advances the kinetic equation to stop_tol and works on any geometry. steady_state hands the problem to the 0-D root find instead, which is far faster but has no cell axis — its state is f(E) with a single gap — so it needs a 1×1 mask. A multi-cell device reaches its steady state by time-marching; that is not a lesser answer, it is the same fixed point found by a different route.",
      fields: [
        F("strategy", "Strategy", "select", { options: ["time_march", "steady_state"] }),
        F("solver.method", "Root find (steady_state only)", "select", { options: ["auto", "picard", "coupled_newton"] }),
        F("solver.picard_tol", "Picard tol"),
        F("solver.picard_max_iter", "Picard max iter", "int"),
        F("solver.picard_mixing", "Picard mixing"),
        F("solver.anderson_depth", "Anderson depth", "int"),
        F("solver.newton_tol", "Newton tol"),
        F("solver.newton_max_iter", "Newton max iter", "int"),
      ],
    },
    {
      title: "Time stepping",
      hint: "Read under strategy = time_march. snapshot_interval records the field on the way; leave it empty to keep only the endpoint.",
      fields: [
        F("dt", "dt (ns)"),
        F("max_time", "Max time (ns)"),
        F("stop_tol", "Stop tol max|df/dt| (1/ns)"),
        F("snapshot_interval", "Snapshot interval (ns)", "number", { nullable: true }),
      ],
    },
    PROBE_FIELDS,
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
  mode: "kinetics",
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
  termStatus: {},       // term -> {state, reason}, from /api/terms
  // Why termStatus is empty, when it is. An empty map means "not known", which
  // the panel must SAY rather than draw as "no terms in this model".
  termError: "",
};

/* ---------- new-run view ---------- */

async function switchMode(mode, presetSetup) {
  // Entering the editor by any route leaves the read-only view behind.
  wizard.readOnly = false;
  wizard.title = "";
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
  if (!sections.length) {
    // Not "this dimensionality has no geometry" -- 0-D IS a geometry, a 1x1
    // mask, and the unified mode expresses it that way (measured
    // bit-identical to the old 0-D mode). What lands here is a LEGACY mode
    // that predates the unified geometry and carries no mask at all.
    form.innerHTML = `<p class="hint">This is a legacy mode with no geometry `
      + `of its own. In the unified mode a 0-D run is simply a 1&times;1 mask `
      + `and a 1-D strip a 1&times;N one, so the geometry step applies there.</p>`;
    return;
  }
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
    // Disabled rather than replaced by text: the control keeps its label,
    // units and formatting, so a value reads the same way it would be typed.
    if (wizard.readOnly) {
      grid.querySelectorAll("input, select, textarea, button").forEach((el) => {
        el.disabled = true;
      });
    }
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
    ${r.elapsed_s != null ? " · " + fmt(r.elapsed_s) + " s" : ""}</div>
    <div class="action-row"><button type="button" id="run-settings">View settings</button></div>`;
  for (const n of r.notes || []) html += `<div class="note">⚠ ${esc(n)}</div>`;
  if (r.error) html += `<div class="note">✗ ${esc(r.error)}</div>`;

  if (r.status === "done") {
    const entries = Object.entries(r.summary || {});
    if (entries.length) {
      html += `<table class="list summary-table">${entries.map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td>${esc(fmt(v))}</td></tr>`).join("")}</table>`;
    }
    const families = r.plot_params || {};
    html += `<div class="plot-grid">${(r.plots || [])
      .filter((p) => !families[p])
      .map((p) => `<img src="/api/runs/${esc(id)}/plots/${esc(p)}.png"`
        + ` alt="${esc(p)}" loading="lazy">`).join("")}</div>`;
    // A figure family is one image plus a control per index -- the run
    // replayed, rather than the single frame it happened to end on.
    for (const [name, params] of Object.entries(families)) {
      html += `<figure class="scrubber" data-plot="${esc(name)}" data-run="${esc(id)}">
        <img alt="${esc(name)}">
        <figcaption>${Object.entries(params).map(([p, n]) => `
          <label>${esc(p)}
            <input type="range" min="0" max="${n - 1}" value="0" data-param="${esc(p)}">
            <output>0 / ${n - 1}</output>
          </label>`).join("")}
          ${params.frame > 1
            ? `<button type="button" class="play" data-playing="0">▶ Play</button>`
            : ""}
        </figcaption>
      </figure>`;
    }
    html += `<div class="downloads">${(r.csvs || []).map((c) =>
      `<a class="btn" href="/api/runs/${esc(id)}/csv/${esc(c)}.csv">⬇ ${esc(c)}.csv</a>`).join("")}</div>`;
  }
  html += `<details><summary>Setup used</summary><pre class="json">${esc(JSON.stringify(r.setup, null, 2))}</pre></details>`;
  $("#run-detail").innerHTML = html;
  // A simulation is something that already ran. Its settings are a record
  // of what produced these numbers, so they are shown and never offered for
  // editing: an edited run would pair one set of results with a different
  // set of inputs, and for a test simulation it would also silently
  // invalidate the analytic comparison the case exists to make. Editing
  // belongs to SETUPS, which have not run yet.
  $("#run-settings")?.addEventListener("click", () => {
    openSettingsView(r.name, r.mode, r.setup);
  });
  document.querySelectorAll("#run-detail .scrubber").forEach(initScrubber);
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


/* ---------- figure families ----------
   The server renders one PNG per index, so scrubbing is just a src change.
   Frames are preloaded on open: without it the first pass through a run
   stutters while each frame round-trips, which reads as the simulation
   being jerky rather than the images arriving. */

function initScrubber(figure) {
  const img = figure.querySelector("img");
  const inputs = [...figure.querySelectorAll("input[type=range]")];
  const runId = figure.dataset.run;
  const plot = figure.dataset.plot;

  const url = () => {
    const q = inputs.map((i) => `${i.dataset.param}=${i.value}`).join("&");
    return `/api/runs/${encodeURIComponent(runId)}`
      + `/plots/${encodeURIComponent(plot)}.png?${q}`;
  };
  const draw = () => {
    for (const i of inputs) {
      i.nextElementSibling.textContent = `${i.value} / ${i.max}`;
    }
    img.src = url();
  };
  inputs.forEach((i) => i.addEventListener("input", draw));
  draw();

  const frame = inputs.find((i) => i.dataset.param === "frame");
  if (frame) {
    for (let k = 0; k <= Number(frame.max); k++) {
      const warm = new Image();
      const saved = frame.value;
      frame.value = String(k);
      warm.src = url();
      frame.value = saved;
    }
  }

  const play = figure.querySelector(".play");
  if (!play || !frame) return;
  let timer = null;
  const stop = () => {
    clearInterval(timer);
    timer = null;
    play.dataset.playing = "0";
    play.textContent = "▶ Play";
  };
  play.addEventListener("click", () => {
    if (timer !== null) { stop(); return; }
    play.dataset.playing = "1";
    play.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      const next = Number(frame.value) + 1;
      // Stop at the end rather than looping: a run has a beginning and an
      // end, and a silent wrap makes a monotone decay look periodic.
      if (next > Number(frame.max)) { stop(); return; }
      frame.value = String(next);
      draw();
    }, 320);
  });
  // A scrubber left playing in a torn-down view would keep firing.
  figure.addEventListener("qpsim:teardown", stop);
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

/* The settings a case would run with, resolved the same way running it
   resolves them -- defaults for the mode with the case's overrides applied --
   so what is shown is what would run, not a second description of it. */
async function viewCaseSettings(tc) {
  const { setup, unknown } = await buildCaseSetup(tc);
  openSettingsView(tc.title, tc.mode, setup, { from: "catalogue" });
  if (unknown.length) {
    // Into the wizard's own feedback slot, since that is the view now shown.
    $("#feedback").innerHTML =
      `<div class="err">This case names setup fields that do not exist: ${
        unknown.map(esc).join(", ")}. They are not reflected below.</div>`;
  }
}

/* `openCase` used to load a test case into the editor. It is deliberately
   gone rather than merely unreferenced: a test case's settings are chosen so
   that a particular closed form applies to them, so an edited copy keeps the
   analytic comparison while no longer being the case it describes. Editing
   belongs to setups. `viewCaseSettings` shows the same values, read-only. */

async function runCase(tc, btn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Generating…";
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
       <div class="tc-actions"></div>
       <div id="case-msg-${esc(tc.id)}"></div>`;
    // The claim is stated BEFORE the run, from the server's registry, so the
    // form shown here is the identical string the verdict is computed against.
    const declared = (tc.benchmark && state.benchmarks[tc.benchmark]) || tc.expect;
    if (tc.summary || declared) {
      // ONE disclosure holding everything that is not the title, the mode or
      // the buttons. Browsing a list of cases is choosing which to open, not
      // reading a paragraph and a formula per card; three of those stacked is
      // a wall of text before anything has been done.
      //
      // Collapsed, though, and not removed. The expectation in particular has
      // to stay STATED before the run: it is the identical string the verdict
      // is computed against, and revealing it only afterwards would let a
      // prediction be read in the light of its own result. A disclosure keeps
      // the pre-registration and gives back the page.
      //
      // The tier badge rides on the summary line rather than inside, because
      // it is the one thing here worth seeing without a click -- a T2 result
      // reuses the kernel it is checking, and that should not need expanding
      // to discover. The post-run verdict boxes are separate and unaffected.
      const stated = document.createElement("details");
      stated.className = "expectation";
      const tier = declared && declared.tier
        ? `<span class="ex-tier" title="${esc(TIER_NOTE[declared.tier] || "")}">`
          + `${esc(declared.tier)}</span>`
        : "";
      stated.innerHTML =
        `<summary class="ex-head">`
        + `<span class="ex-verdict">What this checks</span>${tier}</summary>`
        + (tc.summary ? `<p class="tc-summary">${esc(tc.summary)}</p>` : "")
        + (declared
            ? formulaBlock(
                tc.benchmark, declared.headline_latex || declared.formula_latex)
              + `<p class="ex-reason">${esc(declared.reason)}</p>`
            : "");
      card.insertBefore(stated, card.querySelector(".tc-actions"));
    }
    const actions = card.querySelector(".tc-actions");
    // A test simulation is a fixed experiment: its settings are chosen to make
    // a specific closed form apply, so editing them would leave the analytic
    // comparison attached to a case it no longer describes. The settings are
    // therefore viewable and not editable, and the only action is to run it.
    const view = document.createElement("button");
    view.textContent = "View settings";
    view.addEventListener("click", () => viewCaseSettings(tc));
    const run = document.createElement("button");
    run.className = "primary";
    run.textContent = "Generate & save";
    run.addEventListener("click", () => runCase(tc, run));
    actions.append(view, run);
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
      if (b.dataset.go === "physics") {
        openPhysics();
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
  await switchMode("kinetics");
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
/* What the client still needs to know about a term: where to WRITE when the
   user toggles it, and what to call it. Whether a term is in the model, and
   whether it is acting, is NOT decided here -- it comes from /api/terms, so
   there is one implementation of that question rather than two. See
   qpsim/webui/terms.py for why. */
const TERM_FIELDS = {
  // `governedBy` is the wizard page that decides whether the term EXISTS, as
  // opposed to the page that sets its value. Only transport has one: the mask
  // is what gives it faces to act across.
  diff:    { path: "material.D_0", kind: "zeroable", label: "Diffusion",
             governedBy: "geometry" },
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

/* The wizard builds kinetics, but the read-only view has to show any mode a
   saved run or a test case used -- several are 0-D, which has no geometry at
   all. Sections are therefore taken from the mode being shown rather than
   pinned to one, and the geometry step says so when a mode has none, instead
   of presenting an empty page as though something were missing. */
const isGeometrySection = (section) =>
  section.title === "Geometry" || section.title === "Boundary";
const geometrySections = (mode) => (FORMS[mode] || []).filter(isGeometrySection);
const conditionSections = (mode) =>
  (FORMS[mode] || []).filter((section) => !isGeometrySection(section));

const GEOMETRY_SECTIONS = geometrySections("kinetics");
const CONDITION_SECTIONS = conditionSections("kinetics");

/* Geometry first, because it decides part of the model rather than merely
   parameterising it: a single-cell mask has no faces, so there is no transport
   operator to switch on or off. Asking for the terms first offered a choice
   the geometry then overruled, and the panel had no way to say so. */
const WIZARD_STEPS = ["geometry", "equations", "conditions"];
// `readOnly` is what makes the settings view the wizard rather than a copy of
// it: the same renderers run, and only interaction is withdrawn.
const wizard = { index: 0, offD0: null, readOnly: false, title: "", from: "runs" };

/* Term state comes from the server, which derives it from the same gates the
   engine branches on. Three states, and the difference between the last two
   is the point: `on` means the model contains the term and it is acting,
   `off` means it contains it and it is switched off, `absent` means it is not
   in the model at all -- a single cell has no transport, a pinned phonon bath
   has no phonon equation, and a drive at zero photon number applies nothing.
   Before this, the panel worked all that out for itself and got three of them
   wrong while the numbers were right. */

/* "I have not been told" is NOT one of the three states, and must never be
   rendered as one. This defaulted to `absent` -- so a failed or not-yet-made
   request drew every term greyed, which reads as "this model contains no
   physics at all". The numbers were fine and the statement about them was
   false, which is the same failure the module was written to end. */
const UNKNOWN = "unknown";

async function refreshTerms() {
  if (state.setup === null) {
    state.termStatus = {};
    state.termError = "no setup is loaded, so the model cannot be read";
    return false;
  }
  const { ok, status, body } = await postJSON("/api/terms", {
    name: "term-status", setup: state.setup,
  });
  state.termStatus = ok ? body : {};
  state.termError = ok
    ? ""
    : `the server could not read this setup (HTTP ${status}), so which terms `
      + `it solves is unknown`;
  return ok;
}

function termsKnown() {
  return !!state.termStatus && Object.keys(state.termStatus).length > 0;
}

function termState(id) {
  const status = state.termStatus && state.termStatus[id];
  return status ? status.state : UNKNOWN;
}

function termIsOn(id) {
  return termState(id) === "on";
}

/* Unknown is not "applies": a term nobody has asked about must not be offered
   for switching, but it must not be drawn as absent either. */
function termApplies(id) {
  const s = termState(id);
  return s !== "absent" && s !== UNKNOWN;
}

function termReason(id) {
  const status = state.termStatus && state.termStatus[id];
  return (status && status.reason) || "";
}

function nudge(button) {
  button.classList.remove("nudge");
  void button.offsetWidth;              // restart the animation
  button.classList.add("nudge");
}

function clearTermNote() {
  const host = $("#term-note");
  if (host) host.innerHTML = "";
}

/* Why a term cannot be switched. The REASON is the server's -- there is one
   implementation of that question -- and only the "which page governs it"
   hint is the client's, because that is navigation rather than physics. */
function showTermNote(id) {
  const host = $("#term-note");
  if (!host) return;
  const spec = TERM_FIELDS[id];
  const label = spec ? spec.label : id;
  const reason = termReason(id)
    || (spec && spec.why)
    || "this term is not part of the model being solved";
  const step = spec && spec.governedBy;
  const jump = step && !wizard.readOnly
    ? ` <button type="button" class="link" data-goto-step="${step}">Change it</button>`
    : "";
  host.innerHTML = `<b>${esc(label)}</b> is not in this model: ${esc(reason)}.${jump}`;
  const button = host.querySelector("[data-goto-step]");
  if (button) {
    button.addEventListener("click", () =>
      showWizardStep(WIZARD_STEPS.indexOf(button.dataset.gotoStep)));
  }
}

function setTerm(id, on) {
  const spec = TERM_FIELDS[id];
  if (!spec || spec.path === null || !termApplies(id)) return;
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
  // Nothing is known about this setup, so the panel must assert nothing. Draw
  // every term plain and say why, rather than greying them all -- an all-grey
  // panel is a claim ("this model solves no physics"), and it would be false.
  if (!termsKnown()) {
    for (const button of document.querySelectorAll(".term[data-term]")) {
      button.classList.remove("off", "clamped");
      button.title = "";
    }
    for (const op of document.querySelectorAll(".op[data-op-for]")) {
      op.classList.remove("dim");
    }
    const host = $("#term-note");
    if (host) {
      host.innerHTML = `<b>Which terms this model solves is not known:</b> ${
        esc(state.termError || "the model has not been read yet")}.`;
    }
    $("#cons").innerHTML = "";
    return;
  }
  for (const button of document.querySelectorAll(".term[data-term]")) {
    const id = button.dataset.term;
    const spec = TERM_FIELDS[id];
    if (button.classList.contains("locked")) continue;
    const declared = spec && spec.path !== null;
    const expressible = termApplies(id);
    const unavailable = !expressible;
    const on = termIsOn(id);
    // Deliberately exclusive. "off" carries a strike-through, which asserts
    // the model contains this term and it is not acting; a term the mode
    // cannot express has no business making that claim.
    button.classList.toggle("off", expressible && !on);
    button.classList.toggle("clamped", unavailable);
    const label = spec ? spec.label : id;
    const reason = termReason(id);
    button.title = expressible
      ? `${label} — click to ${on ? "drop" : "restore"} it.`
        + (reason ? ` (${reason})` : "")
      : `${label}: ${reason || (declared ? "not in this model" : (spec && spec.why) || "not available")}.`;
  }
  // A sign belongs to the term it introduces and fades with it.
  for (const op of document.querySelectorAll(".op[data-op-for]")) {
    const target = document.querySelector(
      `.term[data-term="${op.dataset.opFor}"]`);
    op.classList.toggle("dim", target !== null && target.classList.contains("off"));
  }

  const live = Object.keys(TERM_FIELDS).filter((id) => termIsOn(id));
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
  // Nothing that would change or launch anything exists in the read-only
  // view; Back and Next remain, because paging is how you read three pages.
  for (const id of ["#btn-validate", "#btn-save-setup", "#btn-run"]) {
    $(id).classList.toggle("hidden", wizard.readOnly || !last);
  }
  $("#run-name").disabled = wizard.readOnly;
  document.body.classList.toggle("viewing-settings", wizard.readOnly);
  crumbsInto("#wizard-crumbs", wizard.readOnly
    ? [
        { label: "Home", go: () => showView("home") },
        wizard.from === "catalogue"
          ? { label: "Test simulations", go: () => openTestSimulations() }
          : { label: "Simulations", go: () => showView("runs") },
        { label: wizard.title || "Settings" },
      ]
    : [
        { label: "Home", go: () => showView("home") },
        { label: "New run" },
      ]);
  const mode = wizard.readOnly ? state.mode : "kinetics";
  // The note answers a click, so it must not outlive the page it was asked on
  // -- and the geometry may have changed in between, which is the whole point.
  clearTermNote();
  // Asked on EVERY page, not just the equations one. The geometry step changes
  // which terms exist, and moving it in front of the equations opened a window
  // where the panel could be reached before anything had been asked.
  refreshTerms().then(renderTermPanel);
  if (name === "geometry") {
    renderStepForm("#form-geometry", geometrySections(mode));
  }
  if (name === "conditions") {
    renderStepForm("#setup-form", conditionSections(mode));
  }
  window.scrollTo(0, 0);
}

/* The wizard, showing settings that already produced something and therefore
   cannot be changed. Same three pages and the same renderers -- a separate
   read-only layout would drift, and then the settings you inspect stop being
   the settings you would set. */
function openSettingsView(name, mode, setup, { from = "runs" } = {}) {
  wizard.readOnly = true;
  wizard.title = name;
  wizard.from = from;
  state.mode = mode;
  // A copy: the wizard's renderers read state.setup, and a stored run must
  // not change because somebody opened it to look.
  state.setup = JSON.parse(JSON.stringify(setup));
  $("#run-name").value = name;
  $("#feedback").innerHTML = "";
  showView("new-run");
  showWizardStep(0);
}

/* =====================================================================
   Physics reference
   The same equation panel the wizard uses, read rather than operated.
   Each entry says three separable things: what the term IS physically,
   how it is DISCRETISED, and what a user most often gets wrong about it.
   The last of those is not padding -- every gotcha listed here cost
   somebody a wrong run at some point.
   ===================================================================== */

const PHYSICS = {
  diff: {
    title: "Diffusion",
    sector: "Quasiparticles",
    formula: "∇·(D₀∇f)",
    physics: `Quasiparticles spread through the film. The flux carries the BCS
      density weight, so transport is energy-dependent: states just above the gap
      have a large density and move differently from states well above it. This is
      the only term that couples one cell of the device to another — collisions,
      the phonon sector and the gap closure are all strictly local.`,
    algorithm: `Finite volume on the device mask. Each face coefficient is the
      harmonic mean of its two cells, which is exact wherever the neighbours share
      a gap. Time stepping is Crank–Nicolson with automatic substepping bounded by
      a monotonicity limit, so a large step cannot produce a negative occupation.
      A separate operator is built for every energy bin, because the diffusion
      weight depends on E.`,
    gotchas: [
      "A single-cell device has no faces, so this term is ABSENT rather than off — there is nothing to transport between, whatever D₀ is.",
      "D₀ = 0 is the off switch: the flux coefficient is D₀·N₁^q, so zero gives an identically zero operator.",
      "The shipped D₀ for Al, Nb and TiN sit outside their own sourced bands; the material record says so on load.",
    ],
  },
  scat: {
    title: "Quasiparticle–phonon scattering",
    sector: "Quasiparticles",
    formula: "N₁ I_sc[f, n]",
    physics: `A quasiparticle absorbs or emits a phonon and moves in energy. The
      number of quasiparticles is unchanged — this term redistributes them, and it
      is what relaxes a driven distribution back toward the bath. Switch it off and
      nothing sets the energy distribution.`,
    algorithm: `A dense (N_E × N_E) kernel carrying the coherence factor
      K⁻ = 1 − Δ²/EE′ and the prefactor ω²/(τ₀(k_BT_c)³). Advanced with ETD2, a
      two-stage exponential integrator: the quasiparticle equation is quadratic in
      f (the Pauli factors depend on the quantity being stepped), so a
      predictor–corrector is needed for second order.`,
    gotchas: [
      "This is the QUASIPARTICLE side of the ledger. The same events appear in the phonon equation with a different prefactor — see the phonon scattering source.",
      "Switching it off leaves a model with no thermal fixed point, so detailed-balance and number-conservation certificates do not apply to the result.",
    ],
  },
  recomb: {
    title: "Recombination and pair breaking",
    sector: "Quasiparticles",
    formula: "N₁ I_r[f, n]",
    physics: `Two quasiparticles recombine into a Cooper pair and emit a phonon at
      ω = E + E′ ≥ 2Δ; the reverse absorbs such a phonon and breaks a pair. This is
      the number-changing channel — it is what fixes how many quasiparticles a
      driven device settles at.`,
    algorithm: `Same ETD2 stepper as scattering, with the coherence factor
      K⁺ = 1 + Δ²/EE′. The pair kinematics are what make this term numerically
      hard: at threshold both partners sit at the gap edge simultaneously, so the
      two density-of-states singularities coalesce.`,
    gotchas: [
      "With this off, an injected device has nothing to remove quasiparticles and there is no steady state to find.",
      "The threshold structure ω → 2Δ is where the discretisation is weakest; see the phonon recombination source.",
    ],
  },
  photsg: {
    title: "Sub-gap photon drive",
    sector: "Quasiparticles",
    formula: "N₁ I_γ,sg[f]",
    physics: `Photons below 2Δ cannot break a pair, but they move existing
      quasiparticles up and down in energy in steps of ω₀ — a ladder. This is the
      readout-power channel: it heats the distribution without directly creating
      quasiparticles.`,
    algorithm: `A ladder coupling bins separated by ω₀. Both the absorption and the
      emission terms carry (n̄ + 1), so the channel has a stimulated part
      proportional to the photon number and a spontaneous part that does not vanish
      with it.`,
    gotchas: [
      "n̄ = 0 does NOT switch this off. The (n̄+1) factor leaves spontaneous emission acting at full strength — measured, the drive still moves f by 19% at c_phot = 1e−4 and 80% at 1e−2.",
      "c_phot is the real off switch: every term is multiplied by it.",
      "ω₀ must be commensurate with the energy grid, or the ladder lands between bins.",
    ],
  },
  photpb: {
    title: "Pair-breaking photon drive",
    sector: "Quasiparticles",
    formula: "N₁ I_γ,pb[f]",
    physics: `A photon at ω ≥ 2Δ breaks a Cooper pair directly, creating two
      quasiparticles at E and ω − E. Unlike the sub-gap drive this changes the
      quasiparticle number.`,
    algorithm: `Couples each bin to its reflection partner about ω/2. Both partners
      must exist on the energy lattice, which is a stronger grid condition than the
      sub-gap ladder needs.`,
    gotchas: [
      "Needs the reflection partners on-lattice: 2·min_factor·N/(max−min) must be integral, or the setup is rejected.",
      "Like the sub-gap drive, c_phot_PB is the off switch rather than n̄_PB.",
    ],
  },
  src: {
    title: "External injection",
    sector: "Quasiparticles",
    formula: "N₁ S(E, r, t)",
    physics: `A prescribed source of quasiparticles — a tunnel junction, a
      normal-metal contact, an absorbed photon event. Whatever creates
      quasiparticles by a mechanism this model does not resolve.`,
    algorithm: `Added directly to the gain side of the collision step. The general
      form accepts an arbitrary g(E, x, y, t) as a compiled expression, so a source
      can be shaped in energy, in space and in time independently.`,
    gotchas: [
      "A source aimed at energies the grid does not represent is refused rather than silently discarded — injecting into a state that does not exist is undefined.",
      "On a gap-step device, bins below the local gap of the high-gap region are unrepresented there; the same nominal source is not the same source on both sides.",
    ],
  },
  psc: {
    title: "Phonon scattering source",
    sector: "Phonons",
    formula: "Σ_sc[f, n]",
    physics: `The other side of the ledger from quasiparticle scattering. Every
      scattering event that moves a quasiparticle in energy also creates or destroys
      a phonon, and this term books that into the phonon equation.`,
    algorithm: `Uses the phonon-side kernel 2K⁻/(πΔτ₀^PB) from F&C 2023 Eq. 12 —
      a genuinely different prefactor from the quasiparticle side, and τ₀^PB is
      about 1700× smaller than τ₀. The phonon equation is linear in n, so it is
      integrated exactly rather than by a multi-stage scheme.`,
    gotchas: [
      "This switch is independent of quasiparticle scattering. They are one physical process recorded on two sides, and whether BOTH sides are recorded is what these controls decide.",
      "Confusing τ₀ with τ₀^PB is a documented trap in this codebase and was a real shipped defect — the two differ by ~1700×.",
    ],
  },
  prc: {
    title: "Phonon recombination source",
    sector: "Phonons",
    formula: "Σ_r[f, n]",
    physics: `Phonons created by recombination and consumed by pair breaking. These
      are the ω ≥ 2Δ phonons that can break pairs again — the feedback loop behind
      phonon trapping.`,
    algorithm: `The pair integral at fixed ω runs along the line E + E′ = ω, so it
      has two density-of-states singularities that coalesce as ω → 2Δ. The exact
      value there is Kaplan's Δ·S₊(ω/Δ); a naive sum over an energy-aligned grid
      instead gives 4Δ, off by 4/π, and that error does not shrink under grid
      refinement.`,
    gotchas: [
      "This threshold structure is the numerically weakest point of the whole scheme, and the treatment of it is under active review.",
      "Recombination phonons and scattering phonons live on two different frequency lattices; a grid where they share no bins is now refused, because the two channels would evolve independently.",
    ],
  },
  pesc: {
    title: "Phonon escape",
    sector: "Phonons",
    formula: "(n − n_B(T_b)) / τ_esc",
    physics: `Phonons leak into the substrate at a finite rate, relaxing the phonon
      population toward the bath. A long escape time traps them in the film, where
      they break pairs again — which is why phonon trapping raises the steady-state
      quasiparticle density well above what the drive alone would give.`,
    algorithm: `A linear relaxation term. Together with the source terms it makes the
      phonon equation affine in n, dn/dt = a(f) + b(f)·n, which an exponential
      integrator solves EXACTLY at frozen f — no iteration and no truncation error
      in the phonon step itself.`,
    gotchas: [
      "The default phonon seed IS this term's own fixed point, so escape is unmeasurable until the initial population is moved away from the bath.",
      "τ_l = 0 is the τ → ∞ sentinel meaning NO substrate coupling, not instantaneous escape — the opposite of what the number suggests.",
    ],
  },
  gapeq: {
    title: "Self-consistent gap",
    sector: "Order parameter",
    formula: "1/(N₀V) = ∫ (1−2f)/√(E²−Δ²) dE",
    physics: `The order parameter is not a fixed parameter but a functional of the
      occupation. Quasiparticles occupying states above the gap suppress it, which
      moves the band edge, which changes every other term. Switching this off pins
      Δ at its material value.`,
    algorithm: `Brent's method, NOT Newton — a bracketing root find. The integration
      limit is the unknown and the integrand is singular exactly there, so a
      derivative-based step would need d/dΔ through the singularity. Out of
      equilibrium the residual can have several roots, so the solver carries the
      previous Δ as a reference and takes the nearest sign change. The outer
      self-consistency loop is under-relaxed Picard.`,
    gotchas: [
      "Even-multiplicity (tangent) roots produce no sign change and are invisible to a bracketing scan.",
      "It refuses to report Δ = 0 unless it has an analytic certificate: not finding a root is never treated as proof the material went normal.",
      "A self-consistent gap needs sub-gap grid cells, so min_factor must be below 1.",
    ],
  },
};

const PHYSICS_SOLVERS = [
  ["", "<b>Quasiparticles — ETD2.</b> The occupation equation is quadratic in f, "
     + "so its rates depend on the very quantity being advanced. A two-stage "
     + "predictor–corrector, with the linear relaxation handled exactly at each "
     + "stage, gives second order."],
  ["", "<b>Phonons — exponential Euler, and it is EXACT.</b> The phonon equation is "
     + "linear in n at frozen f, so a one-stage exponential step is the exact flow "
     + "of that sub-problem. A second stage could not improve on it."],
  ["", "<b>Composition.</b> Symmetric throughout, phonons included: "
     + "<span class=\"eq\">T(½dt) · P(½dt) · C(dt) · P(½dt) · T(½dt)</span> — "
     + "half a transport step, half a phonon advance, the quasiparticle step, "
     + "then the mirror image. Because both sub-flows are exact for their own "
     + "frozen-coefficient problem, ALL remaining time-integration error lives in "
     + "this composition rather than in either integrator."],
  ["derived", "That symmetry is the whole accuracy of the scheme, and it is fragile: "
     + "a step is only as accurate as its weakest factor, so advancing the phonons "
     + "ONCE per step rather than in two halves drops the entire scheme to first "
     + "order no matter how exact the pieces are. The spatial engine did exactly "
     + "that until it was corrected; measured on a driven 2×3 mesh, halving dt then "
     + "cut the error by 2.1× where the symmetric composition cuts it by 4.0×."],
  ["", "<b>Steady state.</b> Three routes: Newton on f with the phonons pinned at the "
     + "bath; Picard over n with an inner Newton on f, optionally Anderson-accelerated; "
     + "or a coupled Newton on (f, n) together. They should agree, and where they do "
     + "not, that disagreement is itself a measurement."],
  ["derived", "The gap is the exception: a bracketing root find, because its unknown "
     + "is an integration limit sitting on a singularity."],
];

function openPhysics() {
  crumbsInto("#physics-crumbs", [
    { label: "Home", go: () => showView("home") },
    { label: "Physics" },
  ]);
  const host = $("#physics-equation");
  host.innerHTML = "";
  // Cloned, never re-authored: the reader must be looking at the same equation
  // the wizard operates and the solver integrates.
  const source = document.querySelector("#step-equations .stack");
  if (source !== null) {
    const stack = source.cloneNode(true);
    stack.querySelectorAll(".term").forEach((button) => {
      button.classList.remove("off", "clamped", "nudge");
      const id = button.dataset.term;
      const entry = id && PHYSICS[id];
      button.disabled = false;
      if (entry) {
        button.title = `${entry.title} — click to read how it works.`;
        button.addEventListener("click", () => openPhysicsTerm(id));
      } else {
        // The left-hand sides carry no term of their own to describe.
        button.classList.add("locked");
        button.title = "The quantity being solved for.";
      }
    });
    host.appendChild(stack);
  }
  const solvers = $("#physics-solvers");
  solvers.innerHTML = "";
  for (const [cls, text] of PHYSICS_SOLVERS) {
    const li = document.createElement("li");
    li.className = cls;
    li.innerHTML = `<span class="mk"></span><span>${text}</span>`;
    solvers.appendChild(li);
  }
  showView("physics");
  window.scrollTo(0, 0);
}

function openPhysicsTerm(id) {
  const entry = PHYSICS[id];
  if (!entry) return openPhysics();
  crumbsInto("#physics-term-crumbs", [
    { label: "Home", go: () => showView("home") },
    { label: "Physics", go: () => openPhysics() },
    { label: entry.title },
  ]);
  const list = (items) =>
    items.map((g) => `<li><span class="mk"></span><span>${g}</span></li>`).join("");
  $("#physics-term-body").innerHTML = `
    <div class="home-head">
      <p class="eyebrow">${esc(entry.sector)}</p>
      <h1>${esc(entry.title)}</h1>
      <p class="physics-formula">${esc(entry.formula)}</p>
    </div>
    <section class="physics-section">
      <h2>What it is</h2>
      <p>${entry.physics}</p>
    </section>
    <section class="physics-section">
      <h2>How it is solved</h2>
      <p>${entry.algorithm}</p>
    </section>
    <section class="cons-wrap">
      <h2>Worth knowing</h2>
      <ul class="cons">${list(entry.gotchas)}</ul>
    </section>`;
  showView("physics-term");
  window.scrollTo(0, 0);
}

function initWizard() {
  for (const button of document.querySelectorAll(".term[data-term]")) {
    button.addEventListener("click", () => {
      if (document.body.classList.contains("editing")) return;
      // The settings of something that already ran are a record. Toggling a
      // term here would show a model that produced none of these numbers.
      if (wizard.readOnly) return;
      const id = button.dataset.term;
      if (button.classList.contains("locked")) {
        nudge(button);
        return;
      }
      // A term the model does not contain cannot be switched. Refusing
      // silently is what made a greyed term look like a live one: say which
      // term, and say what would have to change for it to exist.
      if (!termApplies(id)) {
        nudge(button);
        showTermNote(id);
        return;
      }
      if (!TERM_FIELDS[id] || TERM_FIELDS[id].path === null) return;
      clearTermNote();
      setTerm(id, !termIsOn(id));
      // Re-ask rather than assume: a toggle can change what OTHER terms are,
      // and the server is the one that knows. Switching the phonon sector to
      // a dynamic mode, for instance, brings three terms into the model.
      refreshTerms().then(renderTermPanel);
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

/* A formula is typeset by the server (matplotlib mathtext, the same engine
   behind every figure here) rather than shown as LaTeX source. `formula_latex`
   is the FULL statement -- several are align environments carrying the limit,
   the normalisation and the discretisation -- so it is not a banner; the
   headline is. If the image 404s the source is revealed instead, which is
   honest, where a half-typeset equation would not be. */
function formulaBlock(name, sourceLatex) {
  const src = name
    ? `/api/benchmarks/${encodeURIComponent(name)}/formula.png` : "";
  const fallback = sourceLatex
    ? `<code class="ex-source">${esc(sourceLatex)}</code>` : "";
  if (!src) return fallback;
  return `<span class="ex-formula">`
    + `<img class="formula" src="${src}" alt="${esc(sourceLatex || name)}"`
    + ` onerror="this.classList.add('failed');`
    + `this.nextElementSibling&&(this.nextElementSibling.hidden=false)">`
    + `<span class="ex-source" hidden>${esc(sourceLatex || "")}</span></span>`;
}

/* The full statement, kept reachable but out of the banner. */
function fullStatement(latex) {
  if (!latex) return "";
  return `<details class="ex-full"><summary>Full statement (LaTeX)</summary>`
    + `<pre class="ex-source">${esc(latex)}</pre></details>`;
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
    + formulaBlock(bench.name, bench.headline_latex || bench.formula_latex)
    + `</div>`
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
  html += fullStatement(bench.formula_latex);
  box.innerHTML = html;
  host.appendChild(box);
  return bench;
}
