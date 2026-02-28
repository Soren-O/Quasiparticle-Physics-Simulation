from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from ..initial_conditions import (
    canonicalize_initial_condition,
    resolve_energy_spec,
    resolve_spatial_spec,
)
from ..models import BoundaryCondition, ExternalGenerationSpec, InitialConditionSpec
from .theme import FONT_MONO, RETRO_PANEL


# Literature values for quasiparticle diffusion in common superconductors.
# Each entry: (material, Tc_K, gap_ueV, D0_um2_per_ns, D0_range_str, references)
# D0 is the normal-state electron diffusion coefficient.
# gap is the single-particle gap Δ(0) in μeV.
MATERIAL_REFERENCE_TABLE: list[dict[str, Any]] = [
    {
        "material": "Aluminum (Al)",
        "Tc_K": 1.2,
        "gap_ueV": 180,
        "D0_nom": 6.0,
        "D0_range": "2\u201310",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 440.0,
        "refs": [
            ("Chi & Clarke, PRB 19 (1979)", "D \u2248 60 cm\u00b2/s in thin films"),
            ("Heikkil\u00e4 et al., arXiv:1911.02434", "D = 100 cm\u00b2/s (nanothermometry)"),
            ("Hubbell & Briscoe, PRL 20 (1968)", "D = 22.5 cm\u00b2/s (first measurement)"),
        ],
        "notes": "Widely used in transmon qubits. D\u2080 varies ~20\u2013100 cm\u00b2/s depending on film quality/thickness.",
    },
    {
        "material": "Niobium (Nb)",
        "Tc_K": 9.25,
        "gap_ueV": 1530,
        "D0_nom": 1.0,
        "D0_range": "0.5\u20132",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 0.15,
        "refs": [
            ("Kaplan et al., PRB 14 (1976)", "Recombination/scattering times; BCS parameters"),
            ("SQMS / Bal et al., PRApplied 20 (2023)", "QP spectroscopy in Nb transmon films"),
        ],
        "notes": "Strong-coupling superconductor (2\u0394/k_BT_c \u2248 3.8). Short mean free path in sputtered films gives low D\u2080.",
    },
    {
        "material": "Tantalum (Ta)",
        "Tc_K": 4.47,
        "gap_ueV": 700,
        "D0_nom": 0.82,
        "D0_range": "0.5\u20131.5",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 1.8,
        "refs": [
            ("Poelaert et al., PRB 61 (2000)", "D = 8.2 cm\u00b2/s, \u03c4 = 83 \u03bcs at 0.21 K (STJ)"),
        ],
        "notes": "Used in STJ X-ray detectors and emerging qubit platforms. Longer QP lifetime than Nb.",
    },
    {
        "material": "Tin (Sn)",
        "Tc_K": 3.72,
        "gap_ueV": 575,
        "D0_nom": 3.0,
        "D0_range": "2\u20134",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 2.3,
        "refs": [
            ("Kittel, Intro Solid State Physics", "\u0394 = 0.575 meV, standard BCS values"),
        ],
        "notes": "Classical low-T_c superconductor. Less commonly used in modern devices.",
    },
    {
        "material": "NbN",
        "Tc_K": 16.0,
        "gap_ueV": 2460,
        "D0_nom": 0.05,
        "D0_range": "0.02\u20130.1",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 0.02,
        "refs": [
            ("Il'in et al., 2019 (NbN thin films)", "D ~ 0.5 cm\u00b2/s, strongly disordered"),
        ],
        "notes": "Highly disordered; very short mean free path. Used in SNSPDs.",
    },
    {
        "material": "TiN",
        "Tc_K": 4.5,
        "gap_ueV": 700,
        "D0_nom": 0.1,
        "D0_range": "0.05\u20130.3",
        "D0_unit": "\u03bcm\u00b2/ns",
        "tau_0_ns": 1.5,
        "refs": [
            ("Leduc et al., APL 2010", "Used in KID detectors; disordered thin films"),
        ],
        "notes": "Tunable T_c via nitrogen content. Used in kinetic inductance detectors.",
    },
]


def show_material_reference(
    parent: tk.Misc,
    on_select: None | (callable) = None,
) -> None:
    """Show a reference table of superconductor material parameters.

    If *on_select* is provided it is called with (D0_nom, gap_ueV, Tc_K, tau_0_ns)
    when the user clicks 'Load' on a row.
    """
    window = tk.Toplevel(parent)
    window.title("Superconductor Material Reference")
    window.configure(bg=RETRO_PANEL)
    window.geometry("960x560")

    tk.Label(
        window,
        text="Measured quasiparticle diffusion parameters for common superconductors",
        bg=RETRO_PANEL,
        font=("Tahoma", 10, "bold"),
    ).pack(anchor="w", padx=10, pady=(10, 2))

    tk.Label(
        window,
        text=(
            "D\u2080 values vary significantly with film thickness, purity, and deposition method. "
            "Ranges below reflect the spread in the literature. "
            "Click a row and press 'Load Selected' to populate D\u2080 and \u0394 in the setup form."
        ),
        bg=RETRO_PANEL,
        wraplength=860,
        justify="left",
    ).pack(anchor="w", padx=10, pady=(0, 8))

    # Treeview table
    columns = ("material", "Tc", "gap", "D0_nom", "D0_range", "tau0", "notes")
    tree = ttk.Treeview(window, columns=columns, show="headings", height=len(MATERIAL_REFERENCE_TABLE))
    tree.heading("material", text="Material")
    tree.heading("Tc", text="T\u2081 (K)")
    tree.heading("gap", text="\u0394 (\u03bceV)")
    tree.heading("D0_nom", text="D\u2080 (\u03bcm\u00b2/ns)")
    tree.heading("D0_range", text="D\u2080 range")
    tree.heading("tau0", text="\u03c4\u2080 (ns)")
    tree.heading("notes", text="Notes")
    tree.column("material", width=110, minwidth=90)
    tree.column("Tc", width=55, minwidth=40, anchor="center")
    tree.column("gap", width=65, minwidth=50, anchor="center")
    tree.column("D0_nom", width=90, minwidth=70, anchor="center")
    tree.column("D0_range", width=80, minwidth=60, anchor="center")
    tree.column("tau0", width=70, minwidth=50, anchor="center")
    tree.column("notes", width=380, minwidth=200)

    for entry in MATERIAL_REFERENCE_TABLE:
        tree.insert("", "end", values=(
            entry["material"],
            f"{entry['Tc_K']:.2f}",
            str(entry["gap_ueV"]),
            f"{entry['D0_nom']}",
            entry["D0_range"],
            f"{entry['tau_0_ns']}",
            entry["notes"],
        ))

    tree.pack(fill="x", padx=10, pady=(0, 6))

    # References detail area
    ref_label = tk.Label(
        window,
        text="Select a material above to see literature references.",
        bg=RETRO_PANEL,
        justify="left",
        wraplength=860,
        font=FONT_MONO,
    )
    ref_label.pack(anchor="w", padx=10, fill="x")

    def on_tree_select(_event: Any = None) -> None:
        sel = tree.selection()
        if not sel:
            return
        idx = tree.index(sel[0])
        entry = MATERIAL_REFERENCE_TABLE[idx]
        lines = [f"References for {entry['material']}:"]
        for ref_text, detail in entry["refs"]:
            lines.append(f"  \u2022 {ref_text}: {detail}")
        ref_label.configure(text="\n".join(lines))

    tree.bind("<<TreeviewSelect>>", on_tree_select)

    # Buttons
    btn_frame = tk.Frame(window, bg=RETRO_PANEL)
    btn_frame.pack(fill="x", padx=10, pady=(10, 10))

    def do_load() -> None:
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a material row first.", parent=window)
            return
        idx = tree.index(sel[0])
        entry = MATERIAL_REFERENCE_TABLE[idx]
        if on_select is not None:
            on_select(entry["D0_nom"], entry["gap_ueV"], entry["Tc_K"], entry["tau_0_ns"])
        window.destroy()

    if on_select is not None:
        tk.Button(btn_frame, text="Load Selected", width=16, command=do_load).pack(side="left", padx=4)
    tk.Button(btn_frame, text="Close", width=12, command=window.destroy).pack(side="right", padx=4)

    window.transient(parent)
    window.grab_set()


_BOUNDARY_LABELS = {
    "reflective": "Insulated / Reflective (zero flux)",
    "neumann": "Neumann (non-zero flux gradient)",
    "dirichlet": "Dirichlet (fixed value)",
    "absorbing": "Absorbing",
    "robin": "Robin (mixed)",
}


def ask_boundary_condition(parent: tk.Misc, current: BoundaryCondition | None = None) -> BoundaryCondition | None:
    window = tk.Toplevel(parent)
    window.title("Assign Boundary Condition")
    window.configure(bg=RETRO_PANEL)
    window.resizable(False, False)

    result: list[BoundaryCondition | None] = [None]
    kind_var = tk.StringVar(value=(current.kind.lower() if current else "reflective"))
    value_var = tk.StringVar(value="" if current is None or current.value is None else str(current.value))
    aux_var = tk.StringVar(value="" if current is None or current.aux_value is None else str(current.aux_value))

    tk.Label(window, text="Type:", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
    kind_menu = ttk.Combobox(
        window,
        state="readonly",
        width=42,
        values=[f"{k} - {_BOUNDARY_LABELS[k]}" for k in _BOUNDARY_LABELS],
    )
    kind_menu.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
    kind_menu.set(f"{kind_var.get()} - {_BOUNDARY_LABELS[kind_var.get()]}")

    value_label = tk.Label(window, text="Value:", bg=RETRO_PANEL)
    value_label.grid(row=1, column=0, sticky="w", padx=8, pady=4)
    value_entry = tk.Entry(window, textvariable=value_var, width=20)
    value_entry.grid(row=1, column=1, sticky="w", padx=8, pady=4)

    aux_label = tk.Label(window, text="Aux Value (optional):", bg=RETRO_PANEL)
    aux_label.grid(row=2, column=0, sticky="w", padx=8, pady=4)
    aux_entry = tk.Entry(window, textvariable=aux_var, width=20)
    aux_entry.grid(row=2, column=1, sticky="w", padx=8, pady=4)

    hint = tk.Label(
        window,
        text="Neumann uses outward gradient. Robin uses (outward gradient + beta*u = gamma).",
        bg=RETRO_PANEL,
        justify="left",
        wraplength=420,
    )
    hint.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

    def update_fields() -> None:
        selected = kind_menu.get().split(" - ", 1)[0].strip().lower()
        kind_var.set(selected)
        needs_value = selected in {"neumann", "dirichlet", "robin"}
        needs_aux = selected == "robin"
        value_entry.configure(state=("normal" if needs_value else "disabled"))
        aux_entry.configure(state=("normal" if needs_aux else "disabled"))
        if not needs_value:
            value_var.set("")
        if not needs_aux:
            aux_var.set("")

    kind_menu.bind("<<ComboboxSelected>>", lambda _e: update_fields())
    update_fields()

    def on_cancel() -> None:
        result[0] = None
        window.destroy()

    def on_save() -> None:
        kind = kind_var.get().strip().lower()
        value = None
        aux = None
        if kind in {"neumann", "dirichlet", "robin"}:
            try:
                value = float(value_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid Value", "Numeric value is required for this boundary type.", parent=window)
                return
        if kind == "robin":
            raw_aux = aux_var.get().strip()
            if raw_aux:
                try:
                    aux = float(raw_aux)
                except ValueError:
                    messagebox.showerror("Invalid Aux Value", "Robin aux value must be numeric.", parent=window)
                    return
            else:
                aux = 0.0
        result[0] = BoundaryCondition(kind=kind, value=value, aux_value=aux)
        window.destroy()

    btn_row = tk.Frame(window, bg=RETRO_PANEL)
    btn_row.grid(row=4, column=0, columnspan=2, sticky="e", padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Cancel", command=on_cancel, width=12).pack(side="right", padx=4)
    tk.Button(btn_row, text="Assign", command=on_save, width=12).pack(side="right", padx=4)

    window.transient(parent)
    window.grab_set()
    window.wait_window()
    return result[0]


def ask_initial_condition(
    parent: tk.Misc,
    current: InitialConditionSpec,
) -> InitialConditionSpec | None:
    window = tk.Toplevel(parent)
    window.title("Initial Conditions")
    window.configure(bg=RETRO_PANEL)
    window.geometry("820x700")

    result: list[InitialConditionSpec | None] = [None]
    canonical_current = canonicalize_initial_condition(current)
    spatial_kind, spatial_params, spatial_custom_body, spatial_custom_params = resolve_spatial_spec(
        canonical_current
    )
    energy_kind, energy_params, energy_custom_body, energy_custom_params = resolve_energy_spec(canonical_current)
    if spatial_kind not in {"gaussian", "uniform", "point", "custom"}:
        spatial_kind = "gaussian"
    if energy_kind not in {"dos", "fermi_dirac", "uniform", "custom"}:
        energy_kind = "dos"

    spatial_kind_var = tk.StringVar(value=spatial_kind or "gaussian")
    energy_kind_var = tk.StringVar(value=energy_kind or "dos")

    gaussian_amp = tk.StringVar(value=str(spatial_params.get("amplitude", 1.0)))
    gaussian_x0 = tk.StringVar(value=str(spatial_params.get("x0", 0.5)))
    gaussian_y0 = tk.StringVar(value=str(spatial_params.get("y0", 0.5)))
    gaussian_sigma = tk.StringVar(value=str(spatial_params.get("sigma", 0.12)))

    uniform_value = tk.StringVar(value=str(spatial_params.get("value", 1.0)))
    point_value = tk.StringVar(value=str(spatial_params.get("value", 1.0)))
    point_x0 = tk.StringVar(value=str(spatial_params.get("x0", 0.5)))
    point_y0 = tk.StringVar(value=str(spatial_params.get("y0", 0.5)))

    spatial_custom_params_var = tk.StringVar(value=json.dumps(spatial_custom_params or {}))
    energy_custom_params_var = tk.StringVar(value=json.dumps(energy_custom_params or {}))
    energy_fd_temp_var = tk.StringVar(value=str(energy_params.get("temperature", 0.1)))
    energy_uniform_value_var = tk.StringVar(value=str(energy_params.get("value", 1.0)))

    tk.Label(window, text="Configure spatial and energy initial profiles independently.", bg=RETRO_PANEL).pack(
        anchor="w", padx=10, pady=(10, 8)
    )

    top_row = tk.Frame(window, bg=RETRO_PANEL)
    top_row.pack(fill="x", padx=10, pady=(0, 8))
    tk.Label(top_row, text="Spatial Profile:", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w", padx=(0, 6))
    spatial_menu = ttk.Combobox(top_row, state="readonly", width=18, values=["gaussian", "uniform", "point", "custom"])
    spatial_menu.set(spatial_kind_var.get())
    spatial_menu.grid(row=0, column=1, sticky="w", padx=(0, 14))
    tk.Label(top_row, text="Energy Profile:", bg=RETRO_PANEL).grid(row=0, column=2, sticky="w", padx=(0, 6))
    energy_menu = ttk.Combobox(top_row, state="readonly", width=18, values=["dos", "fermi_dirac", "uniform", "custom"])
    energy_menu.set(energy_kind_var.get())
    energy_menu.grid(row=0, column=3, sticky="w")

    container = tk.Frame(window, bg=RETRO_PANEL)
    container.pack(fill="both", expand=True, padx=10, pady=6)
    spatial_box = tk.LabelFrame(container, text="Spatial Profile", bg=RETRO_PANEL)
    spatial_box.pack(fill="x", pady=(0, 8))
    energy_box = tk.LabelFrame(container, text="Energy Profile", bg=RETRO_PANEL)
    energy_box.pack(fill="both", expand=True)

    spatial_frames: dict[str, tk.Frame] = {}
    energy_frames: dict[str, tk.Frame] = {}

    gauss_frame = tk.Frame(spatial_box, bg=RETRO_PANEL)
    spatial_frames["gaussian"] = gauss_frame
    tk.Label(gauss_frame, text="amplitude", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_amp, width=12).grid(row=0, column=1, sticky="w")
    tk.Label(gauss_frame, text="x0 (0..1)", bg=RETRO_PANEL).grid(row=1, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_x0, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(gauss_frame, text="y0 (0..1)", bg=RETRO_PANEL).grid(row=2, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_y0, width=12).grid(row=2, column=1, sticky="w")
    tk.Label(gauss_frame, text="sigma", bg=RETRO_PANEL).grid(row=3, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_sigma, width=12).grid(row=3, column=1, sticky="w")

    uniform_frame = tk.Frame(spatial_box, bg=RETRO_PANEL)
    spatial_frames["uniform"] = uniform_frame
    tk.Label(uniform_frame, text="value", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(uniform_frame, textvariable=uniform_value, width=12).grid(row=0, column=1, sticky="w")

    point_frame = tk.Frame(spatial_box, bg=RETRO_PANEL)
    spatial_frames["point"] = point_frame
    tk.Label(point_frame, text="value", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_value, width=12).grid(row=0, column=1, sticky="w")
    tk.Label(point_frame, text="x0 (0..1)", bg=RETRO_PANEL).grid(row=1, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_x0, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(point_frame, text="y0 (0..1)", bg=RETRO_PANEL).grid(row=2, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_y0, width=12).grid(row=2, column=1, sticky="w")

    spatial_custom_frame = tk.Frame(spatial_box, bg=RETRO_PANEL)
    spatial_frames["custom"] = spatial_custom_frame
    tk.Label(spatial_custom_frame, text="def user_expression(x, y, params):", bg=RETRO_PANEL, font=FONT_MONO).pack(anchor="w")
    spatial_custom_text = tk.Text(spatial_custom_frame, width=94, height=8, font=FONT_MONO)
    spatial_custom_text.pack(fill="x", expand=False)
    spatial_custom_text.insert("1.0", spatial_custom_body)
    tk.Label(spatial_custom_frame, text="Custom params (JSON dict):", bg=RETRO_PANEL).pack(anchor="w", pady=(6, 2))
    tk.Entry(spatial_custom_frame, textvariable=spatial_custom_params_var, width=94).pack(anchor="w")

    dos_frame = tk.Frame(energy_box, bg=RETRO_PANEL)
    energy_frames["dos"] = dos_frame
    tk.Label(dos_frame, text="Use default DOS-based energy weighting.", bg=RETRO_PANEL).pack(anchor="w")

    fd_frame = tk.Frame(energy_box, bg=RETRO_PANEL)
    energy_frames["fermi_dirac"] = fd_frame
    tk.Label(fd_frame, text="temperature (K)", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(fd_frame, textvariable=energy_fd_temp_var, width=12).grid(row=0, column=1, sticky="w")
    tk.Label(
        fd_frame,
        text="Thermal profile: n(E) proportional to N_s(E) * f(E,T).",
        bg=RETRO_PANEL,
        justify="left",
        font=FONT_MONO,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    energy_uniform_frame = tk.Frame(energy_box, bg=RETRO_PANEL)
    energy_frames["uniform"] = energy_uniform_frame
    tk.Label(energy_uniform_frame, text="value (relative weight)", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(energy_uniform_frame, textvariable=energy_uniform_value_var, width=12).grid(row=0, column=1, sticky="w")

    energy_custom_frame = tk.Frame(energy_box, bg=RETRO_PANEL)
    energy_frames["custom"] = energy_custom_frame
    tk.Label(energy_custom_frame, text="def energy_profile(E, gap, params):", bg=RETRO_PANEL, font=FONT_MONO).pack(anchor="w")
    energy_custom_text = tk.Text(energy_custom_frame, width=94, height=8, font=FONT_MONO)
    energy_custom_text.pack(fill="x", expand=False)
    energy_custom_text.insert("1.0", energy_custom_body)
    tk.Label(energy_custom_frame, text="Custom params (JSON dict):", bg=RETRO_PANEL).pack(anchor="w", pady=(6, 2))
    tk.Entry(energy_custom_frame, textvariable=energy_custom_params_var, width=94).pack(anchor="w")

    def show_spatial_frame(kind: str) -> None:
        for frame in spatial_frames.values():
            frame.pack_forget()
        spatial_frames[kind].pack(fill="x", expand=False, anchor="w", padx=6, pady=6)

    def show_energy_frame(kind: str) -> None:
        for frame in energy_frames.values():
            frame.pack_forget()
        energy_frames[kind].pack(fill="x", expand=False, anchor="w", padx=6, pady=6)

    def on_spatial_change(*_args) -> None:
        kind = spatial_menu.get().strip().lower()
        spatial_kind_var.set(kind)
        show_spatial_frame(kind)

    def on_energy_change(*_args) -> None:
        kind = energy_menu.get().strip().lower()
        energy_kind_var.set(kind)
        show_energy_frame(kind)

    spatial_menu.bind("<<ComboboxSelected>>", on_spatial_change)
    energy_menu.bind("<<ComboboxSelected>>", on_energy_change)
    show_spatial_frame(spatial_kind_var.get())
    show_energy_frame(energy_kind_var.get())

    def on_cancel() -> None:
        result[0] = None
        window.destroy()

    def parse_float(name: str, raw: str) -> float:
        try:
            return float(raw.strip())
        except Exception:
            raise ValueError(f"'{name}' must be numeric.")

    def on_apply() -> None:
        spatial_kind_val = spatial_kind_var.get().strip().lower()
        energy_kind_val = energy_kind_var.get().strip().lower()
        try:
            if spatial_kind_val == "gaussian":
                spatial_params_val = {
                    "amplitude": parse_float("amplitude", gaussian_amp.get()),
                    "x0": parse_float("x0", gaussian_x0.get()),
                    "y0": parse_float("y0", gaussian_y0.get()),
                    "sigma": parse_float("sigma", gaussian_sigma.get()),
                }
                spatial_custom_body_val = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)"
                spatial_custom_params_val: dict[str, Any] = {}
            elif spatial_kind_val == "uniform":
                spatial_params_val = {"value": parse_float("value", uniform_value.get())}
                spatial_custom_body_val = "return 1.0"
                spatial_custom_params_val = {}
            elif spatial_kind_val == "point":
                spatial_params_val = {
                    "value": parse_float("value", point_value.get()),
                    "x0": parse_float("x0", point_x0.get()),
                    "y0": parse_float("y0", point_y0.get()),
                }
                spatial_custom_body_val = "return 0.0"
                spatial_custom_params_val = {}
            elif spatial_kind_val == "custom":
                raw_spatial_params = spatial_custom_params_var.get().strip()
                spatial_custom_params_val = {}
                if raw_spatial_params:
                    spatial_custom_params_val = json.loads(raw_spatial_params)
                    if not isinstance(spatial_custom_params_val, dict):
                        raise ValueError("Spatial custom params must be a JSON object.")
                spatial_params_val = {}
                spatial_custom_body_val = spatial_custom_text.get("1.0", "end").strip()
            else:
                raise ValueError(f"Unsupported spatial initial condition type: {spatial_kind_val}")

            if energy_kind_val == "dos":
                energy_params_val: dict[str, Any] = {}
                energy_custom_body_val = "return np.ones_like(E)"
                energy_custom_params_val: dict[str, Any] = {}
            elif energy_kind_val == "fermi_dirac":
                energy_params_val = {"temperature": parse_float("temperature", energy_fd_temp_var.get())}
                energy_custom_body_val = "return np.ones_like(E)"
                energy_custom_params_val = {}
            elif energy_kind_val == "uniform":
                energy_params_val = {"value": parse_float("value", energy_uniform_value_var.get())}
                energy_custom_body_val = "return np.ones_like(E)"
                energy_custom_params_val = {}
            elif energy_kind_val == "custom":
                raw_energy_params = energy_custom_params_var.get().strip()
                energy_custom_params_val = {}
                if raw_energy_params:
                    energy_custom_params_val = json.loads(raw_energy_params)
                    if not isinstance(energy_custom_params_val, dict):
                        raise ValueError("Energy custom params must be a JSON object.")
                energy_params_val = {}
                energy_custom_body_val = energy_custom_text.get("1.0", "end").strip()
            else:
                raise ValueError(f"Unsupported energy initial condition type: {energy_kind_val}")

            spec = InitialConditionSpec(
                kind=spatial_kind_val,
                params=spatial_params_val,
                custom_body=spatial_custom_body_val,
                custom_params=spatial_custom_params_val,
                spatial_kind=spatial_kind_val,
                spatial_params=spatial_params_val,
                spatial_custom_body=spatial_custom_body_val,
                spatial_custom_params=spatial_custom_params_val,
                energy_kind=energy_kind_val,
                energy_params=energy_params_val,
                energy_custom_body=energy_custom_body_val,
                energy_custom_params=energy_custom_params_val,
            )
        except Exception as exc:
            messagebox.showerror("Invalid Initial Condition", str(exc), parent=window)
            return
        result[0] = canonicalize_initial_condition(spec)
        window.destroy()

    controls = tk.Frame(window, bg=RETRO_PANEL)
    controls.pack(fill="x", padx=10, pady=(4, 10))
    tk.Button(controls, text="Cancel", width=14, command=on_cancel).pack(side="right", padx=4)
    tk.Button(controls, text="Apply", width=14, command=on_apply).pack(side="right", padx=4)

    window.transient(parent)
    window.grab_set()
    window.wait_window()
    return result[0]


def ask_external_generation(
    parent: tk.Misc,
    current: ExternalGenerationSpec,
) -> ExternalGenerationSpec | None:
    """Dialog for configuring external quasiparticle generation."""
    window = tk.Toplevel(parent)
    window.title("External Generation")
    window.configure(bg=RETRO_PANEL)
    window.geometry("640x480")

    result: list[ExternalGenerationSpec | None] = [None]
    mode_var = tk.StringVar(value=current.mode)
    rate_var = tk.StringVar(value=str(current.rate))
    pulse_start_var = tk.StringVar(value=str(current.pulse_start))
    pulse_duration_var = tk.StringVar(value=str(current.pulse_duration))
    pulse_rate_var = tk.StringVar(value=str(current.pulse_rate))
    custom_params_var = tk.StringVar(value=json.dumps(current.custom_params or {}))

    tk.Label(window, text="Mode:", bg=RETRO_PANEL).pack(anchor="w", padx=10, pady=(10, 4))
    mode_menu = ttk.Combobox(
        window, state="readonly", width=20,
        values=["none", "constant", "pulse", "custom"],
    )
    mode_menu.set(mode_var.get())
    mode_menu.pack(anchor="w", padx=10, pady=(0, 8))

    container = tk.Frame(window, bg=RETRO_PANEL)
    container.pack(fill="both", expand=True, padx=10, pady=6)

    frames: dict[str, tk.Frame] = {}

    # None frame
    none_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["none"] = none_frame
    tk.Label(none_frame, text="No external generation.", bg=RETRO_PANEL).pack(anchor="w")

    # Constant frame
    const_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["constant"] = const_frame
    tk.Label(const_frame, text="Rate (\u03bceV\u207b\u00b9 \u03bcm\u207b\u00b2 ns\u207b\u00b9)", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(const_frame, textvariable=rate_var, width=14).grid(row=0, column=1, sticky="w")

    # Pulse frame
    pulse_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["pulse"] = pulse_frame
    tk.Label(pulse_frame, text="Pulse Rate", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(pulse_frame, textvariable=pulse_rate_var, width=14).grid(row=0, column=1, sticky="w")
    tk.Label(pulse_frame, text="Start Time (ns)", bg=RETRO_PANEL).grid(row=1, column=0, sticky="w")
    tk.Entry(pulse_frame, textvariable=pulse_start_var, width=14).grid(row=1, column=1, sticky="w")
    tk.Label(pulse_frame, text="Duration (ns)", bg=RETRO_PANEL).grid(row=2, column=0, sticky="w")
    tk.Entry(pulse_frame, textvariable=pulse_duration_var, width=14).grid(row=2, column=1, sticky="w")

    # Custom frame
    custom_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["custom"] = custom_frame
    tk.Label(custom_frame, text="def _g_ext(E, x, y, t, params):", bg=RETRO_PANEL, font=FONT_MONO).pack(anchor="w")
    tk.Label(custom_frame, text="    # E in \u03bceV, x/y normalized 0..1, t in ns", bg=RETRO_PANEL, font=FONT_MONO).pack(anchor="w")
    tk.Label(custom_frame, text="Body:", bg=RETRO_PANEL).pack(anchor="w", pady=(8, 2))
    custom_text = tk.Text(custom_frame, width=70, height=8, font=FONT_MONO)
    custom_text.pack(fill="x", expand=False)
    custom_text.insert("1.0", current.custom_body or "return 0.0")
    tk.Label(custom_frame, text="Custom params (JSON dict):", bg=RETRO_PANEL).pack(anchor="w", pady=(8, 2))
    tk.Entry(custom_frame, textvariable=custom_params_var, width=70).pack(anchor="w")

    def show_frame(mode: str) -> None:
        for f in frames.values():
            f.pack_forget()
        frames.get(mode, frames["none"]).pack(fill="both", expand=True, anchor="w")

    def on_mode_change(*_args) -> None:
        mode_var.set(mode_menu.get().strip().lower())
        show_frame(mode_var.get())

    mode_menu.bind("<<ComboboxSelected>>", on_mode_change)
    show_frame(mode_var.get())

    def parse_float(name: str, raw: str) -> float:
        try:
            return float(raw.strip())
        except Exception:
            raise ValueError(f"'{name}' must be numeric.")

    def on_cancel() -> None:
        result[0] = None
        window.destroy()

    def on_apply() -> None:
        mode = mode_var.get().strip().lower()
        try:
            if mode == "none":
                spec = ExternalGenerationSpec(mode="none")
            elif mode == "constant":
                spec = ExternalGenerationSpec(
                    mode="constant",
                    rate=parse_float("rate", rate_var.get()),
                )
            elif mode == "pulse":
                spec = ExternalGenerationSpec(
                    mode="pulse",
                    pulse_rate=parse_float("pulse rate", pulse_rate_var.get()),
                    pulse_start=parse_float("start time", pulse_start_var.get()),
                    pulse_duration=parse_float("duration", pulse_duration_var.get()),
                )
            elif mode == "custom":
                raw_params = custom_params_var.get().strip()
                params = {}
                if raw_params:
                    params = json.loads(raw_params)
                    if not isinstance(params, dict):
                        raise ValueError("Custom params must be a JSON object.")
                spec = ExternalGenerationSpec(
                    mode="custom",
                    custom_body=custom_text.get("1.0", "end").strip(),
                    custom_params=params,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc), parent=window)
            return
        result[0] = spec
        window.destroy()

    gen_controls = tk.Frame(window, bg=RETRO_PANEL)
    gen_controls.pack(fill="x", padx=10, pady=(4, 10))
    tk.Button(gen_controls, text="Cancel", width=14, command=on_cancel).pack(side="right", padx=4)
    tk.Button(gen_controls, text="Apply", width=14, command=on_apply).pack(side="right", padx=4)

    window.transient(parent)
    window.grab_set()
    window.wait_window()
    return result[0]
