from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk

from ..models import BoundaryCondition, InitialConditionSpec
from .theme import FONT_MONO, RETRO_PANEL


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
    window.geometry("700x560")

    result: list[InitialConditionSpec | None] = [None]
    kind_var = tk.StringVar(value=current.kind.lower())

    gaussian_amp = tk.StringVar(value=str(current.params.get("amplitude", 1.0)))
    gaussian_x0 = tk.StringVar(value=str(current.params.get("x0", 0.5)))
    gaussian_y0 = tk.StringVar(value=str(current.params.get("y0", 0.5)))
    gaussian_sigma = tk.StringVar(value=str(current.params.get("sigma", 0.12)))

    uniform_value = tk.StringVar(value=str(current.params.get("value", 1.0)))
    point_value = tk.StringVar(value=str(current.params.get("value", 1.0)))
    point_x0 = tk.StringVar(value=str(current.params.get("x0", 0.5)))
    point_y0 = tk.StringVar(value=str(current.params.get("y0", 0.5)))

    custom_params_var = tk.StringVar(value=json.dumps(current.custom_params or {}))

    tk.Label(window, text="Type:", bg=RETRO_PANEL).pack(anchor="w", padx=10, pady=(10, 4))
    kind_menu = ttk.Combobox(window, state="readonly", width=28, values=["gaussian", "uniform", "point", "custom"])
    kind_menu.set(kind_var.get())
    kind_menu.pack(anchor="w", padx=10, pady=(0, 8))

    container = tk.Frame(window, bg=RETRO_PANEL)
    container.pack(fill="both", expand=True, padx=10, pady=6)

    frames: dict[str, tk.Frame] = {}

    gauss_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["gaussian"] = gauss_frame
    tk.Label(gauss_frame, text="amplitude", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_amp, width=12).grid(row=0, column=1, sticky="w")
    tk.Label(gauss_frame, text="x0 (0..1)", bg=RETRO_PANEL).grid(row=1, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_x0, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(gauss_frame, text="y0 (0..1)", bg=RETRO_PANEL).grid(row=2, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_y0, width=12).grid(row=2, column=1, sticky="w")
    tk.Label(gauss_frame, text="sigma", bg=RETRO_PANEL).grid(row=3, column=0, sticky="w")
    tk.Entry(gauss_frame, textvariable=gaussian_sigma, width=12).grid(row=3, column=1, sticky="w")

    uniform_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["uniform"] = uniform_frame
    tk.Label(uniform_frame, text="value", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(uniform_frame, textvariable=uniform_value, width=12).grid(row=0, column=1, sticky="w")

    point_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["point"] = point_frame
    tk.Label(point_frame, text="value", bg=RETRO_PANEL).grid(row=0, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_value, width=12).grid(row=0, column=1, sticky="w")
    tk.Label(point_frame, text="x0 (0..1)", bg=RETRO_PANEL).grid(row=1, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_x0, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(point_frame, text="y0 (0..1)", bg=RETRO_PANEL).grid(row=2, column=0, sticky="w")
    tk.Entry(point_frame, textvariable=point_y0, width=12).grid(row=2, column=1, sticky="w")

    custom_frame = tk.Frame(container, bg=RETRO_PANEL)
    frames["custom"] = custom_frame
    tk.Label(
        custom_frame,
        text="Fixed scaffold (not editable):",
        bg=RETRO_PANEL,
    ).pack(anchor="w")
    tk.Label(
        custom_frame,
        text="def user_expression(x, y, params):",
        bg=RETRO_PANEL,
        justify="left",
        font=FONT_MONO,
    ).pack(anchor="w")
    tk.Label(
        custom_frame,
        text="    # x and y are normalized coordinates in [0, 1]",
        bg=RETRO_PANEL,
        justify="left",
        font=FONT_MONO,
    ).pack(anchor="w")
    tk.Label(
        custom_frame,
        text="Editable body:",
        bg=RETRO_PANEL,
    ).pack(anchor="w", pady=(8, 2))
    custom_text = tk.Text(custom_frame, width=78, height=12, font=FONT_MONO)
    custom_text.pack(fill="x", expand=False)
    custom_text.insert("1.0", current.custom_body or "return 0.0")

    tk.Label(custom_frame, text="Custom params (JSON dict):", bg=RETRO_PANEL).pack(anchor="w", pady=(8, 2))
    tk.Entry(custom_frame, textvariable=custom_params_var, width=78).pack(anchor="w")

    def show_frame(kind: str) -> None:
        for frame in frames.values():
            frame.pack_forget()
        frames[kind].pack(fill="both", expand=True, anchor="w")

    def on_type_change(*_args) -> None:
        kind = kind_menu.get().strip().lower()
        kind_var.set(kind)
        show_frame(kind)

    kind_menu.bind("<<ComboboxSelected>>", on_type_change)
    show_frame(kind_var.get())

    def on_cancel() -> None:
        result[0] = None
        window.destroy()

    def parse_float(name: str, raw: str) -> float:
        try:
            return float(raw.strip())
        except Exception:
            raise ValueError(f"'{name}' must be numeric.")

    def on_apply() -> None:
        kind = kind_var.get().strip().lower()
        try:
            if kind == "gaussian":
                spec = InitialConditionSpec(
                    kind="gaussian",
                    params={
                        "amplitude": parse_float("amplitude", gaussian_amp.get()),
                        "x0": parse_float("x0", gaussian_x0.get()),
                        "y0": parse_float("y0", gaussian_y0.get()),
                        "sigma": parse_float("sigma", gaussian_sigma.get()),
                    },
                )
            elif kind == "uniform":
                spec = InitialConditionSpec(
                    kind="uniform",
                    params={"value": parse_float("value", uniform_value.get())},
                )
            elif kind == "point":
                spec = InitialConditionSpec(
                    kind="point",
                    params={
                        "value": parse_float("value", point_value.get()),
                        "x0": parse_float("x0", point_x0.get()),
                        "y0": parse_float("y0", point_y0.get()),
                    },
                )
            elif kind == "custom":
                raw_params = custom_params_var.get().strip()
                custom_params = {}
                if raw_params:
                    custom_params = json.loads(raw_params)
                    if not isinstance(custom_params, dict):
                        raise ValueError("Custom params must be a JSON object.")
                spec = InitialConditionSpec(
                    kind="custom",
                    params={},
                    custom_body=custom_text.get("1.0", "end").strip(),
                    custom_params=custom_params,
                )
            else:
                raise ValueError(f"Unsupported initial condition type: {kind}")
        except Exception as exc:
            messagebox.showerror("Invalid Initial Condition", str(exc), parent=window)
            return
        result[0] = spec
        window.destroy()

    controls = tk.Frame(window, bg=RETRO_PANEL)
    controls.pack(fill="x", padx=10, pady=(4, 10))
    tk.Button(controls, text="Cancel", width=14, command=on_cancel).pack(side="right", padx=4)
    tk.Button(controls, text="Apply", width=14, command=on_apply).pack(side="right", padx=4)

    window.transient(parent)
    window.grab_set()
    window.wait_window()
    return result[0]

