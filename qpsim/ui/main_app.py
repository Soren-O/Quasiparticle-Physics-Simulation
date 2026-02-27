from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..geometry import (
    create_geometry_from_gds,
    create_intrinsic_geometry,
    discover_gds_layers,
    gds_support_available,
    point_to_segment_distance,
)
from ..initial_conditions import build_initial_field, default_initial_condition
from ..models import (
    BoundaryCondition,
    ExternalGenerationSpec,
    GeometryData,
    InitialConditionSpec,
    SetupData,
    SimulationParameters,
    SimulationResultData,
    TestGeometryGroupData,
    TestSuiteData,
    utc_now_iso,
)
from ..paths import ROOT_DIR, SIMULATIONS_DIR, ensure_data_dirs
from ..solver import (
    BoundaryAssignmentError,
    build_energy_grid,
    run_2d_crank_nicolson,
    thermal_qp_weights,
)
from ..storage import (
    create_setup_id,
    create_simulation_id,
    frame_from_jsonable,
    frame_to_jsonable,
    latest_test_suite_file,
    list_simulation_files,
    load_precomputed,
    load_test_geometry_group,
    load_setup,
    load_simulation,
    load_test_suite,
    precomputed_exists,
    save_precomputed,
    save_setup,
    save_simulation,
)
from ..precompute import estimate_precompute_memory, precompute_arrays, validate_precomputed
from ..test_cases import generate_and_save_test_suite
from .dialogs import ask_boundary_condition, ask_external_generation, ask_initial_condition, show_material_reference
from .theme import FONT_MONO, FONT_TITLE, RETRO_ACCENT, RETRO_BG, RETRO_PANEL, apply_retro_theme


class BusyDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str, message: str):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=RETRO_PANEL)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        body = tk.Frame(self, bg=RETRO_PANEL)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        tk.Label(body, text=message, bg=RETRO_PANEL, justify="left").pack(anchor="w", pady=(0, 8))
        self.progress = ttk.Progressbar(body, mode="indeterminate", length=320)
        self.progress.pack(fill="x", expand=True)
        self.progress.start(12)
        self.update_idletasks()

    def close(self) -> None:
        try:
            self.progress.stop()
        except Exception:
            pass
        if self.winfo_exists():
            self.grab_release()
            self.destroy()


class SimulationViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, result: SimulationResultData):
        super().__init__(parent)
        self.title(f"Live Simulation - {result.setup_name}")
        self.configure(bg=RETRO_PANEL)
        self.geometry("1200x700")

        self.times = np.asarray(result.times, dtype=float)
        self.frames = [frame_from_jsonable(frame) for frame in result.frames]
        self.mass = np.asarray(result.mass_over_time, dtype=float)
        self.index = 0
        self.playing = True
        self.after_id: str | None = None

        top_controls = tk.Frame(self, bg=RETRO_PANEL)
        top_controls.pack(fill="x", padx=8, pady=6)
        self.play_btn = tk.Button(top_controls, text="Pause", width=10, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=4)
        tk.Button(top_controls, text="Prev", width=10, command=self.prev_frame).pack(side="left", padx=4)
        tk.Button(top_controls, text="Next", width=10, command=self.next_frame).pack(side="left", padx=4)
        self.time_label = tk.Label(top_controls, text="", bg=RETRO_PANEL)
        self.time_label.pack(side="left", padx=12)

        self.scale = tk.Scale(
            top_controls,
            from_=0,
            to=max(0, len(self.frames) - 1),
            orient="horizontal",
            showvalue=False,
            command=self.on_scale,
            length=380,
            bg=RETRO_PANEL,
        )
        self.scale.pack(side="left", padx=8)

        fig = Figure(figsize=(11.0, 6.5), dpi=100)
        self.ax_field = fig.add_subplot(1, 2, 1)
        self.ax_mass = fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        vmin, vmax = float(result.color_limits[0]), float(result.color_limits[1])
        self.heatmap = self.ax_field.imshow(self.frames[0], origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
        has_energy = result.energy_bins is not None
        self.ax_field.set_title("Quasiparticle Density (energy-integrated)" if has_energy else "2D Density Heatmap")
        self.ax_field.set_xlabel("x (\u03bcm)")
        self.ax_field.set_ylabel("y (\u03bcm)")
        self.ax_field.set_aspect("equal")
        fig.colorbar(self.heatmap, ax=self.ax_field, fraction=0.046, pad=0.04)

        self.mass_line, = self.ax_mass.plot([], [], color="#004488", linewidth=2.0, label="Total QP Number" if has_energy else "Total Mass")
        self.mass_marker, = self.ax_mass.plot([], [], "o", color="#AA2200")
        self.ax_mass.set_title("Total Quasiparticle Number" if has_energy else "Mass Over Time")
        self.ax_mass.set_xlabel("Time (ns)")
        self.ax_mass.set_ylabel("Total Quasiparticle Number" if has_energy else "Total Mass")
        self.ax_mass.grid(True, alpha=0.3)
        self.ax_mass.legend(loc="best")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.render_frame()
        self.schedule_tick()

    def on_close(self) -> None:
        self.playing = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.destroy()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text=("Pause" if self.playing else "Play"))
        if self.playing:
            self.schedule_tick()

    def prev_frame(self) -> None:
        self.index = max(0, self.index - 1)
        self.render_frame()

    def next_frame(self) -> None:
        self.index = min(len(self.frames) - 1, self.index + 1)
        self.render_frame()

    def on_scale(self, raw_value: str) -> None:
        self.index = int(float(raw_value))
        self.render_frame()

    def schedule_tick(self) -> None:
        if not self.playing:
            return
        self.after_id = self.after(120, self.tick)

    def tick(self) -> None:
        if self.playing:
            self.index = (self.index + 1) % len(self.frames)
            self.render_frame()
            self.schedule_tick()

    def render_frame(self) -> None:
        self.index = int(np.clip(self.index, 0, len(self.frames) - 1))
        self.scale.set(self.index)
        t = float(self.times[self.index])
        self.time_label.configure(text=f"t = {t:.3f} ns")

        self.heatmap.set_data(self.frames[self.index])
        self.mass_line.set_data(self.times[: self.index + 1], self.mass[: self.index + 1])
        self.mass_marker.set_data([self.times[self.index]], [self.mass[self.index]])
        self.ax_mass.relim()
        self.ax_mass.autoscale_view()
        self.canvas.draw_idle()


class LineTestSuiteViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, group: TestGeometryGroupData):
        super().__init__(parent)
        self.title(f"Test Simulations - {group.title}")
        self.configure(bg=RETRO_PANEL)
        self.geometry("1240x760")
        self.group = group
        self.cases = list(group.cases)
        if not self.cases:
            raise ValueError("No test cases available for this geometry.")

        self.case_index = 0
        self.frame_index = 0
        self.playing = True
        self.frame_delay_ms = 300
        self.after_id: str | None = None

        controls = tk.Frame(self, bg=RETRO_PANEL)
        controls.pack(fill="x", padx=8, pady=6)
        self.case_label = tk.Label(controls, text="", bg=RETRO_PANEL, fg=RETRO_ACCENT, font=("Tahoma", 10, "bold"))
        self.case_label.pack(side="left", padx=8)
        tk.Button(controls, text="Prev Case", width=12, command=self.prev_case).pack(side="left", padx=4)
        tk.Button(controls, text="Next Case", width=12, command=self.next_case).pack(side="left", padx=4)
        self.play_btn = tk.Button(controls, text="Pause", width=10, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=8)
        self.scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            command=self.on_scale,
            length=360,
            bg=RETRO_PANEL,
        )
        self.scale.pack(side="left", padx=8)
        self.time_label = tk.Label(controls, text="", bg=RETRO_PANEL)
        self.time_label.pack(side="left", padx=8)

        fig = Figure(figsize=(11.5, 7.0), dpi=100)
        self.ax_trace = fig.add_subplot(1, 2, 1)
        self.ax_info = fig.add_subplot(1, 2, 2)
        self.ax_info.axis("off")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.sim_line, = self.ax_trace.plot([], [], color="#1E88E5", linewidth=2.2, label="Simulated")
        self.ana_line, = self.ax_trace.plot([], [], color="#D81B60", linewidth=2.0, linestyle="--", label="Analytic")
        self.ax_trace.set_xlabel("x")
        self.ax_trace.set_ylabel("Density")
        self.ax_trace.set_title("Simulation vs Analytic")
        self.ax_trace.grid(True, alpha=0.25)
        self.ax_trace.legend(loc="best")
        self.info_text = self.ax_info.text(0.02, 0.98, "", va="top", ha="left", wrap=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.render_case(reset_frame=True)
        self.schedule_tick()

    def on_close(self) -> None:
        self.playing = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.destroy()

    def current_case(self):
        return self.cases[self.case_index]

    def prev_case(self) -> None:
        self.case_index = (self.case_index - 1) % len(self.cases)
        self.render_case(reset_frame=True)

    def next_case(self) -> None:
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.render_case(reset_frame=True)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text=("Pause" if self.playing else "Play"))
        if self.playing:
            self.schedule_tick()

    def on_scale(self, raw_value: str) -> None:
        self.frame_index = int(float(raw_value))
        self.render_frame()

    def schedule_tick(self) -> None:
        if not self.playing:
            return
        self.after_id = self.after(self.frame_delay_ms, self.tick)

    def tick(self) -> None:
        if self.playing:
            case = self.current_case()
            self.frame_index = (self.frame_index + 1) % len(case.times)
            self.render_frame()
            self.schedule_tick()

    def render_case(self, reset_frame: bool = False) -> None:
        case = self.current_case()
        self.case_label.configure(text=f"{self.group.title} | Case {self.case_index + 1}/{len(self.cases)}: {case.title}")
        self.scale.configure(to=max(0, len(case.times) - 1))
        if reset_frame:
            self.frame_index = 0

        panel = (
            f"Geometry:\n{self.group.title}\n\n"
            f"Boundary Condition:\n{case.boundary_label}\n\n"
            f"Analytic Formula:\n${case.formula_latex}$\n\n"
            f"Initial Condition:\n${case.initial_condition_latex}$\n\n"
            f"Description:\n{case.description}"
        )
        self.info_text.set_text(panel)

        x_all = np.asarray(case.x, dtype=float)
        sim_all = np.asarray(case.simulated, dtype=float)
        ana_all = np.asarray(case.analytic, dtype=float)
        y_min = float(min(np.min(sim_all), np.min(ana_all)))
        y_max = float(max(np.max(sim_all), np.max(ana_all)))
        pad = 1e-6 if abs(y_max - y_min) < 1e-12 else 0.05 * (y_max - y_min)
        self.ax_trace.set_xlim(float(np.min(x_all)), float(np.max(x_all)))
        self.ax_trace.set_ylim(y_min - pad, y_max + pad)
        self.render_frame()

    def render_frame(self) -> None:
        case = self.current_case()
        self.frame_index = int(np.clip(self.frame_index, 0, len(case.times) - 1))
        self.scale.set(self.frame_index)
        time_value = float(case.times[self.frame_index])
        self.time_label.configure(text=f"t = {time_value:.3f}")

        x = np.asarray(case.x, dtype=float)
        sim = np.asarray(case.simulated[self.frame_index], dtype=float)
        ana = np.asarray(case.analytic[self.frame_index], dtype=float)
        self.sim_line.set_data(x, sim)
        self.ana_line.set_data(x, ana)
        self.canvas.draw_idle()


class TimeSeriesTestViewer(tk.Toplevel):
    """Viewer for time-series test cases (e.g. recombination dynamics).

    Shows the full n(t) trajectory at once — no time slider or animation.
    Left subplot: simulated vs analytic curves.
    Right subplot: info panel with formula and description.
    """

    def __init__(self, parent: tk.Misc, group: TestGeometryGroupData):
        super().__init__(parent)
        self.title(f"Test Simulations - {group.title}")
        self.configure(bg=RETRO_PANEL)
        self.geometry("1240x760")
        self.group = group
        self.cases = list(group.cases)
        if not self.cases:
            raise ValueError("No test cases available for this geometry.")
        self.case_index = 0

        controls = tk.Frame(self, bg=RETRO_PANEL)
        controls.pack(fill="x", padx=8, pady=6)
        self.case_label = tk.Label(
            controls, text="", bg=RETRO_PANEL, fg=RETRO_ACCENT,
            font=("Tahoma", 10, "bold"),
        )
        self.case_label.pack(side="left", padx=8)
        tk.Button(controls, text="Prev Case", width=12, command=self.prev_case).pack(side="left", padx=4)
        tk.Button(controls, text="Next Case", width=12, command=self.next_case).pack(side="left", padx=4)

        fig = Figure(figsize=(11.5, 7.0), dpi=100)
        self.ax_trace = fig.add_subplot(1, 2, 1)
        self.ax_info = fig.add_subplot(1, 2, 2)
        self.ax_info.axis("off")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.sim_line, = self.ax_trace.plot([], [], color="#1E88E5", linewidth=2.2, label="Simulated")
        self.ana_line, = self.ax_trace.plot([], [], color="#D81B60", linewidth=2.0, linestyle="--", label="Analytic")
        self.ax_trace.set_xlabel("Time (ns)")
        self.ax_trace.set_ylabel("QP Spectral Density n(E)")
        self.ax_trace.set_title("Simulation vs Analytic")
        self.ax_trace.grid(True, alpha=0.25)
        self.ax_trace.legend(loc="best")
        self.info_text = self.ax_info.text(0.02, 0.98, "", va="top", ha="left", wrap=True)

        self.render_case()

    def current_case(self):
        return self.cases[self.case_index]

    def prev_case(self) -> None:
        self.case_index = (self.case_index - 1) % len(self.cases)
        self.render_case()

    def next_case(self) -> None:
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.render_case()

    def render_case(self) -> None:
        case = self.current_case()
        self.case_label.configure(
            text=f"{self.group.title} | Case {self.case_index + 1}/{len(self.cases)}: {case.title}",
        )

        panel = (
            f"Geometry:\n{self.group.title}\n\n"
            f"Boundary Condition:\n{case.boundary_label}\n\n"
            f"Analytic Formula:\n${case.formula_latex}$\n\n"
            f"Initial Condition:\n${case.initial_condition_latex}$\n\n"
            f"Description:\n{case.description}"
        )
        self.info_text.set_text(panel)

        # x stores the time axis; simulated/analytic are [[values...]] (single frame)
        t = np.asarray(case.x, dtype=float)
        sim = np.asarray(case.simulated[0], dtype=float)
        ana = np.asarray(case.analytic[0], dtype=float)

        y_min = float(min(np.min(sim), np.min(ana)))
        y_max = float(max(np.max(sim), np.max(ana)))
        pad = 1e-6 if abs(y_max - y_min) < 1e-12 else 0.05 * (y_max - y_min)
        self.ax_trace.set_xlim(float(np.min(t)), float(np.max(t)))
        self.ax_trace.set_ylim(y_min - pad, y_max + pad)

        self.sim_line.set_data(t, sim)
        self.ana_line.set_data(t, ana)
        self.canvas.draw_idle()


class HeatmapTestSuiteViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, group: TestGeometryGroupData):
        super().__init__(parent)
        self.title(f"Test Simulations - {group.title}")
        self.configure(bg=RETRO_PANEL)
        self.geometry("1360x780")
        self.group = group
        self.cases = list(group.cases)
        if not self.cases:
            raise ValueError("No test cases available for this geometry.")

        self.case_index = 0
        self.frame_index = 0
        self.playing = True
        self.frame_delay_ms = 300
        self.after_id: str | None = None
        self.sim_im = None
        self.ana_im = None
        self.colorbar = None
        self._sim_frames_cache: np.ndarray | None = None
        self._ana_frames_cache: np.ndarray | None = None

        controls = tk.Frame(self, bg=RETRO_PANEL)
        controls.pack(fill="x", padx=8, pady=6)
        self.case_label = tk.Label(controls, text="", bg=RETRO_PANEL, fg=RETRO_ACCENT, font=("Tahoma", 10, "bold"))
        self.case_label.pack(side="left", padx=8)
        tk.Button(controls, text="Prev Case", width=12, command=self.prev_case).pack(side="left", padx=4)
        tk.Button(controls, text="Next Case", width=12, command=self.next_case).pack(side="left", padx=4)
        self.play_btn = tk.Button(controls, text="Pause", width=10, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=8)
        self.scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            command=self.on_scale,
            length=360,
            bg=RETRO_PANEL,
        )
        self.scale.pack(side="left", padx=8)
        self.time_label = tk.Label(controls, text="", bg=RETRO_PANEL)
        self.time_label.pack(side="left", padx=8)

        self.fig = Figure(figsize=(12.2, 7.0), dpi=100)
        self.ax_sim = self.fig.add_subplot(1, 3, 1)
        self.ax_ana = self.fig.add_subplot(1, 3, 2)
        self.ax_info = self.fig.add_subplot(1, 3, 3)
        self.ax_info.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.info_text = self.ax_info.text(0.02, 0.98, "", va="top", ha="left", wrap=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.render_case(reset_frame=True)
        self.schedule_tick()

    def on_close(self) -> None:
        self.playing = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.destroy()

    def current_case(self):
        return self.cases[self.case_index]

    def prev_case(self) -> None:
        self.case_index = (self.case_index - 1) % len(self.cases)
        self.render_case(reset_frame=True)

    def next_case(self) -> None:
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.render_case(reset_frame=True)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text=("Pause" if self.playing else "Play"))
        if self.playing:
            self.schedule_tick()

    def on_scale(self, raw_value: str) -> None:
        self.frame_index = int(float(raw_value))
        self.render_frame()

    def schedule_tick(self) -> None:
        if not self.playing:
            return
        self.after_id = self.after(self.frame_delay_ms, self.tick)

    def tick(self) -> None:
        if self.playing:
            case = self.current_case()
            self.frame_index = (self.frame_index + 1) % len(case.times)
            self.render_frame()
            self.schedule_tick()

    def _as_frames(self, payload) -> np.ndarray:
        frames = np.array(payload, dtype=float)
        if frames.ndim != 3:
            raise ValueError("Heatmap test case data must have shape [time, y, x].")
        return frames

    def render_case(self, reset_frame: bool = False) -> None:
        case = self.current_case()
        self.case_label.configure(text=f"{self.group.title} | Case {self.case_index + 1}/{len(self.cases)}: {case.title}")
        self.scale.configure(to=max(0, len(case.times) - 1))
        if reset_frame:
            self.frame_index = 0

        panel = (
            f"Geometry:\n{self.group.title}\n\n"
            f"Boundary Condition:\n{case.boundary_label}\n\n"
            f"Analytic Formula:\n${case.formula_latex}$\n\n"
            f"Initial Condition:\n${case.initial_condition_latex}$\n\n"
            f"Description:\n{case.description}"
        )
        self.info_text.set_text(panel)

        self._sim_frames_cache = self._as_frames(case.simulated)
        self._ana_frames_cache = self._as_frames(case.analytic)
        sim_all = self._sim_frames_cache
        ana_all = self._ana_frames_cache
        vmin = float(np.nanmin(np.concatenate([sim_all.ravel(), ana_all.ravel()])))
        vmax = float(np.nanmax(np.concatenate([sim_all.ravel(), ana_all.ravel()])))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-9

        self.ax_sim.clear()
        self.ax_ana.clear()
        self.ax_sim.set_title("Simulated")
        self.ax_ana.set_title("Analytic")
        self.ax_sim.set_axis_off()
        self.ax_ana.set_axis_off()
        self.sim_im = self.ax_sim.imshow(
            sim_all[0],
            origin="lower",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self.ana_im = self.ax_ana.imshow(
            ana_all[0],
            origin="lower",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(
                self.sim_im,
                ax=[self.ax_sim, self.ax_ana],
                fraction=0.046,
                pad=0.03,
            )
            self.colorbar.set_label("Density")
        else:
            self.colorbar.update_normal(self.sim_im)
            self.colorbar.ax.set_visible(True)
            self.colorbar.set_label("Density")
        self.render_frame()

    def render_frame(self) -> None:
        case = self.current_case()
        sim_all = self._sim_frames_cache
        ana_all = self._ana_frames_cache
        if sim_all is None or ana_all is None:
            sim_all = self._as_frames(case.simulated)
            ana_all = self._as_frames(case.analytic)
            self._sim_frames_cache = sim_all
            self._ana_frames_cache = ana_all
        self.frame_index = int(np.clip(self.frame_index, 0, len(case.times) - 1))
        self.scale.set(self.frame_index)
        time_value = float(case.times[self.frame_index])
        self.time_label.configure(text=f"t = {time_value:.3f}")

        if self.sim_im is not None:
            self.sim_im.set_data(sim_all[self.frame_index])
        if self.ana_im is not None:
            self.ana_im.set_data(ana_all[self.frame_index])
        self.canvas.draw_idle()


class TestGeometryLanding(tk.Toplevel):
    def __init__(self, parent: tk.Misc, suite: TestSuiteData, suite_path: str | None = None):
        super().__init__(parent)
        self.title("Test Simulation Geometries")
        self.configure(bg=RETRO_PANEL)
        self.geometry("980x740")
        self.suite = suite
        self.suite_path = suite_path
        self.preview_canvases: list[FigureCanvasTkAgg] = []

        groups = list(suite.geometry_groups)
        if not groups:
            raise ValueError("No test geometries found in this suite.")

        tk.Label(
            self,
            text="Select Test Geometry",
            font=("Tahoma", 14, "bold"),
            fg=RETRO_ACCENT,
            bg=RETRO_PANEL,
        ).pack(anchor="w", padx=12, pady=(10, 6))
        tk.Label(
            self,
            text="Each geometry has its own case set and replay mode.",
            bg=RETRO_PANEL,
        ).pack(anchor="w", padx=12, pady=(0, 10))

        outer = tk.Frame(self, bg=RETRO_PANEL)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        scroll_canvas = tk.Canvas(outer, bg=RETRO_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=scroll_canvas.yview)
        container = tk.Frame(scroll_canvas, bg=RETRO_PANEL)

        container.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.create_window((0, 0), window=container, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event: tk.Event) -> None:
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind("<Destroy>", lambda e: scroll_canvas.unbind_all("<MouseWheel>"), add=True)

        # Stretch the inner frame to the canvas width so cards fill horizontally
        def _resize_inner(event: tk.Event) -> None:
            scroll_canvas.itemconfigure("all", width=event.width)
        scroll_canvas.bind("<Configure>", _resize_inner)

        for group in groups:
            card = tk.Frame(container, bg="#E7E4DA", bd=2, relief="groove")
            card.pack(fill="x", pady=8)

            preview = np.array(group.preview_mask, dtype=float)
            if preview.ndim == 2 and preview.shape[0] <= 2:
                preview = np.repeat(preview, repeats=8, axis=0)
            if preview.ndim == 2 and np.nanmax(preview) - np.nanmin(preview) < 1e-12:
                preview = np.pad(preview, pad_width=2, mode="constant", constant_values=0.0)
            fig = Figure(figsize=(2.0, 1.5), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(preview, origin="lower", cmap="Greys", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            canvas = FigureCanvasTkAgg(fig, master=card)
            canvas.get_tk_widget().pack(side="left", padx=8, pady=8)
            canvas.draw_idle()
            self.preview_canvases.append(canvas)

            text_col = tk.Frame(card, bg="#E7E4DA")
            text_col.pack(side="left", fill="both", expand=True, padx=8, pady=8)
            tk.Label(text_col, text=group.title, bg="#E7E4DA", fg=RETRO_ACCENT, font=("Tahoma", 11, "bold")).pack(anchor="w")
            tk.Label(text_col, text=group.description, bg="#E7E4DA", justify="left", wraplength=520).pack(anchor="w", pady=(4, 2))
            case_total = int(group.case_count or len(group.cases))
            tk.Label(text_col, text=f"Cases: {case_total} | Mode: {group.view_mode}", bg="#E7E4DA").pack(anchor="w")

            tk.Button(
                card,
                text="Open",
                width=12,
                command=lambda g=group: self.open_group(g),
            ).pack(side="right", padx=12)

    def open_group(self, group: TestGeometryGroupData) -> None:
        if not group.cases and self.suite_path:
            dialog = BusyDialog(self, title="Loading Geometry Cases", message=f"Loading cases for: {group.title}")
            result_queue: queue.Queue = queue.Queue()

            def worker() -> None:
                try:
                    loaded_group = load_test_geometry_group(self.suite_path, group.geometry_id)
                except Exception as exc:
                    result_queue.put(("error", exc))
                else:
                    result_queue.put(("ok", loaded_group))

            threading.Thread(target=worker, daemon=True).start()

            def poll() -> None:
                try:
                    kind, payload = result_queue.get_nowait()
                except queue.Empty:
                    if dialog.winfo_exists():
                        self.after(80, poll)
                    return

                dialog.close()
                if kind == "error":
                    messagebox.showerror("Load Failed", str(payload), parent=self)
                    return
                self._open_group_view(payload)

            self.after(80, poll)
            return
        self._open_group_view(group)

    def _open_group_view(self, group: TestGeometryGroupData) -> None:
        mode = (group.view_mode or "").strip().lower()
        if mode == "heatmap2d":
            HeatmapTestSuiteViewer(self, group=group)
        elif mode == "timeseries":
            TimeSeriesTestViewer(self, group=group)
        else:
            LineTestSuiteViewer(self, group=group)

class SetupEditor(tk.Toplevel):
    def __init__(self, parent: tk.Misc, setup: SetupData | None = None):
        super().__init__(parent)
        self.title("Create / Edit Simulation Setup")
        self.configure(bg=RETRO_PANEL)
        self.geometry("1320x760")

        self.setup_id = setup.setup_id if setup else create_setup_id()
        self.geometry_data: GeometryData | None = None
        self.mask: np.ndarray | None = None
        self.initial_condition: InitialConditionSpec = (
            setup.initial_condition if setup else default_initial_condition()
        )
        self.boundary_assignments: dict[str, BoundaryCondition] = (
            dict(setup.boundary_conditions) if setup else {}
        )
        self.latest_result: SimulationResultData | None = None
        self.hover_edge_id: str | None = None
        self.simulation_running = False
        self.last_saved_path: str | None = None
        self._external_generation: ExternalGenerationSpec = (
            setup.parameters.external_generation if setup else ExternalGenerationSpec()
        )

        root = tk.Frame(self, bg=RETRO_PANEL)
        root.pack(fill="both", expand=True, padx=8, pady=8)
        left = tk.Frame(root, bg=RETRO_PANEL, width=340)
        left.pack(side="left", fill="y")
        right = tk.Frame(root, bg=RETRO_PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.name_var = tk.StringVar(value=(setup.name if setup else "Untitled Setup"))
        default_params = setup.parameters if setup else SimulationParameters(
            diffusion_coefficient=6.0,
            dt=2.0,
            total_time=2000.0,
            mesh_size=1.0,
            store_every=5,
            energy_gap=180.0,
            energy_max_factor=10.0,
            num_energy_bins=50,
        )
        self.diff_var = tk.StringVar(value=str(default_params.diffusion_coefficient))
        self.mesh_var = tk.StringVar(value=str(default_params.mesh_size))
        self.dt_var = tk.StringVar(value=str(default_params.dt))
        self.total_var = tk.StringVar(value=str(default_params.total_time))
        self.store_var = tk.StringVar(value=str(default_params.store_every))
        self.energy_gap_var = tk.StringVar(value=str(default_params.energy_gap))
        self.energy_min_var = tk.StringVar(value=str(default_params.energy_min_factor))
        self.energy_max_var = tk.StringVar(value=str(default_params.energy_max_factor))
        self.energy_bins_var = tk.StringVar(value=str(default_params.num_energy_bins))
        self.dynes_gamma_var = tk.StringVar(value=str(default_params.dynes_gamma))
        self.gap_expression_var = tk.StringVar(value=default_params.gap_expression)
        self.collision_solver_var = tk.StringVar(value=default_params.collision_solver)
        self.enable_diffusion_var = tk.BooleanVar(value=default_params.enable_diffusion)
        self.enable_recombination_var = tk.BooleanVar(value=default_params.enable_recombination)
        self.enable_scattering_var = tk.BooleanVar(value=default_params.enable_scattering)
        self.tau_s_var = tk.StringVar(
            value=str(default_params.tau_s if default_params.tau_s is not None else default_params.tau_0)
        )
        self.tau_r_var = tk.StringVar(
            value=str(default_params.tau_r if default_params.tau_r is not None else default_params.tau_0)
        )
        self.T_c_var = tk.StringVar(value=str(default_params.T_c))
        self.bath_temp_var = tk.StringVar(value=str(default_params.bath_temperature))

        tk.Label(left, text="Setup Name", bg=RETRO_PANEL).pack(anchor="w")
        tk.Entry(left, textvariable=self.name_var, width=36).pack(anchor="w", pady=(0, 8))

        def add_param(label: str, var: tk.StringVar) -> None:
            tk.Label(left, text=label, bg=RETRO_PANEL).pack(anchor="w")
            tk.Entry(left, textvariable=var, width=18).pack(anchor="w", pady=(0, 6))

        add_param("Diffusion Coeff D\u2080 (\u03bcm\u00b2/ns)", self.diff_var)
        add_param("Mesh Size dx (\u03bcm)", self.mesh_var)
        add_param("dt (ns)", self.dt_var)
        add_param("Total Time (ns)", self.total_var)
        add_param("Store Every N Steps", self.store_var)
        add_param("Energy Gap \u0394 (\u03bceV)", self.energy_gap_var)
        add_param("Energy Min (\u00d7\u0394)", self.energy_min_var)
        add_param("Energy Max (\u00d7\u0394)", self.energy_max_var)
        add_param("Energy Bins", self.energy_bins_var)
        add_param("Dynes Gamma \u0393 (\u03bceV)", self.dynes_gamma_var)
        tk.Label(left, text="\u0394(x,y) Expression (empty=uniform)", bg=RETRO_PANEL).pack(anchor="w")
        tk.Entry(left, textvariable=self.gap_expression_var, width=36).pack(anchor="w", pady=(0, 6))

        # --- Physics process toggles ---
        tk.Label(left, text="Physics Processes", bg=RETRO_PANEL, font=("Tahoma", 9, "bold")).pack(anchor="w", pady=(8, 2))
        tk.Checkbutton(
            left,
            text="Enable Diffusion",
            variable=self.enable_diffusion_var,
            bg=RETRO_PANEL,
            anchor="w",
            command=self.update_status,
        ).pack(anchor="w")
        tk.Checkbutton(left, text="Enable Recombination", variable=self.enable_recombination_var, bg=RETRO_PANEL, anchor="w",
                        command=self._toggle_recomb_fields).pack(anchor="w")
        self.scattering_cb = tk.Checkbutton(left, text="Enable Scattering", variable=self.enable_scattering_var,
                                             bg=RETRO_PANEL, anchor="w")
        self.scattering_cb.pack(anchor="w")
        tk.Label(left, text="Collision Solver", bg=RETRO_PANEL).pack(anchor="w")
        self.collision_solver_menu = ttk.Combobox(
            left, state="readonly", width=18, textvariable=self.collision_solver_var,
            values=["forward_euler", "bdf"],
        )
        self.collision_solver_menu.pack(anchor="w", pady=(0, 4))

        # --- Recombination parameters (shown/hidden based on checkbox) ---
        self.recomb_frame = tk.Frame(left, bg=RETRO_PANEL)
        self.recomb_frame.pack(anchor="w", fill="x")
        add_param_in = lambda parent, label, var: (
            tk.Label(parent, text=label, bg=RETRO_PANEL).pack(anchor="w"),
            tk.Entry(parent, textvariable=var, width=18).pack(anchor="w", pady=(0, 4)),
        )
        add_param_in(self.recomb_frame, "\u03c4\u2080,s (ns)", self.tau_s_var)
        add_param_in(self.recomb_frame, "\u03c4\u2080,r (ns)", self.tau_r_var)
        add_param_in(self.recomb_frame, "T\u2081 (K)", self.T_c_var)
        add_param_in(self.recomb_frame, "Bath Temperature (K)", self.bath_temp_var)
        self._toggle_recomb_fields()

        tk.Button(
            left, text="Material Reference Table...", width=30,
            command=self.show_material_reference,
        ).pack(anchor="w", pady=(6, 4))
        tk.Button(
            left, text="Pre-compute Arrays", width=30,
            command=self.run_precompute,
        ).pack(anchor="w", pady=4)
        self.precompute_label = tk.Label(left, text="", bg=RETRO_PANEL, fg="#666666", justify="left", wraplength=320)
        self.precompute_label.pack(anchor="w")

        tk.Button(left, text="Import .GDS Geometry", width=30, command=self.import_gds).pack(anchor="w", pady=(10, 4))
        tk.Button(left, text="Use Intrinsic Test Geometry", width=30, command=self.use_intrinsic).pack(anchor="w", pady=4)
        tk.Button(left, text="Initial Conditions...", width=30, command=self.edit_initial_conditions).pack(anchor="w", pady=4)
        tk.Button(left, text="External Generation...", width=30, command=self.edit_external_generation).pack(anchor="w", pady=4)
        tk.Button(left, text="Set Unassigned -> Reflective", width=30, command=self.fill_unassigned_reflective).pack(
            anchor="w", pady=4
        )
        tk.Button(left, text="Save Setup", width=30, command=self.save_setup_file).pack(anchor="w", pady=(14, 4))
        self.run_btn = tk.Button(left, text="Run Simulation", width=30, command=self.run_simulation)
        self.run_btn.pack(anchor="w", pady=4)
        self.auto_live_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="View Live Simulation On Run",
            variable=self.auto_live_var,
            bg=RETRO_PANEL,
            anchor="w",
        ).pack(anchor="w", pady=4)

        self.status_label = tk.Label(left, text="", bg=RETRO_PANEL, justify="left", wraplength=320)
        self.status_label.pack(anchor="w", pady=(12, 6))
        self.hover_label = tk.Label(left, text="Hover edge: none", bg=RETRO_PANEL, fg=RETRO_ACCENT, justify="left")
        self.hover_label.pack(anchor="w")

        fig = Figure(figsize=(8.6, 6.8), dpi=100)
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_title("Import geometry to begin")
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        if setup is not None:
            self.load_setup_data(setup)
        else:
            self.update_status()

    def load_setup_data(self, setup: SetupData) -> None:
        self.setup_id = setup.setup_id
        self.name_var.set(setup.name)
        self.diff_var.set(str(setup.parameters.diffusion_coefficient))
        self.mesh_var.set(f"{float(setup.geometry.mesh_size):g}")
        self.dt_var.set(str(setup.parameters.dt))
        self.total_var.set(str(setup.parameters.total_time))
        self.store_var.set(str(setup.parameters.store_every))
        self.energy_gap_var.set(str(setup.parameters.energy_gap))
        self.energy_min_var.set(str(setup.parameters.energy_min_factor))
        self.energy_max_var.set(str(setup.parameters.energy_max_factor))
        self.energy_bins_var.set(str(setup.parameters.num_energy_bins))
        self.dynes_gamma_var.set(str(setup.parameters.dynes_gamma))
        self.gap_expression_var.set(setup.parameters.gap_expression)
        self.collision_solver_var.set(setup.parameters.collision_solver)
        self._external_generation = setup.parameters.external_generation
        self.enable_diffusion_var.set(setup.parameters.enable_diffusion)
        self.enable_recombination_var.set(setup.parameters.enable_recombination)
        self.enable_scattering_var.set(setup.parameters.enable_scattering)
        self.tau_s_var.set(
            str(setup.parameters.tau_s if setup.parameters.tau_s is not None else setup.parameters.tau_0)
        )
        self.tau_r_var.set(
            str(setup.parameters.tau_r if setup.parameters.tau_r is not None else setup.parameters.tau_0)
        )
        self.T_c_var.set(str(setup.parameters.T_c))
        self.bath_temp_var.set(str(setup.parameters.bath_temperature))
        self._toggle_recomb_fields()
        self.geometry_data = setup.geometry
        self.mask = np.array(setup.geometry.mask, dtype=bool)
        self.boundary_assignments = dict(setup.boundary_conditions)
        self.initial_condition = setup.initial_condition
        self.redraw_geometry()
        self.update_status()

    def _toggle_recomb_fields(self) -> None:
        if self.enable_recombination_var.get():
            self.recomb_frame.pack(anchor="w", fill="x")
        else:
            self.recomb_frame.pack_forget()

    def parse_float(self, name: str, value: str) -> float:
        try:
            return float(value.strip())
        except Exception as exc:
            raise ValueError(f"'{name}' must be numeric.") from exc

    def parse_int(self, name: str, value: str) -> int:
        try:
            return int(value.strip())
        except Exception as exc:
            raise ValueError(f"'{name}' must be an integer.") from exc

    def import_gds(self) -> None:
        if not gds_support_available():
            messagebox.showerror(
                "GDS Support Missing",
                "gdstk is not installed. Run Install Requirements first.",
                parent=self,
            )
            return
        gds_path = filedialog.askopenfilename(
            parent=self,
            title="Select GDS File",
            filetypes=[("GDSII files", "*.gds"), ("All files", "*.*")],
        )
        if not gds_path:
            return
        try:
            mesh_size = self.parse_float("mesh size", self.mesh_var.get())
            layers = discover_gds_layers(gds_path)
            if len(layers) == 1:
                layer = layers[0]
            else:
                layer = simpledialog.askinteger(
                    "Select Layer",
                    f"Layers found: {layers}\nEnter layer number to import:",
                    parent=self,
                )
                if layer is None:
                    return
                if layer not in layers:
                    raise ValueError(f"Layer {layer} is not present in this GDS file.")
            geometry = create_geometry_from_gds(gds_path, layer=layer, mesh_size=mesh_size)
        except Exception as exc:
            messagebox.showerror("GDS Import Failed", str(exc), parent=self)
            return
        self.geometry_data = geometry
        self.mesh_var.set(f"{float(geometry.mesh_size):g}")
        self.mask = np.array(geometry.mask, dtype=bool)
        self.boundary_assignments = {}
        self.hover_edge_id = None
        self.redraw_geometry()
        self.update_status()

    def show_material_reference(self) -> None:
        def on_select(D0: float, gap_ueV: int, Tc_K: float, tau_0_ns: float) -> None:
            self.diff_var.set(str(D0))
            self.energy_gap_var.set(str(gap_ueV))
            self.T_c_var.set(str(Tc_K))
            self.tau_s_var.set(str(tau_0_ns))
            self.tau_r_var.set(str(tau_0_ns))

        show_material_reference(self, on_select=on_select)

    def run_precompute(self) -> None:
        if self.geometry_data is None or self.mask is None:
            messagebox.showerror("Cannot Pre-compute", "Import geometry first.", parent=self)
            return
        try:
            setup = self.build_setup()
        except Exception as exc:
            messagebox.showerror("Pre-compute Failed", str(exc), parent=self)
            return
        if not self.last_saved_path:
            messagebox.showinfo(
                "Save First",
                "Save the setup before pre-computing (needed for .npz sidecar file).",
                parent=self,
            )
            return

        mask_snap = self.mask.copy()

        def task_fn():
            return precompute_arrays(mask_snap, setup.geometry.edges, setup.boundary_conditions, setup.parameters)

        def on_success(arrays):
            from pathlib import Path as P
            npz_path = save_precomputed(P(self.last_saved_path), arrays)
            n_spatial = int(np.sum(mask_snap))
            NE = setup.parameters.num_energy_bins
            is_uniform = bool(arrays.get("is_uniform", True))
            mem = estimate_precompute_memory(n_spatial, NE, is_uniform)
            self.precompute_label.configure(
                text=f"Pre-computed: {npz_path.name} ({mem / 1024 / 1024:.1f} MB est.)"
            )
            messagebox.showinfo("Pre-compute Done", f"Saved: {npz_path}", parent=self)

        self._run_background_task("Pre-compute", "Computing collision kernels...", task_fn, on_success)

    def use_intrinsic(self) -> None:
        try:
            mesh_size = self.parse_float("mesh size", self.mesh_var.get())
            geometry = create_intrinsic_geometry(mesh_size=mesh_size)
        except Exception as exc:
            messagebox.showerror("Geometry Error", str(exc), parent=self)
            return
        self.geometry_data = geometry
        self.mesh_var.set(f"{float(geometry.mesh_size):g}")
        self.mask = np.array(geometry.mask, dtype=bool)
        self.boundary_assignments = {}
        self.hover_edge_id = None
        self.redraw_geometry()
        self.update_status()

    def edit_initial_conditions(self) -> None:
        spec = ask_initial_condition(self, self.initial_condition)
        if spec is not None:
            self.initial_condition = spec
            self.update_status()

    def edit_external_generation(self) -> None:
        current = getattr(self, "_external_generation", ExternalGenerationSpec())
        spec = ask_external_generation(self, current)
        if spec is not None:
            self._external_generation = spec
            self.update_status()

    def fill_unassigned_reflective(self) -> None:
        if not self.geometry_data:
            return
        for edge in self.geometry_data.edges:
            if edge.edge_id not in self.boundary_assignments:
                self.boundary_assignments[edge.edge_id] = BoundaryCondition(kind="reflective")
        self.redraw_geometry()
        self.update_status()

    def edge_color(self, edge_id: str) -> str:
        bc = self.boundary_assignments.get(edge_id)
        if bc is None:
            return "#AA0000"
        kind = bc.kind.lower()
        if kind == "reflective":
            return "#1155AA"
        if kind == "neumann":
            return "#CC7A00"
        if kind == "dirichlet":
            return "#008844"
        if kind == "absorbing":
            return "#333333"
        if kind == "robin":
            return "#7A4A00"
        return "#555555"

    def redraw_geometry(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#F6F6F6")
        if self.mask is None or self.geometry_data is None:
            self.ax.set_title("Import geometry to begin")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        display = np.where(self.mask, 1.0, np.nan)
        self.ax.imshow(display, origin="lower", cmap="Greys", alpha=0.28)
        for edge in self.geometry_data.edges:
            self.ax.plot(
                [edge.x0, edge.x1],
                [edge.y0, edge.y1],
                color=self.edge_color(edge.edge_id),
                linewidth=2.2,
                solid_capstyle="butt",
            )

        if self.hover_edge_id is not None:
            edge = next((e for e in self.geometry_data.edges if e.edge_id == self.hover_edge_id), None)
            if edge is not None:
                self.ax.plot(
                    [edge.x0, edge.x1],
                    [edge.y0, edge.y1],
                    color="#FFD500",
                    linewidth=4.0,
                    solid_capstyle="round",
                )

        self.ax.set_title(
            f"{self.geometry_data.name} | layer={self.geometry_data.layer} | edges={len(self.geometry_data.edges)}"
        )
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (mesh index)")
        self.ax.set_ylabel("y (mesh index)")
        self.ax.grid(False)
        self.canvas.draw_idle()

    def nearest_edge(self, x: float, y: float) -> str | None:
        if self.geometry_data is None:
            return None
        best_edge = None
        best_dist = float("inf")
        for edge in self.geometry_data.edges:
            dist = point_to_segment_distance(x, y, edge)
            if dist < best_dist:
                best_dist = dist
                best_edge = edge.edge_id
        if best_dist <= 0.45:
            return best_edge
        return None

    def on_motion(self, event) -> None:
        if self.geometry_data is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            if self.hover_edge_id is not None:
                self.hover_edge_id = None
                self.hover_label.configure(text="Hover edge: none")
                self.redraw_geometry()
            return
        candidate = self.nearest_edge(float(event.xdata), float(event.ydata))
        if candidate != self.hover_edge_id:
            self.hover_edge_id = candidate
            if candidate is None:
                self.hover_label.configure(text="Hover edge: none")
            else:
                self.hover_label.configure(text=f"Hover edge: {candidate} (click to assign)")
            self.redraw_geometry()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or self.geometry_data is None:
            return
        if self.hover_edge_id is None:
            return
        current = self.boundary_assignments.get(self.hover_edge_id)
        chosen = ask_boundary_condition(self, current)
        if chosen is None:
            return
        self.boundary_assignments[self.hover_edge_id] = chosen
        self.redraw_geometry()
        self.update_status()

    def update_status(self) -> None:
        if self.geometry_data is None:
            self.status_label.configure(text="No geometry loaded.")
            self.run_btn.configure(state="disabled")
            return
        assigned = len(self.boundary_assignments)
        total = len(self.geometry_data.edges)
        diffusion_enabled = self.enable_diffusion_var.get()
        boundary_ready = assigned == total if diffusion_enabled else True
        setup_ready = total > 0 and boundary_ready
        initial_kind = self.initial_condition.kind
        boundary_note = (
            "Click any highlighted edge to assign boundary conditions."
            if diffusion_enabled
            else "Boundary conditions are optional while diffusion is disabled."
        )
        self.status_label.configure(
            text=(
                f"Geometry: {self.geometry_data.name}\n"
                f"Boundary assignment: {assigned}/{total}\n"
                f"Initial condition: {initial_kind}\n"
                f"{boundary_note}"
            )
        )
        run_state = "disabled" if self.simulation_running else ("normal" if setup_ready else "disabled")
        self.run_btn.configure(state=run_state)

    def build_setup(self) -> SetupData:
        if self.geometry_data is None:
            raise ValueError("No geometry loaded.")
        mesh_value = self.parse_float("mesh size", self.mesh_var.get())
        geometry_mesh = float(self.geometry_data.mesh_size)
        if abs(mesh_value - geometry_mesh) > 1e-12:
            raise ValueError(
                "Mesh Size must match the imported geometry mesh size "
                f"({geometry_mesh:g}). Re-import geometry after changing mesh size."
            )
        params = SimulationParameters(
            diffusion_coefficient=self.parse_float("diffusion coefficient", self.diff_var.get()),
            dt=self.parse_float("dt", self.dt_var.get()),
            total_time=self.parse_float("total time", self.total_var.get()),
            mesh_size=geometry_mesh,
            store_every=max(1, self.parse_int("store_every", self.store_var.get())),
            energy_gap=self.parse_float("energy gap", self.energy_gap_var.get()),
            energy_min_factor=self.parse_float("energy min factor", self.energy_min_var.get()),
            energy_max_factor=self.parse_float("energy max factor", self.energy_max_var.get()),
            num_energy_bins=max(1, self.parse_int("energy bins", self.energy_bins_var.get())),
            dynes_gamma=self.parse_float("Dynes gamma", self.dynes_gamma_var.get()),
            gap_expression=self.gap_expression_var.get().strip(),
            collision_solver=self.collision_solver_var.get().strip() or "forward_euler",
            enable_diffusion=self.enable_diffusion_var.get(),
            enable_recombination=self.enable_recombination_var.get(),
            enable_scattering=self.enable_scattering_var.get(),
            tau_s=self.parse_float("τ₀,s", self.tau_s_var.get()),
            tau_r=self.parse_float("τ₀,r", self.tau_r_var.get()),
            T_c=self.parse_float("T_c", self.T_c_var.get()),
            bath_temperature=self.parse_float("bath temperature", self.bath_temp_var.get()),
            external_generation=self._external_generation,
        )
        return SetupData(
            setup_id=self.setup_id,
            name=self.name_var.get().strip() or "Untitled Setup",
            created_at=utc_now_iso(),
            geometry=self.geometry_data,
            boundary_conditions=dict(self.boundary_assignments),
            parameters=params,
            initial_condition=self.initial_condition,
        )

    def save_setup_file(self) -> None:
        try:
            setup = self.build_setup()
            path = save_setup(setup)
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc), parent=self)
            return
        self.last_saved_path = str(path)
        messagebox.showinfo("Setup Saved", f"Saved setup:\n{path}", parent=self)

    def _run_background_task(
        self,
        title: str,
        message: str,
        task_fn,
        on_success,
    ) -> None:
        self.simulation_running = True
        self.update_status()
        dialog = BusyDialog(self, title=title, message=message)
        result_queue: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                value = task_fn()
            except Exception as exc:
                result_queue.put(("error", exc))
            else:
                result_queue.put(("ok", value))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            try:
                kind, payload = result_queue.get_nowait()
            except queue.Empty:
                if dialog.winfo_exists():
                    self.after(100, poll)
                return

            dialog.close()
            self.simulation_running = False
            self.update_status()
            if kind == "error":
                messagebox.showerror(f"{title} Failed", str(payload), parent=self)
                return
            on_success(payload)

        self.after(100, poll)

    def run_simulation(self) -> None:
        if self.geometry_data is None or self.mask is None:
            messagebox.showerror("Cannot Run", "Import geometry first.", parent=self)
            return
        if self.simulation_running:
            return
        try:
            setup = self.build_setup()
            if (
                setup.parameters.enable_diffusion
                and len(setup.boundary_conditions) != len(setup.geometry.edges)
            ):
                raise BoundaryAssignmentError("All edges must be assigned before running the simulation.")
        except Exception as exc:
            messagebox.showerror("Simulation Failed", str(exc), parent=self)
            return

        mask_snapshot = self.mask.copy()

        # Auto-load precomputed arrays if sidecar exists and is compatible
        precomp_data = None
        if self.last_saved_path:
            from pathlib import Path
            sp = Path(self.last_saved_path)
            if precomputed_exists(sp):
                try:
                    candidate = load_precomputed(sp)
                    mismatch = validate_precomputed(candidate, setup.parameters, mask_snapshot)
                    if mismatch:
                        messagebox.showwarning(
                            "Stale Pre-computed Data",
                            f"Ignoring precomputed sidecar — parameters changed:\n{mismatch}\n\n"
                            "Re-run Pre-compute Arrays to update.",
                            parent=self,
                        )
                    else:
                        precomp_data = candidate
                except Exception:
                    pass

        def task_fn():
            initial = build_initial_field(mask_snapshot, setup.initial_condition)
            # Compute energy weights for Fermi-Dirac initial condition
            e_weights = None
            p = setup.parameters
            if (
                setup.initial_condition.kind.lower() == "fermi_dirac"
                and p.energy_gap > 0
            ):
                E_bins, _ = build_energy_grid(
                    p.energy_gap,
                    p.energy_min_factor,
                    p.energy_max_factor,
                    p.num_energy_bins,
                )
                temp_K = float(setup.initial_condition.params.get("temperature", 0.1))
                e_weights = thermal_qp_weights(E_bins, p.energy_gap, temp_K, p.dynes_gamma)
            times, frames, mass, color_limits, energy_frames, energy_bins = run_2d_crank_nicolson(
                mask=mask_snapshot,
                edges=setup.geometry.edges,
                edge_conditions=setup.boundary_conditions,
                initial_field=initial,
                diffusion_coefficient=p.diffusion_coefficient,
                dt=p.dt,
                total_time=p.total_time,
                dx=p.mesh_size,
                store_every=p.store_every,
                energy_gap=p.energy_gap,
                energy_min_factor=p.energy_min_factor,
                energy_max_factor=p.energy_max_factor,
                num_energy_bins=p.num_energy_bins,
                energy_weights=e_weights,
                enable_diffusion=p.enable_diffusion,
                enable_recombination=p.enable_recombination,
                enable_scattering=p.enable_scattering,
                dynes_gamma=p.dynes_gamma,
                collision_solver=p.collision_solver,
                tau_0=p.tau_0,
                tau_s=p.tau_s,
                tau_r=p.tau_r,
                T_c=p.T_c,
                bath_temperature=p.bath_temperature,
                external_generation=p.external_generation,
                gap_expression=p.gap_expression,
                precomputed=precomp_data,
            )
            serialized_energy_frames = None
            serialized_energy_bins = None
            if energy_frames is not None:
                serialized_energy_frames = [
                    [frame_to_jsonable(eframe) for eframe in time_slice]
                    for time_slice in energy_frames
                ]
                serialized_energy_bins = energy_bins.tolist()
            result = SimulationResultData(
                simulation_id=create_simulation_id(),
                setup_id=setup.setup_id,
                setup_name=setup.name,
                created_at=utc_now_iso(),
                times=[float(t) for t in times],
                frames=[frame_to_jsonable(frame) for frame in frames],
                mass_over_time=[float(v) for v in mass],
                color_limits=[float(color_limits[0]), float(color_limits[1])],
                metadata={
                    "diffusion_coefficient": setup.parameters.diffusion_coefficient,
                    "mesh_size": setup.parameters.mesh_size,
                    "dt": setup.parameters.dt,
                    "total_time": setup.parameters.total_time,
                    "energy_gap": setup.parameters.energy_gap,
                },
                energy_frames=serialized_energy_frames,
                energy_bins=serialized_energy_bins,
            )
            save_error: str | None = None
            path: str | None = None
            try:
                path = str(save_simulation(result))
            except Exception as exc:
                save_error = str(exc)
            return result, path, save_error

        def on_success(payload) -> None:
            result, path, save_error = payload
            self.latest_result = result
            if save_error is not None:
                messagebox.showwarning("Saved In Memory Only", f"Simulation ran but save failed:\n{save_error}", parent=self)
            if self.auto_live_var.get():
                self.open_live_viewer()
                return
            if path is None:
                messagebox.showinfo("Simulation Complete", "Simulation finished.", parent=self)
                return
            messagebox.showinfo("Simulation Complete", f"Simulation saved:\n{path}", parent=self)

        self._run_background_task(
            title="Running Simulation",
            message="Running Crank-Nicolson simulation and saving results...",
            task_fn=task_fn,
            on_success=on_success,
        )

    def open_live_viewer(self) -> None:
        if self.latest_result is None:
            messagebox.showerror("No Result", "Run a simulation first.", parent=self)
            return
        SimulationViewer(self, self.latest_result)

class QuasiparticleMainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quasiparticle Physics Simulator")
        self.geometry("980x640")
        self.minsize(900, 560)
        apply_retro_theme(self)
        ensure_data_dirs()

        self.start_frame = tk.Frame(self, bg=RETRO_BG)
        self.menu_frame = tk.Frame(self, bg=RETRO_BG)
        self.start_frame.pack(fill="both", expand=True)

        self.build_start_frame()
        self.build_menu_frame()

    def build_start_frame(self) -> None:
        frame = self.start_frame
        tk.Label(
            frame,
            text="Quasiparticle Physics Simulator",
            font=FONT_TITLE,
            fg=RETRO_ACCENT,
            bg=RETRO_BG,
        ).pack(pady=(80, 12))
        tk.Label(
            frame,
            text="2D Crank-Nicolson diffusion workflow with GDS geometry and boundary condition assignment",
            bg=RETRO_BG,
        ).pack(pady=(0, 20))

        tk.Button(
            frame,
            text="Run Quasiparticle Physics Simulator",
            width=42,
            height=2,
            command=self.show_menu,
        ).pack(pady=8)
        tk.Button(
            frame,
            text="Install Requirements.bat",
            width=30,
            command=self.run_install_requirements,
        ).pack(pady=6)
        tk.Label(
            frame,
            text=f"Workspace: {ROOT_DIR}",
            bg=RETRO_BG,
            justify="center",
            font=FONT_MONO,
        ).pack(side="bottom", pady=16)

    def build_menu_frame(self) -> None:
        frame = self.menu_frame
        tk.Label(frame, text="Launch Menu", font=FONT_TITLE, fg=RETRO_ACCENT, bg=RETRO_BG).pack(pady=(28, 16))

        menu = tk.Frame(frame, bg=RETRO_BG)
        menu.pack(pady=12)
        tk.Button(menu, text="Create New Setup", width=32, command=self.open_new_setup).pack(pady=6)
        tk.Button(menu, text="Load Saved Setup", width=32, command=self.load_saved_setup).pack(pady=6)
        tk.Button(menu, text="View Saved Simulations", width=32, command=self.view_saved_simulations).pack(pady=6)
        tk.Button(menu, text="Generate Test Simulations", width=32, command=self.generate_tests).pack(pady=6)
        tk.Button(menu, text="View Test Simulations", width=32, command=self.view_tests).pack(pady=6)
        tk.Button(menu, text="Back To Start Screen", width=32, command=self.show_start).pack(pady=(20, 6))

    def show_menu(self) -> None:
        self.start_frame.pack_forget()
        self.menu_frame.pack(fill="both", expand=True)

    def show_start(self) -> None:
        self.menu_frame.pack_forget()
        self.start_frame.pack(fill="both", expand=True)

    def run_install_requirements(self) -> None:
        batch_path = ROOT_DIR / "Install Requirements.bat"
        if not batch_path.exists():
            messagebox.showerror("Missing Script", f"{batch_path} not found.", parent=self)
            return
        subprocess.Popen(["cmd", "/c", "start", "", str(batch_path)], cwd=str(ROOT_DIR))

    def open_new_setup(self) -> None:
        SetupEditor(self)

    def load_saved_setup(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Load Setup JSON",
            initialdir=str(ROOT_DIR / "data" / "setups"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            setup = load_setup(path)
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc), parent=self)
            return
        SetupEditor(self, setup=setup)

    def view_saved_simulations(self) -> None:
        files = list_simulation_files()
        if not files:
            messagebox.showinfo("No Simulations", "No saved simulations found.", parent=self)
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Open Simulation JSON",
            initialdir=str(SIMULATIONS_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result = load_simulation(path)
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc), parent=self)
            return
        SimulationViewer(self, result)

    def _run_background_task(
        self,
        title: str,
        message: str,
        task_fn,
        on_success,
    ) -> None:
        dialog = BusyDialog(self, title=title, message=message)
        result_queue: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                value = task_fn()
            except Exception as exc:
                result_queue.put(("error", exc))
            else:
                result_queue.put(("ok", value))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            try:
                kind, payload = result_queue.get_nowait()
            except queue.Empty:
                if dialog.winfo_exists():
                    self.after(100, poll)
                return

            dialog.close()
            if kind == "error":
                messagebox.showerror(f"{title} Failed", str(payload), parent=self)
                return
            on_success(payload)

        self.after(100, poll)

    def generate_tests(self) -> None:
        def on_success(result) -> None:
            suite, path = result
            total_cases = sum(len(group.cases) for group in suite.geometry_groups) if suite.geometry_groups else len(suite.cases)
            messagebox.showinfo(
                "Test Suite Generated",
                f"Generated {total_cases} cases across {len(suite.geometry_groups) or 1} geometry set(s).\nSaved to:\n{path}",
                parent=self,
            )

        self._run_background_task(
            title="Generating Test Simulations",
            message="Running numerical simulations and writing test-suite files...",
            task_fn=generate_and_save_test_suite,
            on_success=on_success,
        )

    def view_tests(self) -> None:
        path = latest_test_suite_file()
        if path is None:
            messagebox.showinfo("No Test Suite", "Generate test simulations first.", parent=self)
            return

        def on_success(suite: TestSuiteData) -> None:
            TestGeometryLanding(self, suite, suite_path=str(path))

        self._run_background_task(
            title="Loading Test Simulations",
            message=f"Loading test suite:\n{path.name}",
            task_fn=lambda: load_test_suite(path, load_group_cases=False),
            on_success=on_success,
        )


def run_app() -> None:
    app = QuasiparticleMainApp()
    app.mainloop()
