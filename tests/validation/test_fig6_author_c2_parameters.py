from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from qpsim.constants import KB_UEV_PER_K
from validation.fischer_2023 import fig6_author_c2_parameters as c2_parameters
from validation.fischer_2023 import fig6_solve
from validation.fischer_2023.fig6_author_c2_parameters import (
    QPSIM_FIG6_C_PHOTON_NS_INV,
    QPSIM_FIG6_DELTA0_UEV,
    QPSIM_FIG6_TAU0_NS,
    QPSIM_FIG6_TAU0_PB_NS,
    QPSIM_FIG6_TAU_L_NS,
    SCHEMA,
    C2ParameterError,
    author_solve_parameters_from_native,
    build_c2_parameter_plan,
    qpsim_fig6_configuration_endpoint,
    qpsim_fig6_critical_temperature,
    qpsim_fig6_finite_cutoff_ratio,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_C0_SUMMARY = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c0-author-equivalent-score.json"
)


def _parent_parameters() -> dict[str, object]:
    value = json.loads(DEFAULT_C0_SUMMARY.read_text(encoding="utf-8"))
    parameters = dict(value["parameters"])
    assert isinstance(parameters, dict)
    parameters["thermal_gap_eV"] = value["observable"]["thermal_gap_eV"]
    return parameters


def test_c2a_preserves_author_effective_energy_bits() -> None:
    plan = build_c2_parameter_plan(_parent_parameters())
    mappings = {mapping.name: mapping for mapping in plan.energy_mappings}

    assert mappings["gap"].source_eV.hex() == "0x1.797cc39ffd60ep-13"
    assert mappings["gap"].native_ueV.hex() == "0x1.67fffffffffffp+7"
    assert mappings["gap"].author_effective_eV == mappings["gap"].source_eV
    assert mappings["gap"].round_trip_bit_exact is True
    assert mappings["gap"].native_round_trip_eV == mappings["gap"].source_eV

    assert mappings["delta0"].source_eV == mappings["gap"].source_eV
    assert mappings["delta0"].round_trip_bit_exact is True
    assert mappings["h"].native_ueV == 1.0
    assert mappings["h"].round_trip_bit_exact is True


def test_c2a_time_and_rate_views_are_explicit_and_reversible() -> None:
    plan = build_c2_parameter_plan(_parent_parameters())
    times = {mapping.name: mapping for mapping in plan.time_mappings}

    assert times["tau_0"].native_ns == 438.0
    assert times["tau_0"].round_trip_bit_exact is True
    assert times["tau_0_pb"].native_ns == 0.2553360200613721
    assert times["tau_0_pb"].round_trip_bit_exact is True
    assert times["tau_l"].native_ns == 0.255
    assert times["tau_l"].round_trip_bit_exact is True

    rate = plan.photon_rate_mapping
    assert rate.native_ns_inv == 1.0000352684367705e-9
    assert rate.round_trip_bit_exact is True
    assert plan.c2a.kB_ueV_per_K == 86.17330341520024
    assert plan.c2a.eq35_t_star_over_delta == 0.3399078973729434


def test_c2b_steps_change_only_the_declared_parameter_fields() -> None:
    plan = build_c2_parameter_plan(_parent_parameters())
    previous = plan.c2a

    expected = (
        ("delta0_ueV", "gap_ueV"),
        ("kB_ueV_per_K",),
        ("c_photon_ns_inv",),
        ("tau_0_pb_ns",),
        ("T_c_K",),
    )
    for step, expected_fields in zip(plan.c2b_steps, expected, strict=True):
        before = previous.as_record()
        after = step.parameters.as_record()
        actual_fields = tuple(
            sorted(
                key
                for key in before
                if key not in {"eq35_t_star_over_delta", "hex", "omega_0_ueV"}
                and before[key] != after[key]
            )
        )
        assert actual_fields == tuple(sorted(expected_fields))
        assert tuple(sorted(step.changed_fields)) == tuple(sorted(expected_fields))
        previous = step.parameters


def test_c2b_endpoint_is_current_explicit_qpsim_fig6_mapping() -> None:
    plan = build_c2_parameter_plan(_parent_parameters())
    endpoint = plan.c2b

    assert endpoint.gap_ueV == endpoint.delta0_ueV == QPSIM_FIG6_DELTA0_UEV
    assert endpoint.gap_ueV.hex() == "0x1.6800000000000p+7"
    assert endpoint.h_ueV == 1.0
    assert endpoint.omega_0_ueV == 20.0
    assert endpoint.temperature_K == 0.2
    assert endpoint.tau_0_ns == 438.0
    assert endpoint.tau_0_pb_ns == QPSIM_FIG6_TAU0_PB_NS == 0.255
    assert endpoint.tau_l_ns == 0.255
    assert endpoint.c_photon_ns_inv == QPSIM_FIG6_C_PHOTON_NS_INV == 1e-9
    assert endpoint.n_bar == 954548.4566618347
    assert endpoint.kB_ueV_per_K == KB_UEV_PER_K

    assert qpsim_fig6_finite_cutoff_ratio() == fig6_solve.FINITE_CUTOFF_DELTA0_OVER_KBTC
    assert endpoint.T_c_K == fig6_solve.T_C
    assert endpoint.T_c_K == qpsim_fig6_critical_temperature(180.0)
    assert endpoint.T_c_K.hex() == "0x1.2f2ee323c27ffp+0"
    assert endpoint.eq35_t_star_over_delta == 0.3399503360830364
    assert (
        endpoint.eq35_t_star_over_delta
        == fig6_solve._kBTstar_eq35(endpoint.n_bar) / fig6_solve.DELTA_0
    )


def test_live_fig6_endpoint_extractor_reads_declarations_and_built_grid() -> None:
    live = qpsim_fig6_configuration_endpoint()
    E, dE, spectral = fig6_solve._build_grid_and_spectral()

    assert live.delta0_ueV == fig6_solve.DELTA_0
    assert live.tau_0_ns == fig6_solve.TAU_0
    assert live.tau_0_pb_ns == fig6_solve.PAPER_TAU_0_PB_PS / 1000.0
    assert live.tau_l_ns == fig6_solve.PAPER_TAU_0_PB_PS / 1000.0
    assert live.tau_l_model == fig6_solve.TAU_L_MODEL == "tau_0_pb"
    assert live.c_photon_ns_inv == fig6_solve.C_PHOT
    assert live.omega_0_ueV == fig6_solve.OMEGA_0
    assert live.grid_num_bins == E.size == fig6_solve.NUM_BINS
    assert live.grid_spacing_ueV == dE[0] == 1.0
    assert all(float(width) == live.grid_spacing_ueV for width in dE)
    assert live.photon_bin == round(live.omega_0_ueV / live.grid_spacing_ueV) == 20
    assert live.photon_bin * live.grid_spacing_ueV == live.omega_0_ueV
    assert live.energy_min_factor == fig6_solve.E_MIN_FACTOR
    assert live.energy_max_factor == fig6_solve.E_MAX_FACTOR
    assert E[0] - 0.5 * live.grid_spacing_ueV == 160.0
    assert E[-1] + 0.5 * live.grid_spacing_ueV == 1800.0
    assert spectral.gap == live.delta0_ueV
    assert live.T_c_K == fig6_solve.T_C
    assert live.finite_cutoff_ratio == fig6_solve.FINITE_CUTOFF_DELTA0_OVER_KBTC
    assert live.gap_solve_xtol_ueV == fig6_solve.GAP_SOLVE_XTOL_UEV


@pytest.mark.parametrize(
    ("field_name", "drifted_value", "error_label"),
    (
        ("delta0_ueV", 181.0, "Delta_0"),
        ("tau_0_ns", 437.0, "tau_0"),
        ("tau_0_pb_ns", 0.254, "tau_0_pb"),
        ("tau_l_ns", 0.254, "tau_l"),
        ("c_photon_ns_inv", 2.0e-9, "photon coupling"),
        ("omega_0_ueV", 21.0, "photon energy"),
        ("grid_spacing_ueV", 0.5, "grid spacing"),
        ("photon_bin", 21, "photon bin"),
        ("T_c_K", 1.2, "critical temperature"),
    ),
)
def test_c2_plan_rejects_live_fig6_endpoint_drift(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    drifted_value: float | int,
    error_label: str,
) -> None:
    live = qpsim_fig6_configuration_endpoint()
    drifted = replace(live, **{field_name: drifted_value})
    monkeypatch.setattr(
        c2_parameters,
        "qpsim_fig6_configuration_endpoint",
        lambda: drifted,
    )

    with pytest.raises(C2ParameterError, match=error_label):
        build_c2_parameter_plan(_parent_parameters())


def test_live_fig6_endpoint_rejects_nonpaper_tau_l_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "TAU_L_MODEL", "acoustic_escape")
    qpsim_fig6_configuration_endpoint.cache_clear()
    try:
        with pytest.raises(C2ParameterError, match="tau_l model"):
            qpsim_fig6_configuration_endpoint()
    finally:
        # Leave no cached endpoint derived under the monkeypatched module
        # configuration after pytest restores the source declaration.
        qpsim_fig6_configuration_endpoint.cache_clear()


def test_c2_endpoint_binding_never_loads_a_material_or_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_material_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("C2 semantic binding must not load a material/YAML.")

    monkeypatch.setattr(fig6_solve, "load_material", fail_material_load)
    monkeypatch.setattr(fig6_solve, "_fischer_material", fail_material_load)
    qpsim_fig6_configuration_endpoint.cache_clear()
    try:
        plan = build_c2_parameter_plan(_parent_parameters())
        assert plan.c2b.delta0_ueV == QPSIM_FIG6_DELTA0_UEV
    finally:
        qpsim_fig6_configuration_endpoint.cache_clear()


def test_reviewed_fig6_literals_match_the_live_semantic_endpoint() -> None:
    live = qpsim_fig6_configuration_endpoint()

    assert live.delta0_ueV == QPSIM_FIG6_DELTA0_UEV
    assert live.tau_0_ns == QPSIM_FIG6_TAU0_NS
    assert live.tau_0_pb_ns == QPSIM_FIG6_TAU0_PB_NS
    assert live.tau_l_ns == QPSIM_FIG6_TAU_L_NS
    assert live.c_photon_ns_inv == QPSIM_FIG6_C_PHOTON_NS_INV


def test_author_operator_carriers_preserve_c2a_and_map_c2b_explicitly() -> None:
    parent = _parent_parameters()
    plan = build_c2_parameter_plan(parent)

    c2a = plan.c2a_author_effective
    for name in (
        "gap_eV",
        "h_eV",
        "T_c_K",
        "tau_0_s",
        "tau_0_pb_s",
        "tau_l_s",
        "n_bar",
        "c_photon_s_inv",
        "delta0_eV",
        "thermal_gap_eV",
        "relative_step_threshold",
    ):
        expected = parent[name]
        actual = getattr(c2a, name)
        assert isinstance(expected, float)
        assert actual.hex() == expected.hex()
    assert (
        c2a.constants.boltzmann_constant_J_per_K.hex() == parent["boltzmann_constant_J_per_K"].hex()
    )
    assert c2a.constants.electron_charge_C.hex() == parent["electron_charge_C"].hex()

    effective_steps = dict(plan.c2b_author_effective_steps())
    native_energy = effective_steps["C2b1-explicit-energy-literals"]
    assert native_energy.gap_eV.hex() == parent["gap_eV"].hex()
    assert native_energy.delta0_eV.hex() == parent["delta0_eV"].hex()
    modern_kB = effective_steps["C2b2-modern-kB"]
    assert modern_kB.gap_eV.hex() == native_energy.gap_eV.hex()
    assert modern_kB.tau_0_pb_s.hex() == native_energy.tau_0_pb_s.hex()
    assert (
        modern_kB.constants.boltzmann_constant_J_per_K.hex()
        != native_energy.constants.boltzmann_constant_J_per_K.hex()
    )

    endpoint = author_solve_parameters_from_native(
        plan.c2b,
        parent=plan.parent_author_parameters,
    )
    assert endpoint.gap_eV == 180.0 * 1e-6
    assert endpoint.delta0_eV == 180.0 * 1e-6
    assert endpoint.h_eV == 1.0e-6
    assert endpoint.tau_0_s == 438.0 * 1e-9
    assert endpoint.tau_0_pb_s == 0.255 * 1e-9
    assert endpoint.tau_l_s == 0.255 * 1e-9
    assert endpoint.c_photon_s_inv == 1.0
    assert (
        endpoint.constants.electron_charge_C.hex()
        == plan.parent_author_parameters.constants.electron_charge_C.hex()
    )
    assert endpoint.constants.boltzmann_constant_J_per_K == 1.380648987935705e-23
    recovered = (
        endpoint.constants.boltzmann_constant_J_per_K / endpoint.constants.electron_charge_C * 1e6
    )
    assert recovered.hex() == KB_UEV_PER_K.hex()


def test_c2_plan_record_is_provenance_bound_and_does_not_claim_completion() -> None:
    record = build_c2_parameter_plan(_parent_parameters()).as_record()
    assert record["schema"] == SCHEMA
    assert record["no_implicit_material_or_yaml_defaults"] is True
    assert record["c2b"]["frozen_operator_differential_can_complete_parameter_stage"]
    assert record["c2b"]["nonlinear_resolve_required_before_root_or_ordinate_claim"]
    assert record["c2b"]["nonlinear_resolve_stage"] == "later nonlinear-solver stage"
    assert "before claiming a C2b root or ordinate" in record["c2b"]["qualification"]

    sources = record["sources"]
    assert set(sources) == {
        "qpsim/constants.py",
        "qpsim/physics/gap_equation.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    }
    assert all(len(digest) == 64 for digest in sources.values())

    # The record must be strict-JSON serializable without an evidence writer
    # silently replacing NaN or Infinity.
    json.dumps(record, allow_nan=False, sort_keys=True)


def test_c2_parent_missing_or_changed_required_values_fails_loudly() -> None:
    missing = _parent_parameters()
    missing.pop("tau_0_pb_s")
    with pytest.raises(C2ParameterError, match="tau_0_pb_s"):
        build_c2_parameter_plan(missing)

    changed_tau = _parent_parameters()
    changed_tau["tau_0_s"] = 437e-9
    with pytest.raises(C2ParameterError, match="C2 inherited tau_0"):
        build_c2_parameter_plan(changed_tau)
