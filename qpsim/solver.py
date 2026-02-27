from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .models import (
    BoundaryCondition,
    EdgeSegment,
    ExternalGenerationSpec,
    normalize_collision_solver_name,
)
from .safe_eval import compile_safe_expression


class BoundaryAssignmentError(ValueError):
    pass


_DIR_OFFSETS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _normalized_bc(bc: BoundaryCondition) -> BoundaryCondition:
    return replace(bc, kind=bc.normalized_kind())


def _build_face_bc_lookup(
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
) -> dict[tuple[int, int, str], BoundaryCondition]:
    lookup: dict[tuple[int, int, str], BoundaryCondition] = {}
    for edge in edges:
        bc = edge_conditions.get(edge.edge_id)
        if bc is None:
            continue
        checked = _normalized_bc(bc)
        checked.validate()
        for face in edge.faces:
            lookup[(face.row, face.col, face.direction)] = checked
    return lookup


def _mask_to_index(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    index_map = -np.ones(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    for idx, (row, col) in enumerate(coords):
        index_map[row, col] = idx
    return index_map, [tuple(map(int, rc)) for rc in coords]


def build_energy_grid(
    gap: float,
    energy_min_factor: float,
    energy_max_factor: float,
    num_energy_bins: int,
) -> tuple[np.ndarray, float]:
    """Build a cell-centered energy grid and its integration bin width."""
    if gap <= 0:
        raise ValueError("gap must be positive.")
    if num_energy_bins <= 0:
        raise ValueError("num_energy_bins must be >= 1.")

    E_min = energy_min_factor * gap
    E_max = energy_max_factor * gap
    if num_energy_bins == 1:
        # Keep legacy single-bin behavior by using a unit integration weight.
        center = 0.5 * (E_min + E_max)
        return np.array([center], dtype=float), 1.0
    if E_max <= E_min:
        raise ValueError("energy_max_factor must be > energy_min_factor for num_energy_bins > 1.")

    dE = (E_max - E_min) / float(num_energy_bins)
    E_bins = E_min + (np.arange(num_energy_bins, dtype=float) + 0.5) * dE
    return E_bins, dE


def _apply_boundary_contribution(
    bc: BoundaryCondition,
    row_idx: int,
    inv_dx2: float,
    inv_dx: float,
    rows: list[int],
    cols: list[int],
    data: list[float],
    source: np.ndarray,
) -> None:
    kind = bc.normalized_kind()
    if kind == "reflective":
        return
    if kind == "absorbing":
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-2.0 * inv_dx2)
        return
    if kind == "dirichlet":
        g = float(bc.value or 0.0)
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-2.0 * inv_dx2)
        source[row_idx] += 2.0 * g * inv_dx2
        return
    if kind == "neumann":
        qn = float(bc.value or 0.0)
        source[row_idx] += qn * inv_dx
        return
    if kind == "robin":
        beta = float(bc.value or 0.0)
        gamma = float(bc.aux_value or 0.0)
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-beta * inv_dx)
        source[row_idx] += gamma * inv_dx
        return
    raise BoundaryAssignmentError(f"Unsupported boundary kind: {bc.kind}")


def build_laplacian_with_boundaries(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    dx: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    index_map, coords = _mask_to_index(mask)
    n = len(coords)
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")

    face_bc = _build_face_bc_lookup(edges, edge_conditions)
    missing_edges = [edge.edge_id for edge in edges if edge.edge_id not in edge_conditions]
    if missing_edges:
        raise BoundaryAssignmentError(
            f"All edges must be assigned boundary conditions before simulation. Missing: {len(missing_edges)}"
        )

    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    source = np.zeros(n, dtype=float)

    ny, nx = mask.shape
    for p, (row, col) in enumerate(coords):
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                q = int(index_map[nr, nc])
                rows.append(p)
                cols.append(p)
                data.append(-inv_dx2)
                rows.append(p)
                cols.append(q)
                data.append(inv_dx2)
            else:
                bc = face_bc.get((row, col, direction))
                if bc is None:
                    raise BoundaryAssignmentError(
                        f"Missing boundary condition for face at cell ({row}, {col}) direction '{direction}'."
                    )
                _apply_boundary_contribution(
                    bc=bc,
                    row_idx=p,
                    inv_dx2=inv_dx2,
                    inv_dx=inv_dx,
                    rows=rows,
                    cols=cols,
                    data=data,
                    source=source,
                )

    laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return laplacian, source, index_map


def reconstruct_field(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    field = np.full(mask.shape, np.nan, dtype=float)
    field[mask] = values
    return field


def _build_cn_operators(
    identity: sparse.spmatrix,
    laplacian: sparse.spmatrix,
    dt_val: float,
    D: float,
) -> tuple:
    """Build Crank-Nicolson A (implicit) and B (explicit) matrices and LU factor."""
    alpha = 0.5 * dt_val * D
    a_mat = (identity - alpha * laplacian).tocsc()
    b_mat = (identity + alpha * laplacian).tocsr()
    lu = spla.splu(a_mat)
    return b_mat, lu


def build_variable_diffusion_laplacian(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    dx: float,
    D_spatial: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a diffusion operator L_D that includes spatially varying D.

    Uses harmonic mean D at cell interfaces:
      D_{k+1/2} = 2*D_k*D_{k+1} / (D_k + D_{k+1})

    This is the standard choice for conservative discretization on staggered
    grids and ensures correct flux continuity at interfaces with varying D.
    The reference PDF (Eq. 34) shows an arithmetic mean; we intentionally use
    the harmonic mean for better flux continuity across sharp D contrasts.

    Returns (L_D, source) where the CN step uses:
      A = I - 0.5*dt*L_D,  B = I + 0.5*dt*L_D
      rhs = B @ u + dt * source

    Parameters
    ----------
    D_spatial : 1D array of length n_interior, diffusion coefficient per pixel.
    """
    if dx <= 0:
        raise ValueError("dx must be positive.")

    index_map, coords = _mask_to_index(mask)
    n = len(coords)
    face_bc = _build_face_bc_lookup(edges, edge_conditions)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dx = 1.0 / dx

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    source = np.zeros(n, dtype=float)
    ny, nx = mask.shape

    for p, (row, col) in enumerate(coords):
        D_p = D_spatial[p]
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                q = int(index_map[nr, nc])
                D_q = D_spatial[q]
                # Harmonic mean
                D_face = 2.0 * D_p * D_q / max(D_p + D_q, 1e-30)
                rows.append(p)
                cols.append(p)
                data.append(-D_face * inv_dx2)
                rows.append(p)
                cols.append(q)
                data.append(D_face * inv_dx2)
            else:
                bc = face_bc.get((row, col, direction))
                if bc is None:
                    raise BoundaryAssignmentError(
                        f"Missing boundary condition for face at cell ({row}, {col}) direction '{direction}'."
                    )
                kind = bc.normalized_kind()
                if kind == "reflective":
                    pass
                elif kind == "absorbing":
                    rows.append(p)
                    cols.append(p)
                    data.append(-2.0 * D_p * inv_dx2)
                elif kind == "dirichlet":
                    g = float(bc.value or 0.0)
                    rows.append(p)
                    cols.append(p)
                    data.append(-2.0 * D_p * inv_dx2)
                    source[p] += 2.0 * D_p * g * inv_dx2
                elif kind == "neumann":
                    qn = float(bc.value or 0.0)
                    source[p] += D_p * qn * inv_dx
                elif kind == "robin":
                    beta = float(bc.value or 0.0)
                    gamma = float(bc.aux_value or 0.0)
                    rows.append(p)
                    cols.append(p)
                    data.append(-D_p * beta * inv_dx)
                    source[p] += D_p * gamma * inv_dx

    L_D = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return L_D, source


def _bcs_density_of_states(E: np.ndarray, gap: float) -> np.ndarray:
    """BCS density of states: ρ(E) = E / √(E² - Δ²) for E > Δ, else 0."""
    rho = np.zeros_like(E)
    valid = E > gap
    rho[valid] = E[valid] / np.sqrt(E[valid] ** 2 - gap ** 2)
    return rho


def _dynes_density_of_states(E: np.ndarray, gap: float, gamma: float) -> np.ndarray:
    """Dynes density of states: ρ(E) = Re{(E - iΓ) / √((E - iΓ)² - Δ²)}.

    When gamma=0 this reduces to the BCS density of states.
    """
    if gamma <= 0:
        return _bcs_density_of_states(E, gap)
    z = E - 1j * gamma
    with np.errstate(invalid="ignore"):
        result = np.real(z / np.sqrt(z ** 2 - gap ** 2))
    return np.maximum(result, 0.0)


# Boltzmann constant in μeV/K.
# k_B = 8.617333262145e-5 eV/K = 86.17333262145 μeV/K.
_KB_UEV_PER_K = 86.17333262145


def thermal_qp_weights(
    E_bins: np.ndarray,
    gap: float,
    temperature: float,
    dynes_gamma: float = 0.0,
) -> np.ndarray:
    """Thermal equilibrium quasiparticle energy distribution.

    Returns un-normalized weights proportional to N_s(E) × f(E, T) where:
      N_s(E) = Dynes density of states (reduces to BCS when Γ=0)
      f(E, T) = 1 / (exp(E / k_B T) + 1)  (Fermi-Dirac occupation)

    E is the Bogoliubov quasiparticle excitation energy (minimum Δ), *not*
    measured relative to Δ.  The chemical potential of Bogoliubov
    quasiparticles is zero because they are excitations above the BCS ground
    state.

    Parameters
    ----------
    E_bins : array of energy bin centres in μeV
    gap    : superconducting gap Δ in μeV
    temperature : bath temperature in K
    dynes_gamma : Dynes broadening parameter Γ in μeV (0 = BCS)
    """
    rho = _dynes_density_of_states(E_bins, gap, dynes_gamma)
    if temperature <= 0:
        return np.zeros_like(rho)  # T=0: no thermal occupation, n_eq = 0
    kT = _KB_UEV_PER_K * temperature  # μeV
    # Guard against overflow in exp: for E/kT > 500 the Fermi function is ~0
    exponent = np.minimum(E_bins / kT, 500.0)
    fermi = 1.0 / (np.exp(exponent) + 1.0)
    return rho * fermi


def recombination_kernel(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    bath_temperature: float,
) -> np.ndarray:
    """Precompute the NE×NE recombination kernel matrix K^r_{ij}.

    Implements Eq. 17 from the quasiparticle master equation:
      K^r(E_i, E_j) = (1/τ₀) × ((E_i+E_j)/(k_B T_c))² / (k_B T_c)
                       × (1 + Δ²/(E_i E_j)) × N_p(E_i + E_j)

    where N_p(x) = 1/|exp(x / k_B T_p) - 1| is the Bose-Einstein phonon
    occupation at phonon bath temperature T_p.
    """
    kBTc = _KB_UEV_PER_K * T_c          # μeV
    kBTp = _KB_UEV_PER_K * bath_temperature  # μeV

    NE = len(E_bins)
    # Outer sums/products
    E_sum = E_bins[:, None] + E_bins[None, :]         # (NE, NE)
    E_prod = E_bins[:, None] * E_bins[None, :]         # (NE, NE)

    # BCS coherence factor for recombination
    coherence = 1.0 + gap ** 2 / np.maximum(E_prod, 1e-30)

    # Phonon occupation for emission: N_p = 1 + n_BE = 1/(1 - exp(-E/kT))
    # At T=0 this gives N_p = 1 (spontaneous emission).
    if kBTp > 0:
        exponent = np.minimum(E_sum / kBTp, 500.0)
        N_p = 1.0 / (np.exp(exponent) - 1.0) + 1.0
    else:
        # T_p = 0: spontaneous phonon emission, N_p = 1
        N_p = np.ones((NE, NE), dtype=float)

    K_r = (1.0 / tau_0) * (E_sum / kBTc) ** 2 / kBTc * coherence * N_p
    return K_r


def scattering_kernel(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    bath_temperature: float,
) -> np.ndarray:
    """Precompute the NE×NE scattering kernel matrix K^s_{ij}.

    Implements Eq. 16 from the quasiparticle master equation:
      K^s(E_i, E_j) = (1/τ₀) × (E_i - E_j)² / (k_B T_c)³
                       × (1 - Δ²/(E_i E_j)) × N_p(E_i - E_j)

    where N_p(x) = 1/|exp(-x/k_BT) - 1| (footnote 1):
      x > 0 (phonon emission): 1 + n_BE(x)
      x < 0 (phonon absorption): n_BE(|x|)
      x = 0: K^s(E, E) = 0 (no self-scattering)
    """
    kBTc = _KB_UEV_PER_K * T_c
    kBTp = _KB_UEV_PER_K * bath_temperature

    E_diff = E_bins[:, None] - E_bins[None, :]       # (NE, NE)
    E_prod = E_bins[:, None] * E_bins[None, :]

    coherence = np.maximum(1.0 - gap**2 / np.maximum(E_prod, 1e-30), 0.0)

    if kBTp > 0:
        arg = np.minimum(np.abs(E_diff) / kBTp, 500.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_BE = 1.0 / (np.exp(arg) - 1.0)
        N_p = np.where(E_diff > 0, 1.0 + n_BE, n_BE)
        np.fill_diagonal(N_p, 0.0)  # K^s(E, E) = 0
    else:
        # T=0: only spontaneous emission (E_diff > 0), absorption impossible
        N_p = np.where(E_diff > 0, 1.0, 0.0)
        np.fill_diagonal(N_p, 0.0)

    K_s = (1.0 / tau_0) * (E_diff**2) / kBTc**3 * coherence * N_p
    return K_s


def apply_scattering_step(
    state: np.ndarray,
    K_s: np.ndarray,
    rho_bins: np.ndarray,
    dE: float,
    dt: float,
) -> None:
    """Apply one forward-Euler step of quasiparticle-phonon scattering (Eq. 32).

    Modifies *state* in-place.

    state    : shape (num_energy_bins, num_spatial_pts) — spectral density n(E_i, x_k)
    K_s      : shape (NE, NE) — scattering kernel
    rho_bins : shape (NE,) — BCS density of states ρ(E_i)
    dE       : energy bin width (μeV)
    dt       : time step (ns)
    """
    rho = rho_bins[:, None]                     # (NE, 1) broadcast over spatial
    f = state / np.maximum(rho, 1e-30)          # occupation f_i = n_i / ρ_i
    one_minus_f = np.maximum(1.0 - f, 0.0)     # Pauli blocking [1 - f_i]

    # Scattering In:  + ΔE × ρ_i × (1-f_i) × Σ_j n_j K^s_{ji}
    scat_in = dE * rho * one_minus_f * (K_s.T @ state)

    # Scattering Out: - n_i × ΔE × Σ_j K^s_{ij} ρ_j (1-f_j)
    M = K_s * rho_bins[None, :]                 # (NE, NE) — K^s_{ij} ρ_j
    scat_out = state * dE * (M @ one_minus_f)   # (NE, N_spatial)

    state += dt * (scat_in - scat_out)
    np.maximum(state, 0.0, out=state)


def apply_recombination_step(
    state: np.ndarray,
    K_r: np.ndarray,
    G_therm: np.ndarray,
    dE: float,
    dt: float,
) -> None:
    """Apply one forward-Euler step of recombination + thermal generation.

    Modifies *state* in-place.

    state  : shape (num_energy_bins, num_spatial_pts)
    K_r    : shape (num_energy_bins, num_energy_bins)  — recombination kernel
    G_therm: shape (num_energy_bins,) — precomputed thermal generation rate
    dE     : energy bin width (μeV)
    dt     : time step (ns)
    """
    # K_r @ state gives shape (NE, N_spatial): the sum over j of K_r[i,j]*state[j,:]
    Kr_dot_state = K_r @ state                        # (NE, N_spatial)
    recomb_rate = 2.0 * state * dE * Kr_dot_state     # (NE, N_spatial)
    # G_therm[:, None] broadcasts over spatial dimension
    state += dt * (G_therm[:, None] - recomb_rate)
    np.maximum(state, 0.0, out=state)


def _collision_rhs(
    n: np.ndarray,
    K_r: np.ndarray | None,
    K_s: np.ndarray | None,
    rho_bins: np.ndarray | None,
    G_therm: np.ndarray | None,
    dE: float,
) -> np.ndarray:
    """Combined collision RHS for one spatial pixel: dn/dt from recombination + scattering.

    Parameters
    ----------
    n : (NE,) spectral density at one spatial point
    """
    rhs = np.zeros_like(n)

    if K_r is not None and G_therm is not None:
        Kr_dot_n = K_r @ n
        recomb_rate = 2.0 * n * dE * Kr_dot_n
        rhs += G_therm - recomb_rate

    if K_s is not None and rho_bins is not None:
        f = n / np.maximum(rho_bins, 1e-30)
        one_minus_f = np.maximum(1.0 - f, 0.0)
        scat_in = dE * rho_bins * one_minus_f * (K_s.T @ n)
        M = K_s * rho_bins[None, :]
        scat_out = n * dE * (M @ one_minus_f)
        rhs += scat_in - scat_out

    return rhs


def apply_collision_step_bdf(
    state: np.ndarray,
    K_r: np.ndarray | None,
    K_s: np.ndarray | None,
    rho_bins: np.ndarray | None,
    G_therm: np.ndarray | None,
    dE: float,
    dt: float,
) -> int:
    """Apply one BDF collision step using scipy.integrate.solve_ivp per spatial pixel.

    Falls back to forward Euler (with non-negativity clamp) for any pixel where
    BDF fails to converge.  Modifies *state* in-place.

    Returns the number of pixels that fell back to forward Euler.
    """
    from scipy.integrate import solve_ivp

    NE, n_spatial = state.shape
    fallbacks = 0
    for px in range(n_spatial):
        n0 = state[:, px].copy()

        def rhs_fn(t, y):
            return _collision_rhs(y, K_r, K_s, rho_bins, G_therm, dE)

        try:
            sol = solve_ivp(rhs_fn, (0, dt), n0, method="BDF", rtol=1e-6, atol=1e-10)
            if sol.success:
                state[:, px] = np.maximum(sol.y[:, -1], 0.0)
            else:
                fallbacks += 1
                state[:, px] = np.maximum(n0 + dt * _collision_rhs(n0, K_r, K_s, rho_bins, G_therm, dE), 0.0)
        except Exception:
            fallbacks += 1
            state[:, px] = np.maximum(n0 + dt * _collision_rhs(n0, K_r, K_s, rho_bins, G_therm, dE), 0.0)
    return fallbacks


def apply_collision_step_bdf_nonuniform(
    state: np.ndarray,
    K_r_all: np.ndarray | None,
    K_s_all: np.ndarray | None,
    rho_all: np.ndarray | None,
    G_therm_all: np.ndarray | None,
    dE: float,
    dt: float,
    n_spatial: int,
) -> int:
    """BDF collision step with per-pixel kernels (non-uniform gap).

    Modifies *state* in-place.  Returns fallback count.
    """
    from scipy.integrate import solve_ivp

    fallbacks = 0
    for px in range(n_spatial):
        n0 = state[:, px].copy()
        kr = K_r_all[px] if K_r_all is not None else None
        ks = K_s_all[px] if K_s_all is not None else None
        rho = rho_all[px] if rho_all is not None else None
        gt = G_therm_all[px] if G_therm_all is not None else None

        def rhs_fn(t, y, _kr=kr, _ks=ks, _rho=rho, _gt=gt):
            return _collision_rhs(y, _kr, _ks, _rho, _gt, dE)

        try:
            sol = solve_ivp(rhs_fn, (0, dt), n0, method="BDF", rtol=1e-6, atol=1e-10)
            if sol.success:
                state[:, px] = np.maximum(sol.y[:, -1], 0.0)
            else:
                fallbacks += 1
                state[:, px] = np.maximum(n0 + dt * _collision_rhs(n0, kr, ks, rho, gt, dE), 0.0)
        except Exception:
            fallbacks += 1
            state[:, px] = np.maximum(n0 + dt * _collision_rhs(n0, kr, ks, rho, gt, dE), 0.0)
    return fallbacks


def _apply_collision_nonuniform(
    state: np.ndarray,
    K_r_all: np.ndarray | None,
    K_s_all: np.ndarray | None,
    rho_all: np.ndarray | None,
    G_therm_all: np.ndarray | None,
    enable_recombination: bool,
    enable_scattering: bool,
    dE: float,
    dt: float,
    n_spatial: int,
) -> None:
    """Apply collision step with per-pixel kernels using vectorized batch ops."""
    # Work in [N_spatial, NE] layout for batched einsum contractions.
    n_px = state.T  # view, shape (N_spatial, NE)
    if n_px.shape[0] != n_spatial:
        raise ValueError("Internal error: state shape does not match n_spatial.")

    if enable_recombination and K_r_all is not None and G_therm_all is not None:
        # Kr_dot_n[p, i] = sum_j K_r_all[p, i, j] * n_px[p, j]
        Kr_dot_n = np.einsum("pij,pj->pi", K_r_all, n_px, optimize=True)
        recomb_rate = 2.0 * n_px * dE * Kr_dot_n
        n_px = n_px + dt * (G_therm_all - recomb_rate)

    if enable_scattering and K_s_all is not None and rho_all is not None:
        f = n_px / np.maximum(rho_all, 1e-30)
        one_minus_f = np.maximum(1.0 - f, 0.0)
        # scat_in[p, i] = dE * rho[p, i]*(1-f[p, i]) * sum_j K_s[p, j, i] * n[p, j]
        scat_in_core = np.einsum("pji,pj->pi", K_s_all, n_px, optimize=True)
        scat_in = dE * rho_all * one_minus_f * scat_in_core
        # scat_out[p, i] = n[p, i] * dE * sum_j K_s[p, i, j] * rho[p, j] * (1-f[p, j])
        scat_out_rates = np.einsum("pij,pj,pj->pi", K_s_all, rho_all, one_minus_f, optimize=True)
        scat_out = n_px * dE * scat_out_rates
        n_px = n_px + dt * (scat_in - scat_out)

    np.maximum(n_px, 0.0, out=n_px)
    state[:, :] = n_px.T


def evaluate_external_generation(
    spec: ExternalGenerationSpec,
    E_bins: np.ndarray,
    n_spatial: int,
    t: float,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Evaluate external generation rate g_ext(E, x, t).

    Returns (NE, N_spatial) array, or None if mode is "none".
    """
    def _check_generation(arr: np.ndarray, mode_name: str) -> np.ndarray:
        if arr.shape != (NE, n_spatial):
            raise ValueError(
                f"External generation mode '{mode_name}' returned invalid shape "
                f"{arr.shape}; expected {(NE, n_spatial)}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"External generation mode '{mode_name}' produced non-finite values.")
        if np.any(arr < 0):
            raise ValueError(
                f"External generation mode '{mode_name}' produced negative values. "
                "Generation rates must be non-negative."
            )
        return arr

    mode = spec.mode.strip().lower()
    if mode == "none":
        return None

    NE = len(E_bins)

    if mode == "constant":
        return _check_generation(np.full((NE, n_spatial), spec.rate, dtype=float), mode)

    if mode == "pulse":
        if spec.pulse_start <= t < spec.pulse_start + spec.pulse_duration:
            return _check_generation(np.full((NE, n_spatial), spec.pulse_rate, dtype=float), mode)
        return _check_generation(np.zeros((NE, n_spatial), dtype=float), mode)

    if mode == "custom":
        fn = compile_safe_expression(
            spec.custom_body.strip() or "0.0",
            variable_names=("E", "x", "y", "t", "params"),
        )

        ny, nx = mask.shape
        y_idx, x_idx = np.indices(mask.shape)
        x_norm = (x_idx + 0.5) / max(1, nx)
        y_norm = (y_idx + 0.5) / max(1, ny)
        x_flat = x_norm[mask]
        y_flat = y_norm[mask]

        result = np.empty((NE, n_spatial), dtype=float)
        params = dict(spec.custom_params or {})
        try:
            # Try vectorized: pass arrays for E, x, y
            for i in range(NE):
                E_val = E_bins[i]
                val = fn(E=E_val, x=x_flat, y=y_flat, t=t, params=params)
                arr = np.asarray(val, dtype=float)
                if arr.ndim == 0:
                    result[i] = float(arr)
                else:
                    flat = arr.ravel()
                    if flat.size != n_spatial:
                        raise ValueError(
                            "Vectorized custom generation must return a scalar or "
                            f"exactly {n_spatial} values per energy bin; got {flat.size}."
                        )
                    result[i] = flat
        except Exception:
            # Scalar fallback
            for i in range(NE):
                for px in range(n_spatial):
                    result[i, px] = float(
                        fn(
                            E=float(E_bins[i]),
                            x=float(x_flat[px]),
                            y=float(y_flat[px]),
                            t=t,
                            params=params,
                        )
                    )
        return _check_generation(result, mode)

    return None


def _pauli_occupancy_stats(
    state: np.ndarray,
    rho_state: np.ndarray,
    density_floor: float = 1e-18,
) -> tuple[float, tuple[int, int], tuple[int, int] | None]:
    """Return occupancy diagnostics for state n(E,x) against rho(E,x).

    Returns (max_occupation, max_occ_index, forbidden_index_or_none), where
    indices are (energy_idx, spatial_idx).
    """
    if state.shape != rho_state.shape:
        raise ValueError("state and rho_state shapes must match for Pauli diagnostics.")

    rho_mask = rho_state > 1e-30
    forbidden = (~rho_mask) & (state > density_floor)
    forbidden_idx: tuple[int, int] | None = None
    if np.any(forbidden):
        idx = np.unravel_index(int(np.argmax(forbidden)), forbidden.shape)
        forbidden_idx = (int(idx[0]), int(idx[1]))

    f = np.divide(
        state,
        np.maximum(rho_state, 1e-30),
        out=np.zeros_like(state),
        where=rho_mask,
    )
    max_idx_raw = np.unravel_index(int(np.argmax(f)), f.shape)
    max_idx = (int(max_idx_raw[0]), int(max_idx_raw[1]))
    max_occ = float(f[max_idx])
    return max_occ, max_idx, forbidden_idx


def run_2d_crank_nicolson(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    initial_field: np.ndarray,
    diffusion_coefficient: float,
    dt: float,
    total_time: float,
    dx: float,
    store_every: int = 1,
    energy_gap: float = 0.0,
    energy_min_factor: float = 1.0,
    energy_max_factor: float = 10.0,
    num_energy_bins: int = 50,
    energy_weights: np.ndarray | None = None,
    enable_diffusion: bool = True,
    enable_recombination: bool = False,
    enable_scattering: bool = False,
    dynes_gamma: float = 0.0,
    collision_solver: str = "forward_euler",
    tau_0: float = 440.0,
    tau_s: float | None = None,
    tau_r: float | None = None,
    T_c: float = 1.2,
    bath_temperature: float = 0.1,
    external_generation: ExternalGenerationSpec | None = None,
    gap_expression: str = "",
    precomputed: dict | None = None,
    pauli_warn_threshold: float | None = 0.5,
    pauli_error_threshold: float | None = 1.0,
    enforce_pauli: bool = False,
    pauli_density_floor: float = 1e-18,
) -> tuple[list[float], list[np.ndarray], list[float], list[float],
           list[list[np.ndarray]] | None, np.ndarray | None]:
    """Run 2D Crank-Nicolson diffusion, optionally energy-resolved.

    When energy_gap > 0, the state is n(E, x, y, t) discretized into energy bins.
    Each energy bin has D(E) = D₀ × √(1 − (Δ/E)²).

    *energy_weights* optionally overrides the default BCS-DOS weighting used to
    distribute the initial 2D field across energy bins.  Must have length
    ``num_energy_bins``; will be normalized so that ∫ w dE = 1.

    Physical process toggles:
      enable_diffusion    – include CN diffusion step (default True)
      enable_recombination – include recombination/thermal-generation step
      enable_scattering   – include quasiparticle-phonon scattering step

    Returns (times, frames, mass, color_limits, energy_frames_or_None, energy_bins_or_None).
    frames are always energy-integrated 2D arrays for viewer compatibility.
    """
    if dt <= 0 or total_time <= 0:
        raise ValueError("dt and total_time must be positive.")
    if enable_diffusion and diffusion_coefficient <= 0:
        raise ValueError("Diffusion coefficient must be positive.")
    if store_every <= 0:
        store_every = 1
    if initial_field.shape != mask.shape:
        raise ValueError("Initial field shape must match mask shape.")
    n = int(np.sum(mask))
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")
    tau_s_eff = float(tau_s if tau_s is not None else tau_0)
    tau_r_eff = float(tau_r if tau_r is not None else tau_0)
    if enable_scattering and tau_s_eff <= 0:
        raise ValueError("tau_s must be positive when scattering is enabled.")
    if enable_recombination and tau_r_eff <= 0:
        raise ValueError("tau_r must be positive when recombination is enabled.")
    if external_generation is not None:
        external_generation.validate()
    source = np.zeros(n, dtype=float)
    laplacian = None
    identity = None
    if enable_diffusion:
        laplacian, source, _ = build_laplacian_with_boundaries(mask, edges, edge_conditions, dx)
        identity = sparse.eye(n, format="csc")

    full_steps = int(np.floor(total_time / dt + 1e-12))
    remainder_dt = float(total_time - full_steps * dt)
    if remainder_dt < 1e-12:
        remainder_dt = 0.0
    total_steps = full_steps + (1 if remainder_dt > 0.0 else 0)

    # --- Energy-resolved mode ---
    if energy_gap > 0.0:
        gap = energy_gap
        E_bins, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, num_energy_bins)

        # Auto-precompute if gap_expression is set but no precomputed data provided
        if precomputed is None and gap_expression.strip():
            from .precompute import precompute_arrays
            from .models import SimulationParameters
            _auto_params = SimulationParameters(
                diffusion_coefficient=diffusion_coefficient,
                dt=dt, total_time=total_time, mesh_size=dx,
                energy_gap=energy_gap, energy_min_factor=energy_min_factor,
                energy_max_factor=energy_max_factor, num_energy_bins=num_energy_bins,
                dynes_gamma=dynes_gamma, gap_expression=gap_expression,
                tau_0=tau_0, tau_s=tau_s_eff, tau_r=tau_r_eff,
                T_c=T_c, bath_temperature=bath_temperature,
            )
            precomputed = precompute_arrays(mask, edges, edge_conditions, _auto_params)

        # Determine if we have precomputed arrays and whether gap is non-uniform
        has_precomp = precomputed is not None
        nonuniform_gap = has_precomp and not bool(precomputed.get("is_uniform", True))
        use_bdf = normalize_collision_solver_name(collision_solver) == "bdf"
        if enable_scattering and not use_bdf:
            warnings.warn(
                "Forward Euler collision solver can be unstable for stiff scattering dynamics. "
                "Use collision_solver='bdf' for better robustness.",
                stacklevel=2,
            )

        if has_precomp:
            D_array = precomputed["D_array"]  # (NE, N_spatial)
        else:
            # Per-bin diffusion coefficients: D(E) = D₀ × √(1 − (Δ/E)²)
            D_bins_uniform = diffusion_coefficient * np.sqrt(np.maximum(0.0, 1.0 - (gap / E_bins) ** 2))
            D_array = D_bins_uniform[:, None] * np.ones((1, n))

        # Pre-build CN operators per energy bin (only when diffusion is enabled)
        bin_operators: list[tuple | None] = []
        bin_operators_final: list[tuple | None] = []
        var_sources: list[np.ndarray] = []  # per-bin source vectors for variable diffusion
        use_variable_diffusion = False
        if enable_diffusion:
            # Check if D varies spatially (non-uniform gap)
            if nonuniform_gap:
                use_variable_diffusion = True
                for i in range(num_energy_bins):
                    L_D, src_i = build_variable_diffusion_laplacian(
                        mask, edges, edge_conditions, dx, D_array[i],
                    )
                    var_sources.append(src_i)
                    alpha = 0.5 * dt
                    a_mat = (identity - alpha * L_D).tocsc()
                    b_mat_var = (identity + alpha * L_D).tocsr()
                    lu = spla.splu(a_mat)
                    bin_operators.append((b_mat_var, lu))
                    if remainder_dt > 0.0:
                        alpha_f = 0.5 * remainder_dt
                        a_f = (identity - alpha_f * L_D).tocsc()
                        b_f = (identity + alpha_f * L_D).tocsr()
                        lu_f = spla.splu(a_f)
                        bin_operators_final.append((b_f, lu_f))
                    else:
                        bin_operators_final.append(None)
            else:
                for i in range(num_energy_bins):
                    D_i = float(D_array[i, 0]) if D_array.ndim == 2 else float(D_array[i])
                    b_mat, lu = _build_cn_operators(identity, laplacian, dt, D_i)
                    bin_operators.append((b_mat, lu, D_i))
                    if remainder_dt > 0.0:
                        b_f, lu_f = _build_cn_operators(identity, laplacian, remainder_dt, D_i)
                        bin_operators_final.append((b_f, lu_f, D_i))
                    else:
                        bin_operators_final.append(None)

        # Collision kernel setup
        K_r: np.ndarray | None = None
        G_therm: np.ndarray | None = None
        K_s: np.ndarray | None = None
        rho_bins: np.ndarray | None = None
        # Per-pixel collision arrays (non-uniform gap)
        K_r_all: np.ndarray | None = None
        K_s_all: np.ndarray | None = None
        rho_all: np.ndarray | None = None
        G_therm_all: np.ndarray | None = None

        if has_precomp and nonuniform_gap:
            rho_all = precomputed.get("rho_all")
            if enable_recombination:
                K_r_all = precomputed["K_r_all"]
                G_therm_all = precomputed["G_therm_all"]
            if enable_scattering:
                K_s_all = precomputed["K_s_all"]
        elif has_precomp:
            rho_bins = precomputed.get("rho_bins")
            if enable_recombination:
                K_r = precomputed["K_r"]
                G_therm = precomputed["G_therm"]
            if enable_scattering:
                K_s = precomputed["K_s"]
        else:
            if enable_recombination:
                K_r = recombination_kernel(E_bins, gap, tau_r_eff, T_c, bath_temperature)
                n_eq = thermal_qp_weights(E_bins, gap, bath_temperature, dynes_gamma)
                G_therm = 2.0 * n_eq * dE * (K_r @ n_eq)
            if enable_scattering:
                K_s = scattering_kernel(E_bins, gap, tau_s_eff, T_c, bath_temperature)
            # rho is also needed for occupancy diagnostics.
            rho_bins = _dynes_density_of_states(E_bins, gap, dynes_gamma)

        # Occupancy diagnostics also require rho arrays even if scattering is disabled.
        if nonuniform_gap and rho_all is None and has_precomp:
            gap_values = precomputed.get("gap_values")
            if gap_values is not None:
                rho_all = np.empty((n, num_energy_bins), dtype=float)
                for px in range(n):
                    rho_all[px] = _dynes_density_of_states(E_bins, float(gap_values[px]), dynes_gamma)
        if (not nonuniform_gap) and rho_bins is None:
            rho_bins = _dynes_density_of_states(E_bins, gap, dynes_gamma)

        # Distribute initial 2D field across energy bins
        spatial_values = initial_field[mask].astype(float)
        if energy_weights is not None:
            raw_w = np.asarray(energy_weights, dtype=float)
            if raw_w.ndim != 1:
                raise ValueError("energy_weights must be a 1D array.")
            if raw_w.shape[0] != num_energy_bins:
                raise ValueError(
                    f"energy_weights must have length {num_energy_bins}, got {raw_w.shape[0]}."
                )
            if not np.all(np.isfinite(raw_w)):
                raise ValueError("energy_weights must contain only finite values.")
            if np.any(raw_w < 0):
                raise ValueError("energy_weights must be non-negative.")
            w_integral = np.sum(raw_w) * dE
            if w_integral > 0:
                weights = raw_w / w_integral
            else:
                weights = np.ones(num_energy_bins, dtype=float) / (num_energy_bins * dE)
        else:
            # Default: distribute proportional to density of states
            rho = _dynes_density_of_states(E_bins, gap, dynes_gamma)
            rho_integral = np.sum(rho) * dE
            if rho_integral > 0:
                weights = rho / rho_integral
            else:
                weights = np.ones(num_energy_bins, dtype=float) / (num_energy_bins * dE)

        # state[i] is the 1D interior vector for energy bin i
        state = np.empty((num_energy_bins, n), dtype=float)
        for i in range(num_energy_bins):
            state[i] = spatial_values * weights[i]

        coords = np.argwhere(mask)
        rho_state: np.ndarray | None
        if nonuniform_gap and rho_all is not None:
            rho_state = np.asarray(rho_all, dtype=float).T  # [NE, N_spatial]
        elif rho_bins is not None:
            rho_state = np.asarray(rho_bins, dtype=float)[:, None] * np.ones((1, n), dtype=float)
        else:
            rho_state = None

        pauli_warned = False

        def _enforce_pauli(step_idx: int, time_ns: float) -> None:
            nonlocal pauli_warned
            if rho_state is None:
                return
            max_occ, max_idx, forbidden_idx = _pauli_occupancy_stats(
                state,
                rho_state,
                density_floor=pauli_density_floor,
            )
            if forbidden_idx is not None:
                ie_bad, px_bad = forbidden_idx
                row_bad, col_bad = coords[px_bad]
                msg = (
                    f"Detected non-zero quasiparticle density in forbidden state "
                    f"(rho≈0): step={step_idx}, t={time_ns:.6g} ns, "
                    f"E={E_bins[ie_bad]:.6g} μeV, pixel=({int(row_bad)},{int(col_bad)})."
                )
                if enforce_pauli:
                    raise ValueError(msg)
                if not pauli_warned:
                    warnings.warn(msg, stacklevel=2)
                    pauli_warned = True

            if pauli_error_threshold is not None and max_occ > pauli_error_threshold:
                ie_max, px_max = max_idx
                row_max, col_max = coords[px_max]
                msg = (
                    f"Pauli occupation exceeded limit: f={max_occ:.6g} > {pauli_error_threshold:.6g} "
                    f"at step={step_idx}, t={time_ns:.6g} ns, "
                    f"E={E_bins[ie_max]:.6g} μeV, pixel=({int(row_max)},{int(col_max)})."
                )
                if enforce_pauli:
                    raise ValueError(msg)
                if not pauli_warned:
                    warnings.warn(msg, stacklevel=2)
                    pauli_warned = True

            if pauli_warn_threshold is not None and max_occ > pauli_warn_threshold and not pauli_warned:
                ie_max, px_max = max_idx
                row_max, col_max = coords[px_max]
                warnings.warn(
                    "High occupation detected (Pauli blocking regime): "
                    f"max f={max_occ:.6g} at step={step_idx}, t={time_ns:.6g} ns, "
                    f"E={E_bins[ie_max]:.6g} μeV, pixel=({int(row_max)},{int(col_max)}).",
                    stacklevel=2,
                )
                pauli_warned = True

        _enforce_pauli(step_idx=0, time_ns=0.0)

        # Initial energy-integrated field
        integrated = np.sum(state, axis=0) * dE

        times: list[float] = [0.0]
        frames: list[np.ndarray] = [reconstruct_field(mask, integrated)]
        energy_frames: list[list[np.ndarray]] = [
            [reconstruct_field(mask, state[i]) for i in range(num_energy_bins)]
        ]
        mass: list[float] = [float(np.sum(integrated) * dx * dx)]

        current_time = 0.0
        bdf_fallback_total = 0
        collision_enabled = bool(enable_recombination or enable_scattering)

        def _apply_collision(dt_col: float) -> None:
            nonlocal bdf_fallback_total
            if dt_col <= 0.0 or not collision_enabled:
                return

            if nonuniform_gap:
                if use_bdf:
                    bdf_fallback_total += apply_collision_step_bdf_nonuniform(
                        state, K_r_all, K_s_all, rho_all, G_therm_all, dE, dt_col, n,
                    )
                else:
                    _apply_collision_nonuniform(
                        state, K_r_all, K_s_all, rho_all, G_therm_all,
                        enable_recombination, enable_scattering, dE, dt_col, n,
                    )
                return

            if use_bdf:
                bdf_fallback_total += apply_collision_step_bdf(
                    state, K_r, K_s, rho_bins, G_therm, dE, dt_col,
                )
                return

            if enable_recombination and K_r is not None and G_therm is not None:
                apply_recombination_step(state, K_r, G_therm, dE, dt_col)
            if enable_scattering and K_s is not None and rho_bins is not None:
                apply_scattering_step(state, K_s, rho_bins, dE, dt_col)

        def _apply_diffusion(dt_diff: float, is_final_step: bool) -> None:
            if dt_diff <= 0.0 or not enable_diffusion:
                return
            if use_variable_diffusion:
                for i in range(num_energy_bins):
                    if is_final_step:
                        ops = bin_operators_final[i]
                        if ops is None:
                            raise RuntimeError("Internal error: final-step solver not initialized.")
                        b_step, lu_step = ops
                    else:
                        b_step, lu_step = bin_operators[i]
                    rhs = b_step @ state[i] + dt_diff * var_sources[i]
                    state[i] = lu_step.solve(rhs)
            else:
                for i in range(num_energy_bins):
                    if is_final_step:
                        ops = bin_operators_final[i]
                        if ops is None:
                            raise RuntimeError("Internal error: final-step solver not initialized.")
                        b_step, lu_step, D_i = ops
                    else:
                        b_step, lu_step, D_i = bin_operators[i]
                    rhs = b_step @ state[i] + dt_diff * D_i * source
                    state[i] = lu_step.solve(rhs)

        for step in range(1, total_steps + 1):
            is_final = step > full_steps
            dt_step = remainder_dt if is_final else dt

            # Step 0: External generation — before collision per PDF Appendix A.1
            if external_generation is not None and external_generation.mode != "none":
                g_ext = evaluate_external_generation(
                    external_generation, E_bins, n, current_time, mask,
                )
                if g_ext is not None:
                    state += dt_step * g_ext

            # Symmetric split when both collision and diffusion are enabled:
            # C(dt/2) -> D(dt) -> C(dt/2). Pure-collision and pure-diffusion
            # paths keep a single full step.
            if collision_enabled and enable_diffusion:
                _apply_collision(0.5 * dt_step)
                _apply_diffusion(dt_step, is_final)
                _apply_collision(0.5 * dt_step)
            else:
                _apply_collision(dt_step)
                _apply_diffusion(dt_step, is_final)

            _enforce_pauli(step_idx=step, time_ns=current_time + dt_step)
            current_time += dt_step
            if step % store_every == 0 or step == total_steps:
                integrated = np.sum(state, axis=0) * dE
                times.append(float(current_time))
                frames.append(reconstruct_field(mask, integrated))
                energy_frames.append(
                    [reconstruct_field(mask, state[i]) for i in range(num_energy_bins)]
                )
                mass.append(float(np.sum(integrated) * dx * dx))

        if bdf_fallback_total > 0:
            warnings.warn(
                f"BDF collision solver fell back to forward Euler for "
                f"{bdf_fallback_total} pixel-steps. Consider reducing dt.",
                stacklevel=2,
            )

        min_val = float(np.nanmin(np.stack(frames)))
        max_val = float(np.nanmax(np.stack(frames)))
        if abs(max_val - min_val) < 1e-12:
            max_val = min_val + 1e-9
        return times, frames, mass, [min_val, max_val], energy_frames, E_bins

    # --- Legacy scalar mode (energy_gap == 0) ---
    interior_values = initial_field[mask].astype(float)

    if enable_diffusion:
        if identity is None or laplacian is None:
            raise RuntimeError("Internal error: diffusion matrices were not initialized.")
        b_mat, lu = _build_cn_operators(identity, laplacian, dt, diffusion_coefficient)
        final_lu = None
        final_b_mat = None
        if remainder_dt > 0.0:
            final_b_mat, final_lu = _build_cn_operators(identity, laplacian, remainder_dt, diffusion_coefficient)

    times = [0.0]
    frames = [reconstruct_field(mask, interior_values)]
    mass = [float(np.sum(interior_values) * dx * dx)]

    current = interior_values
    current_time = 0.0
    for step in range(1, total_steps + 1):
        if step <= full_steps:
            dt_step = dt
        else:
            dt_step = remainder_dt
        if enable_diffusion:
            if step <= full_steps:
                b_step = b_mat
                lu_step = lu
            else:
                if final_b_mat is None or final_lu is None:
                    raise RuntimeError("Internal error: final-step solver matrices were not initialized.")
                b_step = final_b_mat
                lu_step = final_lu
            rhs = b_step @ current + dt_step * diffusion_coefficient * source
            current = lu_step.solve(rhs)
        current_time += dt_step
        if step % store_every == 0 or step == total_steps:
            times.append(float(current_time))
            frame = reconstruct_field(mask, current)
            frames.append(frame)
            mass.append(float(np.sum(current) * dx * dx))

    min_val = float(np.nanmin(np.stack(frames)))
    max_val = float(np.nanmax(np.stack(frames)))
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1e-9
    return times, frames, mass, [min_val, max_val], None, None
