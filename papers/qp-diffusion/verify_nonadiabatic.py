"""
verify_nonadiabatic.py -- beyond-assumption frontier: NONADIABATIC spectral dynamics.

The adiabatic time-dependent spectral flow is exact to FIRST Wigner/Moyal
order: the instantaneous BCS gR0 solves [L0,gR0]_star = 0 there
(verify_traces I1).  At second order its residual must be combined with the
normalized spectral correction, rather than interpreted by itself.

2026-06-09 errata: gA0 flipped to the physical continuation
gA0 = -tau3 (gR0)^dag tau3 = -N1 t3 - i N2 t2 = -gR0 (above the gap), per
verify_gA_convention.py; all gK-based results below recomputed.  The
previously-claimed O(h) branch-coherence drive (old eq:nonad_coherence,
4 E^2 (E Dd dE fL + Delta dt fL)/W^3) was an ARTIFACT of the superseded gA:
under the corrected convention the t1 channel vanishes at O(h^1) and the
first nonvanishing coherence drive is O(h^2) -- the same order as the
retarded residual R2.

Results (symbolic, generic Delta(t),fL,fT; numeric guards; conventions = verify_traces):
  Moyal A*B = AB + (i h/2)(dE A dt B - dt A dE B)
            + (1/2)(i h/2)^2 (dEE A dtt B - 2 dEt A dEt B + dtt A dEE B) + ...
  L0 = E t3 + i Delta(t) t2 ;  gR0 = N1 t3 + i N2 t2 ;  gA0 = -gR0
  N1=E/W, N2=Delta/W, W=sqrt(E^2-Delta^2) ;  h = fL 1 + fT t3
  (Dd = dDelta/dt, Ddd = d^2Delta/dt^2)

  R1 : [L0,gR0]_star vanishes at O(h^0),O(h^1), while the instantaneous
       ansatz has O(h^2) residual 3 E D^2 Ddd/(4 W^5) * t1.
       The normalized correction
         d = alpha gR0 + beta m0,
         alpha = (W^2 Dd^2 - 2 E^2 D Ddd)/(8 W^6),
         beta = -3 E D^2 Ddd/(8 W^6),
         m0 = (D/W)t3 + i(E/W)t2,
       gives gR=gR0+h^2 d and gA=-gR0-h^2 d.  Both spectral equations and
       both star-normalizations then hold through O(h^2).
  R2 : i[L0,gK]_star channels (corrected convention):
        plain (fL): O(h^1) = -4 [dt(N1 fL) + dE(Dd N2 fL)]   [= corrected verify_traces
                    I2, the canonical spectral-flow form; NO fT; O(h^0)=O(h^2)=0]
        t3   (fT) : O(h^1) = -4 (E/W^3)(E^2 dt fT + E Delta Dd dE fT + Delta Dd fT)
                    [NO fL -> still no direct energy->charge mixing.  STATUS
                    2026-06-10: the transverse kinetic interpretation, left open
                    here when this was first recorded, is CLOSED by verify_fT.py
                    (its R1): this raw t3-trace is the N1-weighted kernel channel,
                    2 N1 B, of the transverse kinetic equation -- a projection
                    weight, not extra physics]
        t1  (coh) : O(h^1) = 0.  With
                    gK=gR*h-h*gA, including h^2(d*h+h*d), the O(h^2)
                    trace is a generic coherence-channel source but cancels
                    exactly for a shell-slaved distribution fL=G(W), and also
                    for fL=1.  A static gap with independently time-dependent
                    fL retains -2i E Delta dtt(fL)/W^3.
        t2 channel: O(h^1) = i (Delta/E) x (t3 channel result).
       => the completed O(h^2) result still has NO direct L-T coupling in any
          projection checked; the nonzero spectral correction is distinct from
          the shell-slaved coherence-source cancellation.
"""
import sympy as sp

E, t, hbar = sp.symbols('E t hbar', real=True, positive=True)
I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])


def dE(M):
    return M.applyfunc(lambda e: sp.diff(e, E))


def dt(M):
    return M.applyfunc(lambda e: sp.diff(e, t))


def moyal_coeff(A, B, n):
    """Coefficient of hbar^n in A star B for hbar-independent A and B."""
    if n == 0:
        return A * B
    if n == 1:
        return (I_ / 2) * (dE(A) * dt(B) - dt(A) * dE(B))
    if n == 2:
        return -sp.Rational(1, 8) * (
            dE(dE(A)) * dt(dt(B))
            - 2 * dE(dt(A)) * dE(dt(B))
            + dt(dt(A)) * dE(dE(B)))
    return sp.zeros(A.rows, B.cols)


def star2(A, B):
    return sum((hbar**n * moyal_coeff(A, B, n) for n in range(3)), sp.zeros(A.rows, B.cols))


def series_star_coeff(A_terms, B_terms, n):
    """Coefficient-wise star product, truncated without higher-order blow-up."""
    out = sp.zeros(2, 2)
    for p, A in A_terms.items():
        for q, B in B_terms.items():
            r = n - p - q
            if 0 <= r <= 2:
                out += moyal_coeff(A, B, r)
    return out


def series_scomm_coeff(A_terms, B_terms, n):
    return series_star_coeff(A_terms, B_terms, n) - series_star_coeff(B_terms, A_terms, n)


def keldysh_series(g0, d, h):
    gR_terms = {0: g0}
    gA_terms = {0: -g0}
    if d is not None:
        gR_terms[2] = d
        gA_terms[2] = -d
    h_terms = {0: h}
    return {
        n: series_star_coeff(gR_terms, h_terms, n)
        - series_star_coeff(h_terms, gA_terms, n)
        for n in range(3)
    }


def kinetic_series(L0, gK_terms):
    L_terms = {0: L0}
    return {n: I_ * series_scomm_coeff(L_terms, gK_terms, n) for n in range(3)}


def scomm2(A, B):
    return star2(A, B) - star2(B, A)


def hbar_coeff(expr, n):
    return sp.expand(expr).coeff(hbar, n)


def spectral_terms(Delta):
    """Return the instantaneous state and its normalized O(hbar^2) correction."""
    W = sp.sqrt(E**2 - Delta**2)
    Dd = sp.diff(Delta, t)
    Ddd = sp.diff(Delta, t, 2)
    L0 = E * t3 + I_ * Delta * t2
    g0 = (E / W) * t3 + I_ * (Delta / W) * t2
    m0 = (Delta / W) * t3 + I_ * (E / W) * t2
    alpha = (W**2 * Dd**2 - 2 * E**2 * Delta * Ddd) / (8 * W**6)
    beta = -3 * E * Delta**2 * Ddd / (8 * W**6)
    d = alpha * g0 + beta * m0
    return W, L0, g0, m0, alpha, beta, d


def build(Delta, fL, fT):
    W, L0, gR0, _m0, alpha, beta, d = spectral_terms(Delta)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    gA0 = -gR0                              # = -t3 (gR0)^dag t3 (above the gap)
    h = fL * I2 + fT * t3
    L_terms = {0: L0}
    gR0_terms = {0: gR0}
    gA0_terms = {0: gA0}
    gR_terms = {0: gR0, 2: d}
    gA_terms = {0: gA0, 2: -d}             # physical continuation: dA = -dR
    gK0_terms = keldysh_series(gR0, None, h)
    gK_terms = keldysh_series(gR0, d, h)
    K0_terms = kinetic_series(L0, gK0_terms)
    K_terms = kinetic_series(L0, gK_terms)

    def chan(P, n):
        return sp.trace(P * K_terms[n])

    def chan0(P, n):
        return sp.trace(P * K0_terms[n])

    out = {}
    # The instantaneous ansatz residual motivates d but is not the completed equation.
    out['instantaneous R_h0 = 0'] = (
        series_scomm_coeff(L_terms, gR0_terms, 0), sp.zeros(2, 2))
    out['instantaneous R_h1 = 0'] = (
        series_scomm_coeff(L_terms, gR0_terms, 1), sp.zeros(2, 2))
    R2mat = series_scomm_coeff(L_terms, gR0_terms, 2)
    out['instantaneous R2 = (3 E D^2 Ddot/4W^5) t1'] = (
        sp.Matrix([[sp.simplify((R2mat - (3 * E * Delta**2 * sp.diff(Delta, t, 2) / (4 * W**5)) * t1).norm())]]),
        sp.zeros(1, 1))
    out['corrected retarded spectral equation O(h2) = 0'] = (
        series_scomm_coeff(L_terms, gR_terms, 2), sp.zeros(2, 2))
    out['corrected advanced spectral equation O(h2) = 0 (dA=-dR)'] = (
        series_scomm_coeff(L_terms, gA_terms, 2), sp.zeros(2, 2))
    out['corrected retarded star-normalization O(h2) = 0'] = (
        series_star_coeff(gR_terms, gR_terms, 2), sp.zeros(2, 2))
    out['corrected advanced star-normalization O(h2) = 0'] = (
        series_star_coeff(gA_terms, gA_terms, 2), sp.zeros(2, 2))
    out['corrected Keldysh star-normalization O(h0) = 0'] = (
        series_star_coeff(gR_terms, gK_terms, 0) + series_star_coeff(gK_terms, gA_terms, 0),
        sp.zeros(2, 2))
    out['corrected Keldysh star-normalization O(h1) = 0'] = (
        series_star_coeff(gR_terms, gK_terms, 1) + series_star_coeff(gK_terms, gA_terms, 1),
        sp.zeros(2, 2))
    out['corrected Keldysh star-normalization O(h2) = 0'] = (
        series_star_coeff(gR_terms, gK_terms, 2) + series_star_coeff(gK_terms, gA_terms, 2),
        sp.zeros(2, 2))
    d3 = (E * alpha + Delta * beta) / W
    out['gK correction O(h2) = d h + h d'] = (
        gK_terms[2] - gK0_terms[2] - (d * h + h * d), sp.zeros(2, 2))
    out['d h + h d = 2 fL d + 2 fT d3 I'] = (
        d * h + h * d - (2 * fL * d + 2 * fT * d3 * I2), sp.zeros(2, 2))
    # kinetic channels at O(h^1)
    out['fL plain O(h1) = -4[dt(N1 fL) + dE(Dd N2 fL)]'] = (
        sp.Matrix([[sp.simplify(chan(I2, 1)
                                - (-4 * (sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E))))]]), sp.zeros(1, 1))
    out['fT t3 O(h1) = -4(E/W^3)(E^2 dtfT + E D Dd dEfT + D Dd fT)'] = (
        sp.Matrix([[sp.simplify(chan(t3, 1)
                                - (-4 * E / W**3 * (E**2 * sp.diff(fT, t)
                                                    + E * Delta * Dd * sp.diff(fT, E)
                                                    + Delta * Dd * fT)))]]), sp.zeros(1, 1))
    out['coh t1 O(h1) = 0 (old O(h) drive was a gA artifact)'] = (
        sp.Matrix([[sp.simplify(chan(t1, 1))]]), sp.zeros(1, 1))
    out['t2 O(h1) = i(D/E) x t3 channel'] = (
        sp.Matrix([[sp.simplify(chan(t2, 1) - I_ * (Delta / E) * chan(t3, 1))]]), sp.zeros(1, 1))
    from sympy.core.function import AppliedUndef
    # 2026-07-07 (review round): the structural no-coupling checks were
    # .diff()-based, which is blind to dependence entering through
    # Derivative(f, t) / Derivative(f, E) terms.  Replaced by .has(),
    # which sees the function inside any derivative.  (Both referees'
    # re-derivations confirm the strengthened statements still hold.)
    if isinstance(fL, AppliedUndef):     # structural checks only make sense symbolically
        # .has(f.func) rather than .has(f): robust to Subs/chain-rule forms
        # where the function appears with a different argument structure.
        out['t3 channel has NO fL at O(h1),O(h2) [incl. derivatives]'] = (
            sp.Matrix([[sp.Integer(1) if (sp.simplify(chan(t3, 1)).has(fL.func)
                                          or sp.simplify(chan(t3, 2)).has(fL.func))
                        else sp.Integer(0)]]), sp.zeros(1, 1))
        out['coh t1 O(h2) sourced by fL only (no fT, incl. derivatives)'] = (
            sp.Matrix([[sp.Integer(1) if sp.simplify(chan(t1, 2)).has(fT.func)
                        else sp.Integer(0)]]), sp.zeros(1, 1))
    if isinstance(fT, AppliedUndef):
        out['plain channel has NO fT at O(h1),O(h2) [incl. derivatives]'] = (
            sp.Matrix([[sp.Integer(1) if (sp.simplify(chan(I2, 1)).has(fT.func)
                                          or sp.simplify(chan(I2, 2)).has(fT.func))
                        else sp.Integer(0)]]), sp.zeros(1, 1))
    if isinstance(fL, AppliedUndef) and isinstance(fT, AppliedUndef):
        # The normalized d correction changes only the t1 projection at O(h^2).
        completed_t1 = -2 * I_ / W**3 * (
            (E**2 + Delta**2) * Dd * sp.diff(fL, E, t)
            + E * Delta * (sp.diff(fL, t, 2) + Dd**2 * sp.diff(fL, E, 2))
            + Delta**2 * sp.diff(Delta, t, 2) * sp.diff(fL, E))
        out['completed t1 O(h2) trace (generic fL, fT)'] = (
            sp.Matrix([[sp.simplify(chan(t1, 2) - completed_t1)]]), sp.zeros(1, 1))
        out['plain projection unchanged by d at O(h2)'] = (
            sp.Matrix([[sp.simplify(chan(I2, 2) - chan0(I2, 2))]]), sp.zeros(1, 1))
        out['t3 projection unchanged by d at O(h2)'] = (
            sp.Matrix([[sp.simplify(chan(t3, 2) - chan0(t3, 2))]]), sp.zeros(1, 1))
        out['t2 projection unchanged by d at O(h2)'] = (
            sp.Matrix([[sp.simplify(chan(t2, 2) - chan0(t2, 2))]]), sp.zeros(1, 1))
        out['plain channel O(h2) = 0 (generic fL, fT)'] = (
            sp.Matrix([[sp.simplify(chan(I2, 2))]]), sp.zeros(1, 1))
        out['t3 channel O(h2) = 0 (generic fL, fT)'] = (
            sp.Matrix([[sp.simplify(chan(t3, 2))]]), sp.zeros(1, 1))
        out['t2 channel O(h2) = 0 (generic fL, fT)'] = (
            sp.Matrix([[sp.simplify(chan(t2, 2))]]), sp.zeros(1, 1))
        transverse_kernel_h2 = N1 * chan(t3, 2) + I_ * N2 * chan(t2, 2)
        out['transverse kernel combination O(h2) = 0'] = (
            sp.Matrix([[sp.simplify(transverse_kernel_h2)]]), sp.zeros(1, 1))
    return out


def check(out, label, subs=None):
    print(f"---- {label} ----")
    ok = True
    for name, (expr, exp) in out.items():
        d = sp.Matrix(expr) - sp.Matrix(exp)
        if subs:
            d = d.subs(subs)
            val = max(abs(complex(sp.N(e))) for e in d)
            good = val < 1e-9
            print(f"  [{'PASS' if good else 'FAIL'}] max|.|={val:.1e}  {name}")
        else:
            good = all(sp.simplify(e) == 0 for e in d)
            print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good
    return ok


def mclaims_2026_07_10():
    """Checks pinned to the completed manuscript claims (physics audit 2026-07-10):

    M1: including hbar^2(d*h+h*d) removes the spurious undifferentiated-fL
        term from the generic O(hbar^2) t1 trace.  The result vanishes exactly
        for fL=G(W) and fL=1, but survives for a static gap with independently
        time-dependent fL.
    M2: d=alpha*g0+beta*m0 is the normalized spectral correction.  It has no
        identity or t1 component; its t3 component is nonzero and contains
        both Ddd and Dd^2 pieces.
    """
    print("---- (d) 2026-07-10 completed-claims block ----")
    ok = True
    Delta = sp.Function('Delta', positive=True)(t)
    fL = sp.Function('f_L')(E, t)
    W, L0, gR0, _m0, alpha, beta, d = spectral_terms(Delta)
    Dd = sp.diff(Delta, t)
    Ddd = sp.diff(Delta, t, 2)
    h = fL * I2                                  # energy mode only
    K0_terms = kinetic_series(L0, keldysh_series(gR0, None, h))
    K_terms = kinetic_series(L0, keldysh_series(gR0, d, h))
    t1_instantaneous = sp.simplify(sp.trace(t1 * K0_terms[2]))
    t1_h2 = sp.simplify(sp.trace(t1 * K_terms[2]))

    # M1a: pin the full generic t1 O(h^2) coefficient
    pin = I_ * (-2 * Dd * (E**4 - Delta**4) * sp.diff(fL, E, t)
                - 2 * E * Delta * W**2 * (sp.diff(fL, t, 2) + Dd**2 * sp.diff(fL, E, 2))
                - 2 * Delta**2 * W**2 * Ddd * sp.diff(fL, E)) / W**5
    good = sp.simplify(t1_h2 - pin) == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1a: completed generic t1 O(h2) source matches the pinned expression")

    # The omitted Keldysh correction cancels the instantaneous ansatz's
    # undifferentiated-fL contribution exactly.
    missing = sp.simplify(t1_h2 - t1_instantaneous)
    missing_pin = -3 * I_ * E * Delta**2 * Ddd * fL / W**5
    good = sp.simplify(missing - missing_pin) == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1a-correction: hbar^2(dh+hd) contributes -3i E D^2 Ddd fL/W^5")

    # M1b: static gap, time-dependent fL -> source SURVIVES (= -2i E D d2fL/dt2 / W^3)
    Dc = sp.Symbol('Delta_c', positive=True)
    static = t1_h2.subs({Ddd: 0, Dd: 0}).subs(Delta, Dc)
    Wc_ = sp.sqrt(E**2 - Dc**2)
    good = sp.simplify(static - (-2 * I_ * E * Dc * sp.diff(fL, t, 2) / Wc_**3)) == 0 \
        and sp.simplify(static) != 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1b: static-gap t1 O(h2) source = -2i E D d2fL/dt2 / W^3 (nonzero)")

    # M1c: a distribution slaved to the instantaneous shell cancels exactly.
    G = sp.Function('G')
    fLs = G(W)
    hs = fLs * I2
    Kslaved = kinetic_series(L0, keldysh_series(gR0, d, hs))
    t1s = sp.trace(t1 * Kslaved[2])
    good = sp.simplify(t1s) == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1c: shell-slaved fL=G(W) gives zero t1 O(h2) source exactly")

    # M1d: a constant longitudinal distribution is another exact zero.
    h1 = I2
    Kconst = kinetic_series(L0, keldysh_series(gR0, d, h1))
    t1_const = sp.trace(t1 * Kconst[2])
    good = sp.simplify(t1_const) == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1d: constant fL=1 gives zero t1 O(h2) source exactly")

    # M2: pin the Pauli decomposition of the explicit normalized correction.
    d0 = sp.simplify(sp.trace(d) / 2)
    d1 = sp.simplify(sp.trace(t1 * d) / 2)
    d2 = sp.simplify(sp.trace(t2 * d) / 2)
    d3 = sp.simplify(sp.trace(t3 * d) / 2)
    d2_pin = I_ * (Delta * alpha + E * beta) / W
    d3_pin = (E * alpha + Delta * beta) / W
    d3n = sp.expand(sp.numer(sp.together(d3)))
    good = (d0 == 0) and (d1 == 0) \
        and sp.simplify(d2 - d2_pin) == 0 \
        and sp.simplify(d3 - d3_pin) == 0 \
        and d3 != 0 and d3n.has(Ddd) and d3n.has(Dd**2)
    # Explicit d is independently validated by the spectral and normalization
    # residuals in build().
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M2: explicit d has no I/t1 part and its t3 part carries Ddd and Dd^2")
    return ok


if __name__ == "__main__":
    a = check(build(sp.Function('Delta', positive=True)(t),
                    sp.Function('f_L')(E, t), sp.Function('f_T')(E, t)),
              "(a) symbolic, generic Delta(t), fL(E,t), fT(E,t)")
    print()
    b = check(build(1 + sp.Rational(3, 10) * t + sp.Rational(1, 10) * t**2,
                    sp.exp(-E) * sp.cos(t), sp.sin(E) * sp.exp(-t)),
              "(b) numeric, concrete functions", {E: 2, t: sp.Rational(1, 2), hbar: 1})
    print()
    # adiabatic limit: constant gap (Ddot=0 AND Dd=0) -> R2=0 and no coherence source from gap
    print("---- (c) adiabatic limit: constant gap Delta=1 ----")
    Wc = sp.sqrt(E**2 - 1)
    L0c = E * t3 + I_ * 1 * t2
    gR0c = (E / Wc) * t3 + I_ * (1 / Wc) * t2
    R2c = sp.simplify(sp.Matrix(scomm2(L0c, gR0c)).applyfunc(lambda e: hbar_coeff(e, 2)).norm())
    print(f"  [{'PASS' if R2c == 0 else 'FAIL'}] R2 = 0 for constant gap (no Ddot): |R2|={R2c}")
    print()
    d = mclaims_2026_07_10()
    print()
    all_ok = a and b and R2c == 0 and d
    print("ALL PASS" if all_ok else "SOME FAILED -- inspect above")
    raise SystemExit(0 if all_ok else 1)
