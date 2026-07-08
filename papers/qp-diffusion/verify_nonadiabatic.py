"""
verify_nonadiabatic.py -- beyond-assumption frontier: NONADIABATIC spectral dynamics.

The adiabatic time-dependent spectral flow is exact to FIRST Wigner/Moyal
order: the instantaneous BCS gR0 solves [L0,gR0]_star = 0 there
(verify_traces I1).  Nonadiabaticity is the first NONVANISHING residual.

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
  L0 = E t3 + i Delta(t) t2 ;  gR0 = N1 t3 + i N2 t2 ;  gA0 = -N1 t3 - i N2 t2
  N1=E/W, N2=Delta/W, W=sqrt(E^2-Delta^2) ;  h = fL 1 + fT t3 ;  gK = gR0*h - h*gA0
  (Dd = dDelta/dt, Ddd = d^2Delta/dt^2)

  R1 : [L0,gR0]_star  vanishes at O(h^0),O(h^1); O(h^2) residual R2 = 3 E D^2 Ddd/(4 W^5) * t1
       (purely t1 = off-diagonal branch sector; proportional to Ddd)
       [gR-only: unaffected by the errata]
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
        t1  (coh) : O(h^1) = 0; first nonvanishing drive is O(h^2), sourced by fL
                    only (depends on Dd^2 and Ddd).  t2 channel: O(h^1) =
                    i (Delta/E) x (t3 channel result).
       => nonadiabatic branch-coherence generation is pushed to O(h^2), alongside R2;
          there is still NO direct L-T coupling at any order checked.
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


def star2(A, B):
    s0 = A * B
    s1 = (I_ * hbar / 2) * (dE(A) * dt(B) - dt(A) * dE(B))
    s2 = sp.Rational(1, 2) * (I_ * hbar / 2)**2 * (
        dE(dE(A)) * dt(dt(B)) - 2 * dE(dt(A)) * dE(dt(B)) + dt(dt(A)) * dE(dE(B)))
    return s0 + s1 + s2


def scomm2(A, B):
    return star2(A, B) - star2(B, A)


def hbar_coeff(expr, n):
    return sp.expand(expr).coeff(hbar, n)


def build(Delta, fL, fT):
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    L0 = E * t3 + I_ * Delta * t2
    gR0 = N1 * t3 + I_ * N2 * t2
    gA0 = -N1 * t3 - I_ * N2 * t2          # = -t3 (gR0)^dag t3 = -gR0 (corrected 2026-06-09)
    h = fL * I2 + fT * t3
    gK = star2(gR0, h) - star2(h, gA0)

    C = scomm2(L0, gR0)                    # retarded residual
    K = I_ * scomm2(L0, gK)                # kinetic operator

    def chan(P, n):
        return sp.trace(P * K).expand().coeff(hbar, n)

    out = {}
    # retarded residual orders (matrix must vanish at h^0,h^1)
    out['R_h0 = 0'] = (sp.Matrix([[sp.simplify(sp.Matrix(C).applyfunc(lambda e: hbar_coeff(e, 0)).norm())]]), sp.zeros(1, 1))
    out['R_h1 = 0'] = (sp.Matrix([[sp.simplify(sp.Matrix(C).applyfunc(lambda e: hbar_coeff(e, 1)).norm())]]), sp.zeros(1, 1))
    # O(h^2) residual is purely t1 = 3 E D^2 Ddot/(4 W^5)
    R2mat = sp.Matrix(C).applyfunc(lambda e: hbar_coeff(e, 2))
    out['R2 = (3 E D^2 Ddot/4W^5) t1'] = (
        sp.Matrix([[sp.simplify((R2mat - (3 * E * Delta**2 * sp.diff(Delta, t, 2) / (4 * W**5)) * t1).norm())]]),
        sp.zeros(1, 1))
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
        # pins the tex claim that the energy projection receives NO O(h^2)
        # correction (generic fL AND fT retained)
        out['plain channel O(h2) = 0 (generic fL, fT)'] = (
            sp.Matrix([[sp.simplify(chan(I2, 2))]]), sp.zeros(1, 1))
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


def mclaims_2026_07_07():
    """Checks pinned to the corrected manuscript claims (review round 2026-07-07):

    M1 (SM app:nonadiabatic, Summary + coherence paragraph): the generic
        O(h^2) t1 source carries Dd*d2fL/dEdt and Delta*d2fL/dt2 pieces and
        SURVIVES a static gap with time-dependent fL; it is proportional to
        (Ddd, Dd^2) only for gap-slaved distributions fL = G(xi(E,t)).
    M2 (SM app:nonadiabatic, Spectrum paragraph): solving the O(h^2)
        spectral equation with star-normalization, the induced correction
        dg to gR0 has NO t1 component (it lies in the t2/t3 sector) and its
        t3 coefficient is nonzero, carrying BOTH Ddd and Dd^2 pieces.
    """
    print("---- (d) 2026-07-07 corrected-claims block ----")
    ok = True
    Delta = sp.Function('Delta', positive=True)(t)
    fL = sp.Function('f_L')(E, t)
    W = sp.sqrt(E**2 - Delta**2)
    Dd = sp.diff(Delta, t)
    Ddd = sp.diff(Delta, t, 2)
    L0 = E * t3 + I_ * Delta * t2
    gR0 = (E / W) * t3 + I_ * (Delta / W) * t2
    gA0 = -gR0
    h = fL * I2                                  # energy mode only
    gK = star2(gR0, h) - star2(h, gA0)
    K = I_ * scomm2(L0, gK)
    t1_h2 = sp.simplify(sp.expand(sp.trace(t1 * K)).coeff(hbar, 2))

    # M1a: pin the full generic t1 O(h^2) coefficient
    pin = I_ * (-2 * Dd * (E**4 - Delta**4) * sp.diff(fL, E, t)
                - 2 * E * Delta * W**2 * (sp.diff(fL, t, 2) + Dd**2 * sp.diff(fL, E, 2))
                - 2 * Delta**2 * W**2 * Ddd * sp.diff(fL, E)
                + 3 * E * Delta**2 * Ddd * fL) / W**5
    good = sp.simplify(t1_h2 - pin) == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1a: generic t1 O(h2) source matches the pinned expression")

    # M1b: static gap, time-dependent fL -> source SURVIVES (= -2i E D d2fL/dt2 / W^3)
    Dc = sp.Symbol('Delta_c', positive=True)
    static = t1_h2.subs({Ddd: 0, Dd: 0}).subs(Delta, Dc)
    Wc_ = sp.sqrt(E**2 - Dc**2)
    good = sp.simplify(static - (-2 * I_ * E * Dc * sp.diff(fL, t, 2) / Wc_**3)) == 0 \
        and sp.simplify(static) != 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1b: static-gap t1 O(h2) source = -2i E D d2fL/dt2 / W^3 (nonzero)")

    # M1c: gap-slaved distribution fL = G(xi), xi = sqrt(E^2-Delta(t)^2)
    #      -> every term carries Ddd or Dd^2 (vanishes as Dd,Ddd -> 0)
    G = sp.Function('G')
    fLs = G(sp.sqrt(E**2 - Delta**2))
    hs = fLs * I2
    gKs = star2(gR0, hs) - star2(hs, -gR0)
    t1s = sp.expand(sp.trace(t1 * (I_ * scomm2(L0, gKs)))).coeff(hbar, 2)
    slaved_limit = sp.simplify(t1s.subs({Ddd: 0}).subs({Dd: 0}))
    good = slaved_limit == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1c: gap-slaved t1 O(h2) source vanishes as Dd,Ddd -> 0")

    # M1d: strictly 'proportional to Ddd and Dd^2' — no LINEAR-Dd term either
    # (a term ~ Dd^1 with no Ddd would vanish in the M1c limit yet falsify
    # the tex wording).  Map the gap derivatives onto plain symbols and
    # check the Ddd-free part has no linear piece in Dd.
    u, v = sp.symbols('u v')
    t1p = sp.expand(t1s).subs({Ddd: v}).subs({Dd: u})
    lin = sp.simplify(sp.diff(t1p.subs(v, 0), u).subs(u, 0))
    good = lin == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M1d: gap-slaved source has no linear-Dd term (strictly Ddd and Dd^2)")

    # M2: solve for the O(h^2) correction dg = a0 I + a1 t1 + a2 t2 + a3 t3
    #     conditions: (i)  [L0, dg] + (O(h^2) of [L0,gR0]_star) = 0
    #                 (ii) gR0 dg + dg gR0 + (O(h^2) of gR0*gR0 - 1) = 0
    a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3')
    dg = a0 * I2 + a1 * t1 + a2 * t2 + a3 * t3
    spec_res = sp.Matrix(scomm2(L0, gR0)).applyfunc(lambda e: sp.expand(e).coeff(hbar, 2))
    norm_res = sp.Matrix(star2(gR0, gR0) - I2).applyfunc(lambda e: sp.expand(e).coeff(hbar, 2))
    eqs = list((L0 * dg - dg * L0 + spec_res)) + list((gR0 * dg + dg * gR0 + norm_res))
    sol = sp.solve([sp.Eq(sp.simplify(e), 0) for e in eqs], [a0, a1, a2, a3], dict=True)
    good = bool(sol)
    if good:
        s = sol[0]
        # all four unknowns must be SOLVED (a free variable omitted by
        # dict=True must not silently read as 0 — anti-conservative)
        good = set(s.keys()) == {a0, a1, a2, a3}
    if good:
        a0v = sp.simplify(s[a0])
        a1v = sp.simplify(s[a1])
        a3v = sp.simplify(sp.together(s[a3]))
        a3n = sp.expand(sp.numer(a3v))
        good = (a0v == 0) and (a1v == 0) and (a3v != 0) \
            and a3n.has(Ddd) and a3n.has(Dd**2)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] M2: O(h2) dg fully solved with a0=a1=0 (pure t2/t3 sector) and a3 != 0 carrying both Ddd and Dd^2")
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
    d = mclaims_2026_07_07()
    print()
    print("ALL PASS" if (a and b and R2c == 0 and d) else "SOME FAILED -- inspect above")
