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
    if isinstance(fL, AppliedUndef):     # structural checks only make sense symbolically
        out['t3 channel has NO fL at O(h1),O(h2)'] = (
            sp.Matrix([[sp.simplify(chan(t3, 1).diff(fL) + chan(t3, 2).diff(fL))]]), sp.zeros(1, 1))
        out['coh t1 O(h2) sourced by fL only (no fT)'] = (
            sp.Matrix([[sp.simplify(chan(t1, 2).diff(fT))]]), sp.zeros(1, 1))
    if isinstance(fT, AppliedUndef):
        out['plain channel has NO fT at O(h1),O(h2)'] = (
            sp.Matrix([[sp.simplify(chan(I2, 1).diff(fT) + chan(I2, 2).diff(fT))]]), sp.zeros(1, 1))
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
    print("ALL PASS" if (a and b and R2c == 0) else "SOME FAILED -- inspect above")
