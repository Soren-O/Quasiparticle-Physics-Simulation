"""
Symbolic verification that the dirty-limit longitudinal (A1) projection is
complete and uncontaminated for a COMBINED space- and time-dependent gap
Delta(x,t) -- the self-consistent-feedback background, where the gap both
varies along the film and moves in time because the occupation suppresses it.

The paper establishes the two limits separately (verify_traces.py):
  - uniform Delta(t):  fixed-E plain trace -> conservative spectral flow (I2)
  - static  Delta(x):  undressed longitudinal spatial current, no grad-Delta
                       source, no L-T mixing (I7/I8)
This script promotes Delta -> Delta(x,t), f_{L,T} -> f_{L,T}(E,x,t) and checks
that the two reductions compose with LOCAL coefficients and produce no new
terms: no Ddot-gradDelta cross terms in the longitudinal channel, no L<->T
mixing from the drive, and the first Moyal (O(hbar)) correction to gK is
sourced ONLY by f_T -- so it cannot feed the energy mode at the next order
either. Above the local gap edge throughout (gA = -gR there).

Conventions identical to verify_traces.py (corrected gA of
verify_gA_convention.py):
  Moyal:  A*B = AB + (i hbar/2)(dE A . dt B - dt A . dE B)   [(E,t) pair]
  gauge:  L0 = E tau3 + i Delta tau2
  BCS:    gR = N1 tau3 + i N2 tau2,  gA = -gR (above gap),  N1=E/W, N2=Delta/W
  dist:   h  = f_L 1 + f_T tau3,    gK = gR h - h gA (leading SSLO ansatz)
  space:  Usadel current JK = gR dx(gK) + gK dx(gA), plain trace = energy mode
"""
import sympy as sp

E, t, hbar = sp.symbols('E t hbar', real=True, positive=True)
x = sp.symbols('x', real=True)
I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])


def dE(M): return M.applyfunc(lambda e: sp.diff(e, E))
def dt(M): return M.applyfunc(lambda e: sp.diff(e, t))
def dx(M): return M.applyfunc(lambda e: sp.diff(e, x))
def star(A, B): return A * B + (I_ * hbar / 2) * (dE(A) * dt(B) - dt(A) * dE(B))
def scomm(A, B): return star(A, B) - star(B, A)


def build_combined(Delta, fL, fT):
    """All identities at once for Delta(x,t), f_L(E,x,t), f_T(E,x,t)."""
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    L0 = E * t3 + I_ * Delta * t2
    gR = N1 * t3 + I_ * N2 * t2
    gA = -N1 * t3 - I_ * N2 * t2
    h = fL * I2 + fT * t3
    gK = gR * h - h * gA                                   # leading ansatz
    gK_L = gR * (fL * I2) - (fL * I2) * gA                 # f_L sector alone
    gK_T = gR * (fT * t3) - (fT * t3) * gA                 # f_T sector alone
    dgK = (star(gR, h) - gR * h) - (star(h, gA) - h * gA)  # 1st Moyal corr.
    dgK_L = (star(gR, fL * I2) - gR * fL) - (star(fL * I2, gA) - fL * gA)

    JK = gR * dx(gK) + gK * dx(gA)          # leading spatial current
    JK1 = gR * dx(dgK) + dgK * dx(gA)       # O(hbar) cross-order current

    M = sp.Matrix
    return {
        # time side, energy mode: the fixed-E plain trace still produces the
        # canonical conservative spectral flow with LOCAL Delta(x,t) -- the
        # spatial dependence is a spectator of the (E,t) Moyal bracket, so no
        # dx(Delta) term can enter here
        'J1  plaintrace i[L0,gK]_star = -4hbar[dt(N1 fL)+dE(Dd N2 fL)], local Delta(x,t)':
            (M([(I_ * scomm(L0, gK)).trace()
                - (-4 * hbar * (sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E)))]),
             sp.zeros(1, 1)),
        # time side, no mixing either way (even with dx(Delta) != 0):
        'J2  tau3-trace of f_L-sector time side = 0 (drive sources no f_T)':
            (M([(I_ * scomm(L0, gK_L) * t3).trace()]), sp.zeros(1, 1)),
        'J2b plaintrace of f_T-sector time side = 0 (f_T feeds no energy mode)':
            (M([(I_ * scomm(L0, gK_T)).trace()]), sp.zeros(1, 1)),
        # space side: undressed longitudinal current with t a spectator; every
        # dx(Delta) spectral-gradient term cancels even while Delta also moves
        # in time
        'J3  (1/4)Tr[JK] = dx fL (D_L = 1, no grad-Delta source, local in t)':
            (M([sp.Rational(1, 4) * JK.trace() - sp.diff(fL, x)]), sp.zeros(1, 1)),
        'J4  (1/4)Tr[t3 JK] = N1^2 dx fT (transverse dressing, no mixing)':
            (M([sp.Rational(1, 4) * (t3 * JK).trace() - N1**2 * sp.diff(fT, x)]),
             sp.zeros(1, 1)),
        # cross order: the first Moyal correction to gK is sourced ONLY by f_T
        # (the f_L part of h multiplies the identity and drops from the
        # derivative commutators) ...
        'J5  dgK with f_T = 0 vanishes identically (no f_L at cross order)':
            (dgK_L, sp.zeros(2, 2)),
        # ... and its spatial current cannot feed the energy mode: the would-be
        # hbar * Ddot * dxDelta cross term traces to zero in the plain channel
        'J5b plaintrace[gR dx(dgK) + dgK dx(gA)] = 0 (no cross term in energy mode)':
            (M([sp.Rational(1, 4) * JK1.trace()]), sp.zeros(1, 1)),
        # assembled statement: conservative form == advective A1 form with
        # local Delta(x,t) (DOS continuity holds pointwise in x)
        'J6  dt(N1 fL)+dE(Dd N2 fL) = N1[dt + (Delta/E) Dd dE] fL, local Delta(x,t)':
            (M([sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E)
                - N1 * (sp.diff(fL, t) + (Delta / E) * Dd * sp.diff(fL, E))]),
             sp.zeros(1, 1)),
    }


def check_symbolic(out):
    print("---- (a) symbolic simplify, generic Delta(x,t), f_L(E,x,t), f_T(E,x,t) ----")
    allok = True
    for name, (expr, exp) in out.items():
        d = sp.simplify(sp.Matrix(expr) - sp.Matrix(exp))
        ok = all(e == 0 for e in d)
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print("        residual:", list(d))
    return allok


def check_numeric(out, subs):
    print(f"---- (b) numeric substitution {subs} ----")
    allok = True
    for name, (expr, exp) in out.items():
        d = (sp.Matrix(expr) - sp.Matrix(exp)).subs(subs)
        m = max(abs(complex(sp.N(e))) for e in d)
        ok = m < 1e-9
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] max|.|={m:.2e}  {name}")
    return allok


if __name__ == "__main__":
    a = check_symbolic(build_combined(sp.Function('Delta')(x, t),
                                      sp.Function('f_L')(E, x, t),
                                      sp.Function('f_T')(E, x, t)))
    print()
    # E=2 stays above the local gap: Delta(1/3, 1/2) ~ 1.215
    b = check_numeric(build_combined(1 + sp.Rational(3, 10) * t
                                       + sp.Rational(1, 5) * sp.sin(x),
                                     sp.exp(-E) * sp.cos(t) * sp.cos(x),
                                     sp.sin(E) * sp.exp(-t) * sp.exp(-x)),
                      {E: 2, t: sp.Rational(1, 2), x: sp.Rational(1, 3), hbar: 1})
    print()
    print("ALL PASS" if (a and b) else "SOME FAILED -- inspect above")
