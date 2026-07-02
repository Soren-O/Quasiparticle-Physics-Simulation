"""
Independent symbolic verification of the load-bearing trace identities of the
uniform-gap time-dependent dirty-limit projection (paper.tex body + App A/B).

2026-06-09 errata: gA flipped to the physical continuation
gA = -tau3 (gR)^dagger tau3 = -N1 tau3 - i N2 tau2 (= -gR above the gap), per
verify_gA_convention.py (its corrected column is the source of truth for every
expected value below). Reversals vs the superseded convention: the tau2 drive
is NOT annihilated between gR and gA (I3), gA solves the same spectral
equation as gR (I4), the fixed-E plain trace DOES produce the canonical
spectral-flow flux (I2), D_L = 1 / D_T = N1^2 (I5), and the spatial and
interface channel weights swap (I7/I8, I9/I10).

Method is independent of the hand Pauli-algebra route: matrices are carried as
explicit 2x2 sympy objects, the Moyal product / commutator and all E,t
derivatives are done by sympy, and every identity is checked twice ---
(a) by symbolic simplify with generic Delta(t), f_L(E,t), f_T(E,t), and
(b) by numeric substitution with concrete generic functions (guards against
simplify failing to reduce a true zero).

Conventions (parent note):
  Moyal:  A*B = AB + (i hbar/2)(dE A . dt B - dt A . dE B)   [(E,t) pair]
  gauge:  L0 = E tau3 - hatDelta,  hatDelta = -i Delta tau2  =>  L0 = E tau3 + i Delta tau2
  BCS:    gR = N1 tau3 + i N2 tau2,  gA = -N1 tau3 - i N2 tau2,  N1=E/W, N2=Delta/W, W=sqrt(E^2-Delta^2)
  dist:   h  = f_L 1 + f_T tau3,    gK = gR h - h gA (leading SSLO ansatz)
"""
import sympy as sp

E, t, hbar = sp.symbols('E t hbar', real=True, positive=True)
x = sp.symbols('x', real=True)
I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])


def dE(M): return M.applyfunc(lambda x: sp.diff(x, E))
def dt(M): return M.applyfunc(lambda x: sp.diff(x, t))
def star(A, B): return A * B + (I_ * hbar / 2) * (dE(A) * dt(B) - dt(A) * dE(B))
def scomm(A, B): return star(A, B) - star(B, A)


def build(Delta, fL, fT):
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    L0 = E * t3 + I_ * Delta * t2
    gR = N1 * t3 + I_ * N2 * t2
    gA = -N1 * t3 - I_ * N2 * t2
    h = fL * I2 + fT * t3
    gK = gR * h - h * gA                                   # leading ansatz
    dgK = (star(gR, h) - gR * h) - (star(h, gA) - h * gA)  # 1st Moyal correction to gK

    M = sp.Matrix
    return {
        # 1: instantaneous BCS propagator solves the adiabatic retarded Usadel eq
        'I1  [L0,gR]_star = 0':              (scomm(L0, gR),                 sp.zeros(2, 2)),
        'I1b dtN1 + Ddot*dEN2 = 0 (DOS-cty)':(M([sp.diff(N1, t) + Dd * sp.diff(N2, E)]), sp.zeros(1, 1)),
        # 3: spectral-flow drive (tau2 sector) is NOT annihilated between gR and
        #    gA: gR.t2 - t2.gA = 2i N2 (this is what lets the drive survive the
        #    plain trace and produce the I2 flux term)
        'I3  gR.t2 - t2.gA = 2i N2':         (gR * t2 - t2 * gA,            2 * I_ * N2 * I2),
        # 4: advanced propagator solves the SAME adiabatic spectral equation
        #    (gA = -gR above the gap, so this holds including the O(hbar) piece)
        'I4  [L0,gA]_star = 0':              (scomm(L0, gA),                 sp.zeros(2, 2)),
        # 5: longitudinal flux undressed (D_L = 1), transverse dressed (D_T = N1^2)
        'I5  DL = 1':                        (M([sp.Rational(1, 4) * (I2 - gR * gA).trace() - 1]), sp.zeros(1, 1)),
        'I5  DT = N1^2':                     (M([sp.Rational(1, 4) * (I2 - gR * t3 * gA * t3).trace() - N1**2]), sp.zeros(1, 1)),
        # 2: plain trace of i[L0,gK]_star = -4 hbar [dt(N1 fL) + dE(Ddot N2 fL)]:
        #    the canonical conservative spectral-flow form, produced directly by
        #    the fixed-E projection; still no fT coupling
        'I2  plaintrace = -4hbar [dt(N1 fL) + dE(Dd N2 fL)]':
            (M([(I_ * scomm(L0, gK)).trace()
                - (-4 * hbar * (sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E)))]), sp.zeros(1, 1)),
        'I2b fT-sector plaintrace = 0':
            (M([(I_ * scomm(L0, gR * (fT * t3) - (fT * t3) * gA)).trace()]), sp.zeros(1, 1)),
        'I2c Tr i[L0, dgK] = 0 (1st Wigner corr)':
            (M([(I_ * (L0 * dgK - dgK * L0)).trace()]), sp.zeros(1, 1)),
        # 6: exact discrepancy (fixed-E conservative) vs (moving-xi advective)
        'I6  dt(N1 fL)+dE(Ddot N2 fL) = N1(dt+ (D/E)Ddot dE)fL':
            (M([sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E)
                - N1 * (sp.diff(fL, t) + (Delta / E) * Dd * sp.diff(fL, E))]), sp.zeros(1, 1)),
    }


def check_symbolic(out):
    print("---- (a) symbolic simplify, generic Delta(t), f_L(E,t), f_T(E,t) ----")
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


def build_spatial(Delta, fL, fT):
    """Spatially-varying STATIC gap Delta(x): leading-order dirty-limit
    Keldysh-Usadel spatial current  JK = gR dx(gK) + gK dx(gA).  Channel
    assignment (not in dispute): the longitudinal f_L current is the PLAIN
    Nambu trace; the f_T current is the t3-trace.  Corrected gA: the f_L
    current is UNDRESSED (D_L = 1) and the f_T current carries N1^2 (D_T).
    Result (sympy- and hand-verified): all dx(Delta) spectral-gradient terms
    cancel, so there is NO grad(Delta) source and NO f_L-f_T mixing at this order."""
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    gR = N1 * t3 + I_ * N2 * t2
    gA = -N1 * t3 - I_ * N2 * t2
    h = fL * I2 + fT * t3
    gK = gR * h - h * gA

    def dxg(Mx):
        return Mx.applyfunc(lambda e: sp.diff(e, x))

    JK = gR * dxg(gK) + gK * dxg(gA)
    M = sp.Matrix
    return {
        # longitudinal spatial current is cleanly dx fL (D_L = 1): every dx(Delta)
        # spectral-gradient term cancels -> no grad(Delta) source, no L-T mixing
        'I7  (1/4)Tr[JK] = dx fL (longitudinal=plain trace, D_L=1)':
            (M([sp.Rational(1, 4) * JK.trace() - sp.diff(fL, x)]), sp.zeros(1, 1)),
        # transverse channel dressed (D_T = N1^2), cleanly separate from f_L
        'I8  (1/4)Tr[t3 JK] = N1^2 dx fT (transverse, D_T=N1^2)':
            (M([sp.Rational(1, 4) * (t3 * JK).trace() - N1**2 * sp.diff(fT, x)]), sp.zeros(1, 1)),
    }


def build_interface(D1, D2, fL1, fL2, fT1, fT2):
    """Kupriyanov-Lukichev interface BC, tunnel limit: the scalar interface
    current is the longitudinal projection of the Keldysh component of the
    matrix-current commutator [g_1,g_2].  Two sides with INDEPENDENT gaps
    Delta_1,Delta_2 and distributions h_i = fL_i 1 + fT_i t3.  Result (sympy-
    and hand-verified, corrected gA): the energy current carries the coherence
    factor N1_1 N1_2 - N2_1 N2_2 (= 1 at matched gaps -- regular; Maki-Griffin
    heat-current factor), the charge current the DOS product N1_1 N1_2 (which
    carries the SIS matched-gap edge singularity), NO L-T mixing."""
    def side(D, fL, fT):
        W = sp.sqrt(E**2 - D**2)
        N1, N2 = E / W, D / W
        gR = N1 * t3 + I_ * N2 * t2
        gA = -N1 * t3 - I_ * N2 * t2
        h = fL * I2 + fT * t3
        gK = gR * h - h * gA
        return gR, gA, gK, N1, N2

    gR1, gA1, gK1, N1_1, N2_1 = side(D1, fL1, fT1)
    gR2, gA2, gK2, N1_2, N2_2 = side(D2, fL2, fT2)
    CK = gR1 * gK2 + gK1 * gA2 - gR2 * gK1 - gK2 * gA1   # Keldysh block of [g_1,g_2]
    M = sp.Matrix
    return {
        # longitudinal (energy) interface current: coherence-factor weight
        # N1N1 - N2N2 (= 1 at matched gaps); drives the fL jump, no L-T mixing
        'I9  (1/4)Tr[CK] = -2 (N1_1 N1_2 - N2_1 N2_2)(fL1-fL2) (energy)':
            (M([sp.Rational(1, 4) * CK.trace()
                - (-2 * (N1_1 * N1_2 - N2_1 * N2_2) * (fL1 - fL2))]), sp.zeros(1, 1)),
        # charge (transverse) interface current: DOS-product weight N1N1
        # (carries the SIS matched-gap edge singularity)
        'I10 (1/4)Tr[t3 CK] = -2 N1_1 N1_2 (fT1-fT2) (charge, no L-T mixing)':
            (M([sp.Rational(1, 4) * (t3 * CK).trace()
                - (-2 * N1_1 * N1_2 * (fT1 - fT2))]), sp.zeros(1, 1)),
    }


if __name__ == "__main__":
    a = check_symbolic(build(sp.Function('Delta')(t),
                             sp.Function('f_L')(E, t),
                             sp.Function('f_T')(E, t)))
    print()
    b = check_numeric(build(1 + sp.Rational(3, 10) * t,
                            sp.exp(-E) * sp.cos(t),
                            sp.sin(E) * sp.exp(-t)),
                      {E: 2, t: sp.Rational(1, 2), hbar: 1})
    print()
    a2 = check_symbolic(build_spatial(sp.Function('Delta')(x),
                                      sp.Function('f_L')(E, x),
                                      sp.Function('f_T')(E, x)))
    print()
    b2 = check_numeric(build_spatial(1 + sp.Rational(1, 5) * sp.sin(x),
                                     sp.exp(-E) * sp.cos(x),
                                     sp.sin(E) * sp.exp(-x)),
                       {E: 2, x: sp.Rational(1, 3)})
    print()
    a3 = check_symbolic(build_interface(*sp.symbols('Delta_1 Delta_2', positive=True),
                                        *sp.symbols('fL1 fL2 fT1 fT2', real=True)))
    print()
    b3 = check_numeric(build_interface(sp.Integer(1), sp.Rational(13, 10),
                                       sp.Rational(2, 5), sp.Rational(1, 7),
                                       sp.Rational(1, 3), sp.Rational(1, 9)),
                       {E: 2})
    print()
    print("ALL PASS" if (a and b and a2 and b2 and a3 and b3)
          else "SOME FAILED -- inspect above")
