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

The script also guards two convention-level identities used in the text:
the main/supplement starting equations are related by antipodal trajectory
relabeling and multiplication by -i (not by a reversed Moyal kernel), and
antipodal relabeling leaves the full angular average invariant even though it
flips odd harmonics.

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


def check_conventions_and_relabeling():
    """Regression guards for M1 and the antipodal-average identity."""
    print("---- (c) convention bridge and antipodal angular relabeling ----")
    ok = True

    # If the main equation is V+iC=0, evaluation at -p changes V -> -V.
    # Multiplication by -i then gives the supplement convention iV+C=0.
    V, C = sp.symbols('V C', commutative=True)
    bridge = sp.simplify(-I_ * (-V + I_ * C) - (I_ * V + C))
    good = bridge == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] -i[-V+iC] = iV+C")

    # For scalar symbols the two half-Poisson terms in the star commutator
    # add, giving i*hbar*{A,B}_PB rather than half that coefficient.
    As = sp.Function('A')(E, t)
    Bs = sp.Function('B')(E, t)
    AM, BM = sp.Matrix([[As]]), sp.Matrix([[Bs]])
    pb = sp.diff(As, E) * sp.diff(Bs, t) - sp.diff(As, t) * sp.diff(Bs, E)
    scalar_comm = sp.simplify(scomm(AM, BM)[0] - I_ * hbar * pb)
    good = scalar_comm == 0
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] scalar [A,B]_star = i*hbar*{{A,B}}_PB")

    # Axisymmetric l=0,1,2 expansion: antipodal relabeling flips the odd
    # harmonic but preserves the full-sphere average.
    mu = sp.symbols('mu', real=True)
    a0, a1, a2 = sp.symbols('a0 a1 a2')
    p2 = (3 * mu**2 - 1) / 2
    X = a0 + a1 * mu + a2 * p2
    RX = sp.expand(X.subs(mu, -mu))
    expected = a0 - a1 * mu + a2 * p2
    parity = sp.simplify(RX - expected)
    avg = lambda expr: sp.integrate(expr, (mu, -1, 1)) / 2
    average = sp.simplify(avg(RX) - avg(X))
    good_parity = parity == 0
    good_average = average == 0
    ok &= good_parity and good_average
    print(f"  [{'PASS' if good_parity else 'FAIL'}] R_- flips l=1 and preserves l=0,2")
    print(f"  [{'PASS' if good_average else 'FAIL'}] <R_s X> = <X>")
    return ok


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
    Delta_1,Delta_2, a common phase, E above both gaps, and distributions
    h_i = fL_i 1 + fT_i t3.  Result (sympy- and hand-verified, corrected
    gA): the energy trace carries the coherence
    factor N1_1 N1_2 - N2_1 N2_2 (= 1 at matched gaps -- regular; Maki-Griffin
    heat-current factor), the transverse trace the DOS product N1_1 N1_2
    (which carries the SIS matched-gap edge singularity), NO L-T mixing.
    Conductivity and observable-current normalizations are checked separately."""
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


def build_interface_phase(D1, D2, fL1, fL2, fT1, fT2, phase):
    """Rigid ideal-BCS interface above both gaps at phase difference phase."""
    def side(D, fL, fT, chi):
        W = sp.sqrt(E**2 - D**2)
        N1, N2 = E / W, D / W
        anomalous = sp.cos(chi) * t2 + sp.sin(chi) * t1
        gR = N1 * t3 + I_ * N2 * anomalous
        gA = -gR
        h = fL * I2 + fT * t3
        return gR, gA, gR * h - h * gA, N1, N2

    # chi_1=0, chi_2=-phase, so chi_1-chi_2=phase.
    gR1, gA1, gK1, N1_1, N2_1 = side(D1, fL1, fT1, 0)
    gR2, gA2, gK2, N1_2, N2_2 = side(D2, fL2, fT2, -phase)
    CK = gR1 * gK2 + gK1 * gA2 - gR2 * gK1 - gK2 * gA1
    WL = N1_1 * N1_2 - N2_1 * N2_2 * sp.cos(phase)
    M = sp.Matrix
    return {
        'I11 phase-biased ideal BCS: longitudinal weight has cos(delta chi)':
            (M([sp.trace(CK) / 4 + 2 * WL * (fL1 - fL2)]), sp.zeros(1, 1)),
        'I12 phase-biased ideal BCS: transverse weight is DOS product':
            (M([sp.trace(t3 * CK) / 4
                + 2 * N1_1 * N1_2 * (fT1 - fT2)]), sp.zeros(1, 1)),
    }


def build_interface_complex(a1, b1, c1, d1, a2, b2, c2, d2,
                            fL1, fL2, fT1, fT2):
    """Common-phase interface with generic complex normal/anomalous spectra.

    g_i^R=(a_i+i b_i)t3+i(c_i+i d_i)t2 and causal
    g_i^A=-t3 (g_i^R)^dagger t3.  No ideal-BCS restriction is used.
    """
    def side(a, b, c, d, fL, fT):
        gR = (a + I_ * b) * t3 + I_ * (c + I_ * d) * t2
        gA = -t3 * gR.conjugate().T * t3
        h = fL * I2 + fT * t3
        return gR, gA, gR * h - h * gA

    gR1, gA1, gK1 = side(a1, b1, c1, d1, fL1, fT1)
    gR2, gA2, gK2 = side(a2, b2, c2, d2, fL2, fT2)
    CK = gR1 * gK2 + gK1 * gA2 - gR2 * gK1 - gK2 * gA1
    WL = a1 * a2 - c1 * c2
    WT = a1 * a2 + d1 * d2
    M = sp.Matrix
    return {
        'I13 complex common-phase longitudinal weight = Re(g1)Re(g2)-Re(f1)Re(f2)':
            (M([sp.trace(CK) / 4 + 2 * WL * (fL1 - fL2)]), sp.zeros(1, 1)),
        'I14 complex common-phase transverse weight = Re(g1)Re(g2)+Im(f1)Im(f2)':
            (M([sp.trace(t3 * CK) / 4 + 2 * WT * (fT1 - fT2)]), sp.zeros(1, 1)),
    }


def check_interface_normalization():
    """Side-dependent diffusion normalization and normal-state prefactors."""
    print("---- (d) K-L current normalization and normal-state limits ----")
    e, N01, N02, G, weight, jump = sp.symbols(
        'e N01 N02 G weight jump', positive=True)
    dV, kB, temp = sp.symbols('dV kB T', real=True)
    g1 = G / (2 * e**2 * N01)
    g2 = G / (2 * e**2 * N02)
    calJ = G * weight * jump
    j1, j2 = g1 * weight * jump, g2 * weight * jump
    checks = {
        'calJ = 2 e^2 N01 j1': sp.simplify(calJ - 2 * e**2 * N01 * j1),
        'calJ = 2 e^2 N02 j2': sp.simplify(calJ - 2 * e**2 * N02 * j2),
        'j1/j2 = N02/N01': sp.simplify(j1 / j2 - N02 / N01),
        'unequal-DOS guard: N01=2,N02=3 gives j1/j2=3/2':
            sp.simplify((j1 / j2).subs({N01: 2, N02: 3}) - sp.Rational(3, 2)),
        'normal voltage calibration G/(2e) x 2e dV = G dV':
            sp.simplify(G * (2 * e * dV) / (2 * e) - G * dV),
        'Wiedemann-Franz prefactor magnitude':
            sp.simplify(
                G * (2 * sp.pi**2 * kB**2 * temp / 3) / (2 * e**2)
                - sp.pi**2 * kB**2 * temp * G / (3 * e**2)),
    }
    ok = True
    for name, expr in checks.items():
        good = expr == 0
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    return ok


def check_legacy_local_peclet():
    """Operator-level drift and local Peclet diagnostic for placement C."""
    print("---- (e) legacy-C local fixed-energy Peclet diagnostic ----")
    Delta, DN, grad_delta, length = sp.symbols(
        'Delta D_N grad_Delta L_Delta', positive=True)
    N1 = E / sp.sqrt(E**2 - Delta**2)
    diffusivity = DN / N1

    # Placement C is (p,q)=(0,-1), so v_C=-D_N N1^-2 d_x N1.
    drift = sp.simplify(
        -DN * N1**-2 * sp.diff(N1, Delta) * grad_delta)
    speed = sp.simplify(-drift)
    expected_speed = DN * Delta * grad_delta / (
        E * sp.sqrt(E**2 - Delta**2))
    peclet = sp.simplify(speed * length / diffusivity)
    expected_peclet = Delta * length * grad_delta / (E**2 - Delta**2)

    ratios = (sp.Rational(21, 20), sp.Rational(11, 10),
              sp.Rational(13, 10))
    coefficients = tuple(sp.simplify(1 / (ratio**2 - 1)) for ratio in ratios)
    expected_coefficients = (sp.Rational(400, 41), sp.Rational(100, 21),
                             sp.Rational(100, 69))
    amplitudes = (sp.Rational(1, 100000), sp.Rational(1, 1000),
                  sp.Rational(1, 5))
    # (printed value, one unit in its last printed digit).
    reported = (
        ((sp.Rational(976, 10_000_000), sp.Rational(1, 10_000_000)),
         (sp.Rational(476, 10_000_000), sp.Rational(1, 10_000_000)),
         (sp.Rational(145, 10_000_000), sp.Rational(1, 10_000_000))),
        ((sp.Rational(976, 100_000), sp.Rational(1, 100_000)),
         (sp.Rational(476, 100_000), sp.Rational(1, 100_000)),
         (sp.Rational(145, 100_000), sp.Rational(1, 100_000))),
        ((sp.Rational(195, 100), sp.Rational(1, 100)),
         (sp.Rational(952, 1000), sp.Rational(1, 1000)),
         (sp.Rational(290, 1000), sp.Rational(1, 1000))),
    )

    checks = {
        '|v_C| = D_N Delta |d_x Delta| / [E sqrt(E^2-Delta^2)]':
            sp.simplify(speed - expected_speed) == 0,
        'Pe_E = Delta |delta Delta| / (E^2-Delta^2)':
            sp.simplify(peclet - expected_peclet) == 0,
        'E/Delta coefficients are 400/41, 100/21, 100/69':
            coefficients == expected_coefficients,
        'printed triples round correctly to three significant figures':
            all(abs(amplitude * coefficient - shown) < unit / 2
                for amplitude, row in zip(amplitudes, reported)
                for coefficient, (shown, unit) in zip(coefficients, row)),
    }
    ok = True
    for name, good in checks.items():
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    return ok


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
    c = check_conventions_and_relabeling()
    print()
    a3 = check_symbolic(build_interface(*sp.symbols('Delta_1 Delta_2', positive=True),
                                        *sp.symbols('fL1 fL2 fT1 fT2', real=True)))
    print()
    b3 = check_numeric(build_interface(sp.Integer(1), sp.Rational(13, 10),
                                       sp.Rational(2, 5), sp.Rational(1, 7),
                                       sp.Rational(1, 3), sp.Rational(1, 9)),
                       {E: 2})
    print()
    phase = sp.symbols('delta_chi', real=True)
    a4 = check_symbolic(build_interface_phase(
        *sp.symbols('Delta_1 Delta_2', positive=True),
        *sp.symbols('fL1 fL2 fT1 fT2', real=True), phase))
    print()
    b4 = check_numeric(build_interface_phase(
        sp.Integer(1), sp.Rational(13, 10),
        sp.Rational(2, 5), sp.Rational(1, 7),
        sp.Rational(1, 3), sp.Rational(1, 9), phase),
        {E: 2, phase: sp.Rational(2, 5)})
    print()
    complex_args = sp.symbols(
        'a1 b1 c1 d1 a2 b2 c2 d2 fL1 fL2 fT1 fT2', real=True)
    a5 = check_symbolic(build_interface_complex(*complex_args))
    print()
    b5 = check_numeric(
        build_interface_complex(
            *map(sp.Rational, (4, 7, 2, 5, 3, 8, 1, 6, 2, 9, 1, 10))),
        {})
    print()
    d = check_interface_normalization()
    print()
    e = check_legacy_local_peclet()
    print()
    all_ok = (a and b and a2 and b2 and c and a3 and b3
              and a4 and b4 and a5 and b5 and d and e)
    print("ALL PASS" if all_ok else "SOME FAILED -- inspect above")
    raise SystemExit(0 if all_ok else 1)
