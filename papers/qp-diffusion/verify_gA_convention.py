"""Side-by-side symbolic audit of the two advanced-propagator conventions.

Context (2026-06-09 audit): paper.tex, Chapter4Appendix.tex, and all four
verify_*.py scripts use

    PAPER:      gA = -N1 tau3 + i N2 tau2        ( = -(gR)^dagger )

while the standard quasiclassical relation for this parameterization is

    CORRECTED:  gA = -N1 tau3 - i N2 tau2        ( = -tau3 (gR)^dagger tau3 )

(Belzig et al. 1999; Silaev-Virtanen-Bergeret-Heikkila PRL 114, 167002 (2015),
which states gA = -tau3 gR^dagger tau3 explicitly.)

This script evaluates every contested identity under BOTH conventions, using
the same gauge, Moyal product, SSLO ansatz, and trace projections as
verify_traces.py (gA is the only switch):

  T1  normalization (gA)^2 = 1                  [both pass -> cannot distinguish]
  T2  advanced spectral equation [L0, gA]       [paper: 4 E N2 tau1 (the paper's
                                                 own eq:specA); corrected: 0.
                                                 R and A satisfy the SAME static
                                                 spectral equation, so 0 is required]
  T3  equilibrium Keldysh anomalous component   [tau2 part of (gR-gA):
                                                 paper: 0 -> gap equation gives
                                                 Delta = 0; corrected: 2 N2]
  T4  D_L = (1/4)Tr[1 - gR gA]                  [paper: N1^2; corrected: 1]
  T5  D_T = (1/4)Tr[1 - gR t3 gA t3]            [paper: 1;    corrected: N1^2]
  T6  depairing R = (1/4)Tr[hatDelta (gR+gA)]   [paper: Delta*N2; corrected: 0
                                                 (literature: R ~ Delta*Im sinh(theta),
                                                 = 0 for ideal BCS above the gap)]
  T7  plain trace of i[L0, gK]_star             [paper: -4 hbar dt(N1 fL) only;
                                                 corrected: -4 hbar (dt(N1 fL)
                                                 + dE(Ddot N2 fL)) -- the canonical
                                                 spectral-flow flux IS produced by
                                                 the fixed-E projection]
  T8  static Delta(x): spatial current JK       [paper:    Tr/4 = N1^2 dx fL,
                                                            t3-Tr/4 = dx fT;
                                                 corrected: Tr/4 = dx fL,
                                                            t3-Tr/4 = N1^2 dx fT;
                                                 all grad-Delta terms cancel in both]
  T9  Kupriyanov-Lukichev commutator [g1,g2]^K  [paper:    energy weight N1N1',
                                                            charge weight N1N1'-N2N2';
                                                 corrected: energy weight N1N1'-N2N2',
                                                            charge weight N1N1']

Run:  python verify_gA_convention.py   (needs sympy)
Every line should print PASS; the point is WHICH closed form each convention
produces, shown side by side.
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


def spectral(Delta, sign):
    """BCS bulk propagators above the gap; sign = +1 paper, -1 corrected."""
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    gR = N1 * t3 + I_ * N2 * t2
    gA = -N1 * t3 + sign * I_ * N2 * t2
    return gR, gA, N1, N2


def is_zero(M):
    return all(sp.simplify(e) == 0 for e in sp.Matrix(M))


def report(label, results):
    """results: list of (convention, expression, claimed_closed_form)."""
    line = f"{label:<58s}"
    for conv, expr, claim in results:
        ok = is_zero(sp.Matrix(expr) - sp.Matrix(claim))
        line += f" | {conv}: {'PASS' if ok else 'FAIL'}"
    print(line)


Delta = sp.Function('Delta')(t)
fL = sp.Function('f_L')(E, t)
fT = sp.Function('f_T')(E, t)
Dd = sp.diff(Delta, t)
L0 = E * t3 + I_ * Delta * t2          # gap gauge of the paper: hatDelta = -i Delta tau2
hatD = -I_ * Delta * t2

conventions = {'paper': +1, 'corrected': -1}
built = {}
for name, sgn in conventions.items():
    gR, gA, N1, N2 = spectral(Delta, sgn)
    h = fL * I2 + fT * t3
    built[name] = dict(gR=gR, gA=gA, N1=N1, N2=N2, gK=gR * h - h * gA)

N1s, N2s = built['paper']['N1'], built['paper']['N2']   # same scalars either way

print(__doc__.splitlines()[0])
print('=' * 100)

# T1 normalization
report('T1  (gA)^2 = 1 (both normalized - cannot distinguish)', [
    (n, built[n]['gA'] * built[n]['gA'], I2) for n in conventions])

# T2 advanced spectral equation
report('T2  [L0,gA]: paper = 4 E N2 t1 (eq:specA), corrected = 0', [
    ('paper', (L0 * built['paper']['gA'] - built['paper']['gA'] * L0),
     4 * E * N2s * t1),
    ('corrected', (L0 * built['corrected']['gA'] - built['corrected']['gA'] * L0),
     sp.zeros(2, 2))])

# T3 equilibrium anomalous Keldysh weight: (gR-gA) tau2 coefficient
report('T3  (gR-gA): paper = 2N1 t3 (no N2 -> Delta=0), corr = 2N1 t3 + 2iN2 t2', [
    ('paper', built['paper']['gR'] - built['paper']['gA'], 2 * N1s * t3),
    ('corrected', built['corrected']['gR'] - built['corrected']['gA'],
     2 * N1s * t3 + 2 * I_ * N2s * t2)])

# T4 / T5 spectral diffusion coefficients
report('T4  D_L = (1/4)Tr[1-gR gA]: paper = N1^2, corrected = 1', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * (I2 - built['paper']['gR'] * built['paper']['gA']).trace()]),
     sp.Matrix([N1s**2])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * (I2 - built['corrected']['gR'] * built['corrected']['gA']).trace()]),
     sp.Matrix([1]))])
report('T5  D_T = (1/4)Tr[1-gR t3 gA t3]: paper = 1, corrected = N1^2', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * (I2 - built['paper']['gR'] * t3 * built['paper']['gA'] * t3).trace()]),
     sp.Matrix([1])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * (I2 - built['corrected']['gR'] * t3 * built['corrected']['gA'] * t3).trace()]),
     sp.Matrix([N1s**2]))])

# T6 depairing coefficient (Belzig eq. 39)
report('T6  R = (1/4)Tr[hatD (gR+gA)]: paper = Delta N2, corrected = 0', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * (hatD * (built['paper']['gR'] + built['paper']['gA'])).trace()]),
     sp.Matrix([Delta * N2s])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * (hatD * (built['corrected']['gR'] + built['corrected']['gA'])).trace()]),
     sp.Matrix([0]))])

# T7 time-dependent plain-trace projection (the App-B question)
report('T7  Tr i[L0,gK]_star: paper = -4hb dt(N1 fL); corr adds -4hb dE(Dd N2 fL)', [
    ('paper', sp.Matrix([(I_ * scomm(L0, built['paper']['gK'])).trace()]),
     sp.Matrix([-4 * hbar * sp.diff(N1s * fL, t)])),
    ('corrected', sp.Matrix([(I_ * scomm(L0, built['corrected']['gK'])).trace()]),
     sp.Matrix([-4 * hbar * (sp.diff(N1s * fL, t) + sp.diff(Dd * N2s * fL, E))]))])

# T8 static spatially varying gap: leading dirty-limit spatial current
Dx_ = sp.Function('Delta')(x)
fLx = sp.Function('f_L')(E, x)
fTx = sp.Function('f_T')(E, x)
spat = {}
for name, sgn in conventions.items():
    gR, gA, N1x, N2x = spectral(Dx_, sgn)
    hx = fLx * I2 + fTx * t3
    gKx = gR * hx - hx * gA
    spat[name] = (gR * dx(gKx) + gKx * dx(gA), N1x)
report('T8a (1/4)Tr[JK]: paper = N1^2 dx fL, corrected = dx fL (no grad-D source)', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * spat['paper'][0].trace()]),
     sp.Matrix([spat['paper'][1]**2 * sp.diff(fLx, x)])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * spat['corrected'][0].trace()]),
     sp.Matrix([sp.diff(fLx, x)]))])
report('T8b (1/4)Tr[t3 JK]: paper = dx fT, corrected = N1^2 dx fT', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * (t3 * spat['paper'][0]).trace()]),
     sp.Matrix([sp.diff(fTx, x)])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * (t3 * spat['corrected'][0]).trace()]),
     sp.Matrix([spat['corrected'][1]**2 * sp.diff(fTx, x)]))])

# T9 Kupriyanov-Lukichev interface commutator, independent gaps
D1, D2 = sp.symbols('Delta_1 Delta_2', positive=True)
fL1, fL2, fT1, fT2 = sp.symbols('fL1 fL2 fT1 fT2', real=True)
kl = {}
for name, sgn in conventions.items():
    def side(D, fl, ft, sgn=sgn):
        gR, gA, n1, n2 = spectral(D, sgn)
        hh = fl * I2 + ft * t3
        return gR, gA, gR * hh - hh * gA, n1, n2
    gR1, gA1, gK1, n11, n21 = side(D1, fL1, fT1)
    gR2, gA2, gK2, n12, n22 = side(D2, fL2, fT2)
    kl[name] = (gR1 * gK2 + gK1 * gA2 - gR2 * gK1 - gK2 * gA1, n11, n21, n12, n22)
report('T9a (1/4)Tr[CK] (energy): paper = -2 N1N1 dfL, corr = -2(N1N1-N2N2) dfL', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * kl['paper'][0].trace()]),
     sp.Matrix([-2 * kl['paper'][1] * kl['paper'][3] * (fL1 - fL2)])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * kl['corrected'][0].trace()]),
     sp.Matrix([-2 * (kl['corrected'][1] * kl['corrected'][3]
                      - kl['corrected'][2] * kl['corrected'][4]) * (fL1 - fL2)]))])
report('T9b (1/4)Tr[t3 CK] (charge): paper = -2(N1N1-N2N2) dfT, corr = -2 N1N1 dfT', [
    ('paper', sp.Matrix([sp.Rational(1, 4) * (t3 * kl['paper'][0]).trace()]),
     sp.Matrix([-2 * (kl['paper'][1] * kl['paper'][3]
                      - kl['paper'][2] * kl['paper'][4]) * (fT1 - fT2)])),
    ('corrected', sp.Matrix([sp.Rational(1, 4) * (t3 * kl['corrected'][0]).trace()]),
     sp.Matrix([-2 * kl['corrected'][1] * kl['corrected'][3] * (fT1 - fT2)]))])

print('=' * 100)
print('Physical anchors selecting the corrected column: T2 (gA must solve the same')
print('static spectral equation as gR), T3 (equilibrium anomalous Keldysh weight must')
print('be 2 N2 tanh(E/2T) or the gap equation yields Delta = 0), Belzig 1999 text')
print('(D_L = 0 below the gap), PRL 114 167002 Eqs. (15)-(16), dirty-limit kappa_s,')
print('sub-gap Andreev (charge penetrates, energy blocked), SIS I-V (N1N1 = charge).')
