"""
Exploratory check (beyond-assumption frontier): does a supercurrent / phase
gradient reopen f_L-f_T (longitudinal-transverse) mixing that is absent for a
real gap (session results I7/I8)?

Mechanism: gauge-covariant gradient  cov(M) = d_x M - i Q [t3, M], with Q the
superfluid momentum (phase gradient p_s).  Real-gap BCS spectral functions
(uniform |Delta| here, to isolate the supercurrent effect); 2026-06-09 errata:
gA flipped to the physical continuation (verify_gA_convention.py):
  gR = N1 t3 + i N2 t2,  gA = -N1 t3 - i N2 t2,  N1=E/W, N2=Delta/W, W=sqrt(E^2-Delta^2)
Distribution h = fL(x) 1 + fT(x) t3,  gK = gR h - h gA.
Matrix current (Keldysh block, gauge-covariant):
  JK = gR cov(gK) + gK cov(gA)
Projections: longitudinal jL=(1/4)Tr[JK], transverse jT=(1/4)Tr[t3 JK].

Expect: at Q=0, jL = d_x fL with NO fT (recovers corrected I7, D_L = 1) and
jT = N1^2 d_x fT (D_T).  Result (re-verified under the corrected gA): the
frozen-spectrum Q-terms cancel identically at ALL Q -- jL = d_x fL,
jT = N1^2 d_x fT, no L-T mixing -- so the supercurrent coupling enters only
through the DEPAIRED spectrum. Its fixed-energy outer expansion is O(Q^3),
but the rounded BCS edge is nonuniform and carries O(Q) integrated weight.
That computation involves only the retarded angle and is gA-independent.
"""
import sympy as sp
import mpmath as mp
from time import perf_counter

E, x = sp.symbols('E x', real=True, positive=True)
Q = sp.symbols('Q', real=True)          # superfluid momentum / phase gradient
Delta = sp.symbols('Delta', real=True, positive=True)
I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])

fL = sp.Function('f_L')(x)
fT = sp.Function('f_T')(x)

W = sp.sqrt(E**2 - Delta**2)
N1, N2 = E / W, Delta / W
gR = N1 * t3 + I_ * N2 * t2
gA = -N1 * t3 - I_ * N2 * t2
h = fL * I2 + fT * t3
gK = gR * h - h * gA


def comm(A, B):
    return A * B - B * A


def cov(M):
    """gauge-covariant gradient: d_x M - i Q [t3, M]"""
    dM = M.applyfunc(lambda e: sp.diff(e, x))
    return dM - I_ * Q * comm(t3, M)


JK = gR * cov(gK) + gK * cov(gA)
jL = sp.expand(sp.trace(JK) / 4)
jT = sp.expand(sp.trace(t3 * JK) / 4)

dfL, dfT = sp.diff(fL, x), sp.diff(fT, x)


def report(j, name):
    print(f"=== {name} ===")
    cL, cT = sp.simplify(j.coeff(dfL)), sp.simplify(j.coeff(dfT))
    rem = sp.expand(j - j.coeff(dfL) * dfL - j.coeff(dfT) * dfT)
    cfL, cfT = sp.simplify(rem.coeff(fL)), sp.simplify(rem.coeff(fT))
    rem2 = sp.simplify(rem - rem.coeff(fL) * fL - rem.coeff(fT) * fT)
    print(f"  d_x f_L : {cL}")
    print(f"  d_x f_T : {cT}")
    print(f"  f_L     : {cfL}")
    print(f"  f_T     : {cfT}")
    print(f"  remainder: {rem2}")
    print()


report(jL, "longitudinal  jL = (1/4)Tr[JK]")
report(jT, "transverse    jT = (1/4)Tr[t3 JK]")
assert sp.simplify(jL - dfL) == 0
assert sp.simplify(jT - N1**2 * dfT) == 0

print("-- Q=0 sanity (should be jL = d_x fL, jT = N1^2 d_x fT, no mixing) --")
print("  jL|Q=0:", sp.simplify(jL.subs(Q, 0)))
print("  jT|Q=0:", sp.simplify(jT.subs(Q, 0)))


# ================================================================
# DEPAIRED SPECTRUM: where the supercurrent L-T coupling appears
# Usadel depairing eqn: E sinh(t) - D cosh(t) + i*Gamma*sinh(t)*cosh(t) = 0,
#   Gamma = 2 hbar D Q^2.  L-T coupling = S(E) = 2 Q Im[sinh^2(theta)].
# ================================================================
print("\n" + "=" * 62)
print("DEPAIRED SPECTRUM: L-T coupling S(E) = 2Q Im[sinh^2(theta)]")
print("=" * 62)
Wr = sp.symbols('W', real=True, positive=True)        # W = sqrt(E^2 - Delta^2)
Er, Dr, Gam, Qs = sp.symbols('E Delta Gamma Q', real=True, positive=True)
s0, c0 = Dr / Wr, Er / Wr                              # BCS: tanh(theta0)=Delta/E
th1 = -sp.I * s0 * c0 / Wr                             # theta = theta0 + Gamma*th1 + ...
sinh2 = sp.expand(s0**2 + 2 * s0 * c0 * Gam * th1)     # sinh^2(theta) to O(Gamma)
Im_s2 = sp.simplify(sp.im(sinh2))
print("Im[sinh^2(theta)] =", Im_s2, "  (expect -2 Gamma E^2 Delta^2 / W^5)")
assert sp.simplify(Im_s2 + 2 * Gam * Er**2 * Dr**2 / Wr**5) == 0
S = sp.simplify(2 * Qs * Im_s2)
print("S = 2Q Im[sinh^2]  =", S)
hbar, Dco = sp.symbols('hbar D', real=True, positive=True)
print("with Gamma=2 hbar D Q^2:  S =", sp.simplify(S.subs(Gam, 2 * hbar * Dco * Qs**2)),
      " => fixed-energy outer O(Q^3)")
print("BCS limit Gamma->0:  S =", S.subs(Gam, 0), "  (no mixing, recovers I7/I8)")

print("\n-- numeric cross-check (E=2, Delta=1, Gamma=0.1; W=sqrt 3) --")
tt = sp.symbols('tt')
root = sp.nsolve(2 * sp.sinh(tt) - sp.cosh(tt) + sp.I * sp.Rational(1, 10) * sp.sinh(tt) * sp.cosh(tt),
                 tt, sp.Float('0.549306') - sp.Float('0.05') * sp.I)
exact = complex(sp.sinh(root)**2).imag
approx = float(Im_s2.subs({Er: 2, Dr: 1, Wr: sp.sqrt(3), Gam: sp.Rational(1, 10)}))
print("  Im[sinh^2] exact    =", round(exact, 5))
print("  Im[sinh^2] O(Gamma) =", round(approx, 5), " (should match for small Gamma)")


# ================================================================
# NONUNIFORM EDGE: physical-root continuation and distributional sum rule
# Exact quartic after y=exp(-theta):
#   -i Gamma y^4 - 2(E+Delta)y^3 + 2(E-Delta)y + i Gamma = 0.
# Starting at high energy and continuing downward is essential: solving the
# four roots independently near the edge can jump to the advanced branch.
# ================================================================
mp.mp.dps = 28


def qpoly(y, energy, gamma, delta):
    return (-mp.j * gamma * y**4
            - 2 * (energy + delta) * y**3
            + 2 * (energy - delta) * y
            + mp.j * gamma)


def qpoly_y(y, energy, gamma, delta):
    return (-4 * mp.j * gamma * y**3
            - 6 * (energy + delta) * y**2
            + 2 * (energy - delta))


def qpoly_energy(y):
    return 2 * y * (1 - y**2)


def correct_root(y, energy, gamma, delta):
    """Damped Newton correction initialized on the continued root."""
    for _ in range(18):
        residual = qpoly(y, energy, gamma, delta)
        step = residual / qpoly_y(y, energy, gamma, delta)
        if abs(step) < mp.mpf("1e-22") * (1 + abs(y)):
            break

        damping = mp.mpf(1)
        old_norm = abs(residual)
        for _ in range(6):
            trial = y - damping * step
            if abs(qpoly(trial, energy, gamma, delta)) <= old_norm:
                y = trial
                break
            damping /= 2
        else:
            raise RuntimeError("Newton correction left the physical branch")

    if abs(qpoly(y, energy, gamma, delta)) > mp.mpf("1e-15"):
        raise RuntimeError(f"Physical-root residual too large at E={energy}")
    return y


def physical_root_above_gap(energy, gamma, delta):
    y_bcs = mp.sqrt((energy - delta) / (energy + delta))
    return correct_root(y_bcs, energy, gamma, delta)


def two_im_sinh2(y):
    sinh_theta = (1 / y - y) / 2
    return 2 * mp.im(sinh_theta**2)


def edge_integral(gamma, delta, n_edge=1000, n_tail=300):
    """Integrate S/Q=2 Im[sinh(theta)^2] from E=0 to 20 Delta."""
    edge_scale = delta * (gamma / delta)**(mp.mpf(2) / 3)
    edge_lo = max(mp.mpf(0), delta - 30 * edge_scale)
    edge_hi = delta + 60 * edge_scale
    e_max = 20 * delta

    if edge_lo > 0:
        distances = [
            delta * mp.exp(mp.log((delta - edge_lo) / delta) * k / n_tail)
            for k in range(n_tail + 1)
        ]
        energies = [delta - distance for distance in distances]
    else:
        energies = [mp.mpf(0)]

    energies += [
        edge_lo + (edge_hi - edge_lo) * k / n_edge
        for k in range(1, n_edge + 1)
    ]

    upper_start = edge_hi - delta
    upper_stop = e_max - delta
    energies += [
        delta + upper_start * mp.exp(
            mp.log(upper_stop / upper_start) * k / n_tail
        )
        for k in range(1, n_tail + 1)
    ]

    descending = list(reversed(energies))
    previous_energy = descending[0]
    y = physical_root_above_gap(previous_energy, gamma, delta)
    descending_values = []

    for energy in descending:
        if energy != previous_energy:
            predictor = y + (
                -qpoly_energy(y) / qpoly_y(y, previous_energy, gamma, delta)
            ) * (energy - previous_energy)
            y = correct_root(predictor, energy, gamma, delta)
        descending_values.append(two_im_sinh2(y))
        previous_energy = energy

    values = list(reversed(descending_values))
    return mp.fsum(
        (energies[k + 1] - energies[k])
        * (values[k + 1] + values[k]) / 2
        for k in range(len(energies) - 1)
    )


print("\n-- fixed-energy outer and rounded-edge checks --")
start = perf_counter()
delta_mp = mp.mpf(1)
energy_fixed = mp.mpf("1.5")
w_fixed = mp.sqrt(energy_fixed**2 - delta_mp**2)
outer_errors = []

for gamma_mp in map(mp.mpf, ("1e-2", "1e-3", "1e-4")):
    y = physical_root_above_gap(energy_fixed, gamma_mp, delta_mp)
    exact_outer = two_im_sinh2(y)
    outer = -4 * gamma_mp * energy_fixed**2 * delta_mp**2 / w_fixed**5
    error = abs(exact_outer - outer) / abs(outer)
    outer_errors.append(error)
    print(f"  Gamma/Delta={float(gamma_mp):.0e}: "
          f"relative outer error={float(error):.3e}")

assert outer_errors[2] < outer_errors[1] < outer_errors[0]
assert outer_errors[-1] < mp.mpf("1e-5")

gamma_edge = mp.mpf("1e-4") * delta_mp
integral = edge_integral(gamma_edge, delta_mp)
integral_error = abs(integral + mp.pi * delta_mp) / (mp.pi * delta_mp)
print("  integral 2 Im[sinh(theta)^2] dE =", mp.nstr(integral, 12))
print("  relative error from -pi Delta =", f"{float(integral_error):.3e}")
print(f"  elapsed={perf_counter() - start:.2f} s")

assert integral < 0
assert integral_error < mp.mpf("0.01")
print("  [PASS] fixed-energy outer limit and edge sum rule")
print("\nALL PASS")
