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
through the DEPAIRED spectrum (S(E) = 2Q Im sinh^2 theta, O(Q^3), below;
that computation involves only the retarded angle and is gA-independent).
"""
import sympy as sp

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
S = sp.simplify(2 * Qs * Im_s2)
print("S = 2Q Im[sinh^2]  =", S)
hbar, Dco = sp.symbols('hbar D', real=True, positive=True)
print("with Gamma=2 hbar D Q^2:  S =", sp.simplify(S.subs(Gam, 2 * hbar * Dco * Qs**2)), " => O(Q^3)")
print("BCS limit Gamma->0:  S =", S.subs(Gam, 0), "  (no mixing, recovers I7/I8)")

print("\n-- numeric cross-check (E=2, Delta=1, Gamma=0.1; W=sqrt 3) --")
tt = sp.symbols('tt')
root = sp.nsolve(2 * sp.sinh(tt) - sp.cosh(tt) + sp.I * sp.Rational(1, 10) * sp.sinh(tt) * sp.cosh(tt),
                 tt, sp.Float('0.549306') - sp.Float('0.05') * sp.I)
exact = complex(sp.sinh(root)**2).imag
approx = float(Im_s2.subs({Er: 2, Dr: 1, Wr: sp.sqrt(3), Gam: sp.Rational(1, 10)}))
print("  Im[sinh^2] exact    =", round(exact, 5))
print("  Im[sinh^2] O(Gamma) =", round(approx, 5), " (should match for small Gamma)")
