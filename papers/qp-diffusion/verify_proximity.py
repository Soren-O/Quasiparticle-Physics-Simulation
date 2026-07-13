"""
verify_proximity.py  --  beyond-assumption frontier: NON-BCS / proximity spectra.

Generalizes the real-gap A1 spatial identities (verify_traces.py I5/I7/I8) to an
ARBITRARY COMPLEX spectral angle theta = a + i b  (a=Re theta, b=Im theta:
proximity minigap, depairing, sub-gap states).

Conventions match the paper:
  Pauli t1,t2,t3 ;  gR = cosh(theta) t3 + i sinh(theta) t2
  DOS  N1 = Re cosh(theta) = cosh(a) cos(b)
  distribution h = fL 1 + fT t3 ,  gK = gR h - h gA
  energy/longitudinal current = (1/4)Tr[JK] ;  charge/transverse = (1/4)Tr[t3 JK]
  spatial Keldysh current  JK = gR dx(gK) + gK dx(gA)

The advanced function is the crux of the off-BCS extension:
  PHYSICAL    gA = - t3 gR^dagger t3     (Keldysh symmetry; uses theta*, the correct
                                          continuation for complex theta; for real
                                          theta this is the corrected real-gap
                                          gA = -N1 t3 - i N2 t2)
  PAPER       gA = - (gR)^dagger         (SUPERSEDED 2026-06-09; for real theta it
                                          coincides with the naive algebraic
                                          -t3 gR t3 = -N1 t3 + i N2 t2, which is how
                                          the wrong choice slipped through)
  ALGEBRAIC   gA = - t3 gR t3            (naive shortcut; uses theta, not theta*)
PAPER and ALGEBRAIC coincide only when theta is real.  All cosh/sinh of complex
argument are built from explicit real+imaginary parts so sympy never needs
conjugate() (avoids dummies).

2026-06-09 errata: Part C now uses the PHYSICAL advanced (its header said
"physical" but the code used the paper continuation).  Corrected channel
weights (verify_gA_convention.py T4/T5): ENERGY D_L = cos^2(Im theta),
CHARGE D_T = cosh^2(Re theta).  Part B is kept verbatim as the historical
record that detected the swap.
"""
import sympy as sp

I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])


def tr(M):
    return sp.trace(M)


def sx(e):
    return sp.simplify(sp.expand(e))


# ============================================================
# PART A : spectral coefficients, complex theta = a + i b
# cosh(a+ib)=cosh a cos b + i sinh a sin b ; sinh(a+ib)=sinh a cos b + i cosh a sin b
# ============================================================
a, b = sp.symbols('a b', real=True)
ch = sp.cosh(a) * sp.cos(b) + I_ * sp.sinh(a) * sp.sin(b)        # cosh(theta)
sh = sp.sinh(a) * sp.cos(b) + I_ * sp.cosh(a) * sp.sin(b)        # sinh(theta)
chs = sp.cosh(a) * sp.cos(b) - I_ * sp.sinh(a) * sp.sin(b)       # cosh(theta*)
shs = sp.sinh(a) * sp.cos(b) - I_ * sp.cosh(a) * sp.sin(b)       # sinh(theta*)
gR = ch * t3 + I_ * sh * t2
gA_paper = -chs * t3 + I_ * shs * t2                           # = -(gR)^dagger : SUPERSEDED paper advanced
gA_t3sand = -chs * t3 - I_ * shs * t2                          # -t3 (gR)^dag t3 : PHYSICAL / corrected
gA_alg = -t3 * gR * t3                                         # -t3 gR t3 (naive algebraic; uses theta not theta*)
N1 = sp.cosh(a) * sp.cos(b)                                    # DOS = Re cosh theta


def TrA(gA):                                  # paper's "D_L" trace
    return sx(sp.Rational(1, 4) * tr(I2 - gR * gA))


def TrB(gA):                                  # paper's "D_T" trace
    return sx(sp.Rational(1, 4) * tr(I2 - t3 * gR * t3 * gA))


print("=" * 70)
print("PART A   spectral coefficients,  theta = a + i b")
print("=" * 70)
print(f"  DOS  N1 = {N1}")
print(f"  Tr[1-gR gA]/4      : PAPER -(gR)^dag = {TrA(gA_paper)} | algebraic -t3gRt3 = {TrA(gA_alg)} | t3-sandwich = {TrA(gA_t3sand)}")
print(f"  Tr[1-t3 gR t3 gA]/4: PAPER -(gR)^dag = {TrB(gA_paper)} | algebraic -t3gRt3 = {TrB(gA_alg)} | t3-sandwich = {TrB(gA_t3sand)}")
D_dress = sx(sp.cosh(a)**2)        # the coefficient that -> N1^2 at b=0
D_undress = sx(sp.cos(b)**2)       # the coefficient that -> 1    at b=0
print(f"\n  dressed coeff   cosh^2(Re th) = {D_dress}")
print(f"  undressed coeff cos^2(Im th)  = {D_undress}")
ident = sx(D_dress * D_undress - N1**2)
print(f"  [{'PASS' if ident == 0 else 'FAIL'}]  IDENTITY  cosh^2(a) * cos^2(b) - N1^2 = {ident}")
# which trace gives which, physically (corrected 2026-06-09):
print("\n  PHYSICAL gA=-t3(gR)^dag t3   -> Tr[1-gR gA]/4 = cos^2(Im th) [ENERGY: D_L -> 1 at b=0, 0 sub-gap],"
      " Tr[1-t3 gR t3 gA]/4 = cosh^2(Re th) [CHARGE: D_T -> N1^2 at b=0]")
print("  paper's -(gR)^dag SUPERSEDED -> swaps the two channel weights (the 2026-06 errata)")
print("  naive algebraic -t3 gR t3    -> cosh^2(theta): COMPLEX off-BCS (WRONG continuation, uses theta not theta*)")


# ============================================================
# PART B : I7-REDUX -- real spatially-varying gap, BOTH advanced conventions
# theta real => a=theta(x), b=0.  N1=E/W, N2=Delta(x)/W, W=sqrt(E^2-Delta^2).
# ============================================================
print("\n" + "=" * 70)
print("PART B   I7-redux: real gap Delta(x), longitudinal & transverse currents")
print("=" * 70)
x, E = sp.symbols('x E', real=True, positive=True)
Dl = sp.Function('Delta', positive=True)(x)
fLx = sp.Function('f_L', real=True)(x)
fTx = sp.Function('f_T', real=True)(x)
W = sp.sqrt(E**2 - Dl**2)
n1, n2 = E / W, Dl / W
gRr = n1 * t3 + I_ * n2 * t2
gA_alg_r = -n1 * t3 + I_ * n2 * t2          # algebraic (= superseded paper convention)
gA_phys_r = -n1 * t3 - I_ * n2 * t2         # physical  (-t3 gR^dag t3, real n1,n2; corrected)
hr = fLx * I2 + fTx * t3


def dx(M):
    return M.applyfunc(lambda e: sp.diff(e, x))


def spatial_currents(gA):
    gKr = gRr * hr - hr * gA
    JK = gRr * dx(gKr) + gKr * dx(gA)
    jL = sx(tr(JK) / 4)
    jT = sx(tr(t3 * JK) / 4)
    return jL, jT


dfL, dfT = sp.diff(fLx, x), sp.diff(fTx, x)
for label, gA in [("ALGEBRAIC (paper)", gA_alg_r), ("PHYSICAL", gA_phys_r)]:
    jL, jT = spatial_currents(gA)
    print(f"\n  -- {label} advanced --")
    print(f"     jL=(1/4)Tr[JK]   : d_x f_L coeff = {sx(jL.coeff(dfL))} ,"
          f" d_x f_T coeff = {sx(jL.coeff(dfT))} , bare = {sx(jL - jL.coeff(dfL)*dfL - jL.coeff(dfT)*dfT)}")
    print(f"     jT=(1/4)Tr[t3 JK]: d_x f_T coeff = {sx(jT.coeff(dfT))} ,"
          f" d_x f_L coeff = {sx(jT.coeff(dfL))} , bare = {sx(jT - jT.coeff(dfL)*dfL - jT.coeff(dfT)*dfT)}")
print(f"\n  (N1^2 = {sx(n1**2)} ; the dressing attaches to f_L under algebraic,"
      " to f_T under physical)")


# ============================================================
# PART C : full complex theta(x), physical advanced -- decoupling & dressing
# ============================================================
print("\n" + "=" * 70)
print("PART C   complex theta(x)=a(x)+i b(x), physical advanced, real order param")
print("=" * 70)
ax = sp.Function('a', real=True)(x)
bx = sp.Function('b', real=True)(x)
chx = sp.cosh(ax) * sp.cos(bx) + I_ * sp.sinh(ax) * sp.sin(bx)
shx = sp.sinh(ax) * sp.cos(bx) + I_ * sp.cosh(ax) * sp.sin(bx)
chxs = sp.cosh(ax) * sp.cos(bx) - I_ * sp.sinh(ax) * sp.sin(bx)
shxs = sp.sinh(ax) * sp.cos(bx) - I_ * sp.cosh(ax) * sp.sin(bx)
gRc = chx * t3 + I_ * shx * t2
gAc = -chxs * t3 - I_ * shxs * t2          # PHYSICAL advanced -t3 (gR)^dag t3 (theta*); errata 2026-06-09
hc = fLx * I2 + fTx * t3
gKc = gRc * hc - hc * gAc
JKc = gRc * dx(gKc) + gKc * dx(gAc)
jLc = sx(tr(JKc) / 4)
jTc = sx(tr(t3 * JKc) / 4)


def show(j, name):
    cL = sx(j.coeff(dfL))
    cT = sx(j.coeff(dfT))
    rem = sp.expand(j - cL * dfL - cT * dfT)
    cfL = sx(rem.coeff(fLx))
    cfT = sx(rem.coeff(fTx))
    src = sx(rem - cfL * fLx - cfT * fTx)
    print(f"  {name}:  d_xf_L={cL}  d_xf_T={cT}")
    print(f"      f_L={cfL}  f_T={cfT}  source={src}")
    return cL, cT, cfL, cfT, src


cLc, cTc, cfLc, cfTc, srcc = show(jLc, "jL=(1/4)Tr[JK]")
print()
cLt, cTt, cfLt, cfTt, srct = show(jTc, "jT=(1/4)Tr[t3 JK]")
rC = {
    'jL d_xf_L coeff = cos^2(b(x))   (ENERGY: D_L = cos^2 Im theta)': sx(cLc - sp.cos(bx)**2),
    'jL d_xf_T coeff = 0             (no L-T mixing, real order param)': cTc,
    'jT d_xf_T coeff = cosh^2(a(x))  (CHARGE: D_T = cosh^2 Re theta)': sx(cTt - sp.cosh(ax)**2),
    'jT d_xf_L coeff = 0             (no L-T mixing, real order param)': cLt,
}
print()
for name, res in rC.items():
    print(f"  [{'PASS' if res == 0 else 'FAIL'}] {name}")
partC_ok = all(res == 0 for res in rC.values())


# ============================================================
# numeric guards
# ============================================================
print("\n" + "=" * 70)
print("numeric guards (identity + reality)")
print("=" * 70)
ok = True
for (av, bv) in [(sp.Rational(2, 5), sp.Rational(3, 10)),
                 (sp.Rational(1, 1), sp.Rational(-7, 10)),
                 (sp.Rational(3, 2), sp.Rational(0, 1))]:
    dl = float(D_dress.subs({a: av, b: bv}))
    dt = float(D_undress.subs({a: av, b: bv}))
    n1v = float(N1.subs({a: av, b: bv}))
    res = abs(dl * dt - n1v**2)
    ok &= res < 1e-12
    print(f"  a={float(av):+.2f} b={float(bv):+.2f}: D_dress={dl:.4f} D_undress={dt:.4f}"
          f" N1^2={n1v**2:.4f}  |prod-N1^2|={res:.1e}")
all_ok = ident == 0 and ok and partC_ok
print("\nALL PASS" if all_ok else "SOME FAILED -- inspect above")
raise SystemExit(0 if all_ok else 1)
