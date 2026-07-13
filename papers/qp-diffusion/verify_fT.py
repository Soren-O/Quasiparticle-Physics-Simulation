"""
verify_fT.py -- the transverse (charge-imbalance) kinetic equation, assembled.

Closes the open transverse-sector item left by verify_nonadiabatic.py: which
projection of the K-block matrix residual is the transverse kinetic equation
(the "i/reality bridge" to the manifestly-real distribution-matrix form), and
what the linear-in-Ddot coherent terms are.  All claims below are verified
symbolically for generic Delta, fL, fT and guarded numerically.

Conventions = verify_traces / verify_nonadiabatic:
  Moyal A*B = AB + (i h/2)(dE A dt B - dt A dE B) + (second-order term)
  L0 = E t3 + i Delta t2 ;  gR0 = N1 t3 + i N2 t2 ;  gA0 = -gR0  (E > Delta)
  N1 = E/W, N2 = Delta/W, W = sqrt(E^2-Delta^2) ;  h = fL 1 + fT t3
  gK = gR0 * h - h * gA0  (star products) ;  K-block equation
  hbar DN d_x (g d_x g)^K + i[L0, gK]_star + i Icoll = 0.

CHANNEL GEOMETRY (the projection result).  Under X -> [L0, X] the Nambu basis
splits, above the gap with a real order parameter, into
  kernel:  span{ 1, g0 }            (g0 = N1 t3 + i N2 t2,  g0.g0 = +1)
  image:   span{ t1, m }            (m  = N2 t3 + i N1 t2,  m.m  = -1,
                                     [L0,t1] = 2W m, [L0,m] = 2W t1)
with the trace metric v.w = Tr[(v.tau)(w.tau)]/2 (bilinear, no conjugation),
g0.m = 0, t3 = N1 g0 - N2 m, t2 = i N2 g0 - i N1 m.  The scalar kinetic
equations are the KERNEL projections of the K-block residual; the image
channels carry slaved spectral content (no kinetic equation lives there).

RESULTS (all verified below):

  R1  The O(h) residual of i[L0,gK]_star lies EXACTLY in the kernel:
        i[L0,gK]_star|_O(h) = A 1 + B g0 ,   t1- and m-channels = 0,
        A = -2[dt(N1 fL) + dE(Dd N2 fL)]                      (longitudinal)
        B = -2 N1^2 [dt fT + (Delta Dd/E) dE fT + (Delta Dd/E^2) fT]
      The tau3-trace is 2 N1 B = -4(E/W^3)(E^2 dt fT + E Delta Dd dE fT
      + Delta Dd fT), reproducing the verify_nonadiabatic record: the raw
      tau3-trace is N1-weighted kernel content and nothing else.  This is the
      bridge: the real scalar equations are the kernel coefficients; the i and
      the N1 in the trace are projection weights, not physics.

  R2  The first Wigner correction to the Keldysh ansatz,
      delta gK = gK_star - gK_matrix, is
        fL sector: 0
        fT sector: i hbar (E/W^3)(Delta dt fT + E Dd dE fT) t1
      (pure t1 = image channel; it is what converts the naive N1 time weight
      into the N1^2 advective weight of B after [L0, t1] feeds back).

  R3  Sandwich factorization at O(h):
        [L0, gK]_star = gR0 * [L0,h]_star - [L0,h]_star * gA0
      with no remainder, so the kernel projections coincide with the
      distribution-matrix (Belzig Eq. 30 type) route.

  R4  Equivalent forms of the transverse equation B = 0 (per N1):
        N1 [dt + (Delta Dd/E) dE + Delta Dd/E^2] fT
          = dt(N1 fT) + dE(Dd N2 fT) + (Dd N2/E) fT          (conservative)
          = (1/E) [ dt(N1 u) + dE(Dd N2 u) ] ,  u = E fT     (longitudinal
                                                              operator on E fT)
      Collisionless solutions: fL = G(xi), fT = F(xi)/E for arbitrary G, F,
      xi = sqrt(E^2 - Delta(t)^2): branch-odd occupations are frozen on the
      moving shells exactly like the energy mode, and the extra 1/E is the
      BCS effective-charge dilution q = xi/E.  A spatially uniform real
      Ddot does NOT source fT (no fL anywhere in B): with J_T[fL,0,n]=0,
      the zero-transverse solution is invariant under the gap drive.

  R5  Spatial sector, uniform gap (real Delta(t), fL/fT(E,x,t)):
        kernel channels of the full K-block residual give
        longitudinal: dt(N1 fL) + dE(Dd N2 fL) = DN dxx fL
        transverse:   N1[dt + (Delta Dd/E) dE + Delta Dd/E^2] fT = DN dxx fT
      i.e. in the same (time weight N1, undressed flux) normalization as the
      longitudinal A1 pair, the transverse mode diffuses with the SAME
      undressed Laplacian; the established N1^2 transverse dressing is the
      dressing of the physical charge current, jT = -DN N1^2 dx fT
      = -(DN/4) Tr[t3 (g dx g)^K], which is reproduced.  dt(N1 fT) and
      div jT therefore do not balance by themselves: the difference is
      coherent quasiparticle-condensate charge conversion.

  R6  Spatial sector, static inhomogeneous real gap Delta(x):
        The strict local-BCS propagator has a second-gradient retarded
        residual.  The normalized spectral correction
          eta = hbar DN dx(theta)'/(2W),  delta gR = delta gA = i eta m
        cancels the local-current m-channel residual and leaves the scalar
        kernel below unchanged.  Its relaxation trace is curvature
        bookkeeping already contained in that kernel, not an added term.
        longitudinal channel: 2 DN dxx fL  (no drift, no fT -- A1 structure)
        transverse kernel channel:
          B = 2 DN [ dx(N1 dx fT) + N2 dx(thp fT) ],  thp = E Delta'/W^2
              (= dx theta for theta = artanh(Delta/E): pairing-angle gradient)
        slaved m-channel: beta = -2 DN N1 dx(thp) fT (gap curvature; image
        channel -- determines the slaved t1 correction, not dynamics)
        tau3-trace: 2 DN dx(N1^2 dx fT) = divergence of the dressed current
        (consistency: t3 = N1 g0 - N2 m mixes kernel and slaved content).

Run:  .venv/bin/python verify_fT.py   ->  ALL PASS expected.
"""
import sympy as sp
from sympy.core.function import AppliedUndef

E, t, x, hbar = sp.symbols('E t x hbar', real=True, positive=True)
DN = sp.Symbol('D_N', real=True, positive=True)
I_ = sp.I
I2 = sp.eye(2)
t1 = sp.Matrix([[0, 1], [1, 0]])
t2 = sp.Matrix([[0, -I_], [I_, 0]])
t3 = sp.Matrix([[1, 0], [0, -1]])


def dE(M):
    return M.applyfunc(lambda e: sp.diff(e, E)) if hasattr(M, 'applyfunc') else sp.diff(M, E)


def dt(M):
    return M.applyfunc(lambda e: sp.diff(e, t)) if hasattr(M, 'applyfunc') else sp.diff(M, t)


def dx(M):
    return M.applyfunc(lambda e: sp.diff(e, x)) if hasattr(M, 'applyfunc') else sp.diff(M, x)


def star2(A, B):
    s0 = A * B
    s1 = (I_ * hbar / 2) * (dE(A) * dt(B) - dt(A) * dE(B))
    s2 = sp.Rational(1, 2) * (I_ * hbar / 2)**2 * (
        dE(dE(A)) * dt(dt(B)) - 2 * dE(dt(A)) * dE(dt(B)) + dt(dt(A)) * dE(dE(B)))
    return s0 + s1 + s2


def scomm2(A, B):
    return star2(A, B) - star2(B, A)


def hmat(M, n):
    return sp.Matrix(M).applyfunc(lambda e: sp.expand(e).coeff(hbar, n))


def channels(M, N1, N2):
    """Pauli components and kernel/image projections of a 2x2 matrix."""
    r0 = sp.trace(M) / 2
    r1 = sp.trace(t1 * M) / 2
    r2 = sp.trace(t2 * M) / 2
    r3 = sp.trace(t3 * M) / 2
    B = I_ * N2 * r2 + N1 * r3          # g0 (kernel) projection
    beta = -(I_ * N1 * r2 + N2 * r3)    # m (image, slaved) projection
    return r0, r1, r2, r3, B, beta


def flux_K(gR, gA, h):
    """(g dx g)^K for gK = gR h - h gA with normalized gR/gA (exact identity)."""
    return gR * dx(gR) * h - h * gA * dx(gA) + dx(h) - gR * dx(h) * gA


# ---------------------------------------------------------------- time sector
def build_time(Delta, fL, fT):
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    L0 = E * t3 + I_ * Delta * t2
    gR0 = N1 * t3 + I_ * N2 * t2
    gA0 = -gR0
    h = fL * I2 + fT * t3
    gK = star2(gR0, h) - star2(h, gA0)
    K = I_ * scomm2(L0, gK)

    A_exp = -2 * (sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E))
    adv = sp.diff(fT, t) + (Delta * Dd / E) * sp.diff(fT, E) + (Delta * Dd / E**2) * fT
    B_exp = -2 * N1**2 * adv
    t3_rec = -4 * E / W**3 * (E**2 * sp.diff(fT, t) + E * Delta * Dd * sp.diff(fT, E)
                              + Delta * Dd * fT)

    r0, r1, r2, r3, B, beta = channels(hmat(K, 1), N1, N2)

    out = {}
    out['O(h0) residual = 0 (matrix)'] = sp.simplify(hmat(K, 0).norm())
    out['t1 channel = 0 at O(h)'] = sp.simplify(r1)
    out['m (slaved) channel = 0 at O(h)'] = sp.simplify(beta)
    out['A = r0 = -2[dt(N1 fL)+dE(Dd N2 fL)]'] = sp.simplify(r0 - A_exp)
    out['B (g0 kernel) = -2 N1^2 [adv fT]'] = sp.simplify(B - B_exp)
    out['tau3-trace = 2 N1 B (recorded t3 form)'] = sp.simplify(2 * r3 - t3_rec)

    # R3: sandwich factorization at O(h^0), O(h^1)
    Kh = scomm2(L0, h)
    sand = I_ * (star2(gR0, Kh) - star2(Kh, gA0))
    out['sandwich O(h0)'] = sp.simplify((hmat(K, 0) - hmat(sand, 0)).norm())
    out['sandwich O(h1)'] = sp.simplify((hmat(K, 1) - hmat(sand, 1)).norm())

    # R4: form identities
    cons = sp.diff(N1 * fT, t) + sp.diff(Dd * N2 * fT, E) + (Dd * N2 / E) * fT
    out['advective = conservative form'] = sp.simplify(N1 * adv - cons)
    u = E * fT
    out['E x advective = longitudinal op on E fT'] = \
        sp.simplify(E * N1 * adv - (sp.diff(N1 * u, t) + sp.diff(Dd * N2 * u, E)))

    if isinstance(fL, AppliedUndef) and isinstance(fT, AppliedUndef):
        # R2: first Wigner correction to the Keldysh ansatz (needs symbolic f's)
        dgK = hmat(gK - (gR0 * h - h * gA0), 1)
        dgK_fL = dgK.applyfunc(lambda e: e.subs(fT, 0))
        out['delta gK |fL = 0'] = sp.simplify(dgK_fL.norm())
        dgK_fT = dgK.applyfunc(lambda e: e.subs(fL, 0))
        dgK_exp = I_ * (E / W**3) * (Delta * sp.diff(fT, t) + E * Dd * sp.diff(fT, E)) * t1
        out['delta gK |fT = i(E/W^3)(D dtfT + E Dd dEfT) t1'] = \
            sp.simplify((dgK_fT - dgK_exp).norm())
        # structure: no L-T mixing in the kernel channels
        out['B has no fL'] = sp.simplify(B.diff(fL))
        out['A has no fT'] = sp.simplify(r0.diff(fT))
        out['source-free transverse residual vanishes at fT=0'] = \
            sp.simplify(B.subs(fT, 0).doit())
    return out


# ------------------------------------------------------------- spatial sector
def build_space_uniform(Delta, fL, fT):
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    Dd = sp.diff(Delta, t)
    L0 = E * t3 + I_ * Delta * t2
    gR0 = N1 * t3 + I_ * N2 * t2
    gA0 = -gR0
    h = fL * I2 + fT * t3
    gK = star2(gR0, h) - star2(h, gA0)
    M = hbar * DN * dx(flux_K(gR0, gA0, h)) + I_ * scomm2(L0, gK)

    r0, r1, r2, r3, B, beta = channels(hmat(M, 1), N1, N2)
    adv = sp.diff(fT, t) + (Delta * Dd / E) * sp.diff(fT, E) + (Delta * Dd / E**2) * fT

    out = {}
    out['O(h0) residual = 0 (matrix)'] = sp.simplify(hmat(M, 0).norm())
    out['t1 channel = 0'] = sp.simplify(r1)
    out['m (slaved) channel = 0 (uniform gap)'] = sp.simplify(beta)
    out['longitudinal: r0 = -2[dt(N1 fL)+dE(Dd N2 fL)] + 2 DN dxx fL'] = sp.simplify(
        r0 - (-2 * (sp.diff(N1 * fL, t) + sp.diff(Dd * N2 * fL, E))
              + 2 * DN * sp.diff(fL, x, 2)))
    out['transverse: B = -2 N1^2 [adv fT] + 2 DN N1 dxx fT'] = sp.simplify(
        B - (-2 * N1**2 * adv + 2 * DN * N1 * sp.diff(fT, x, 2)))
    out['uniform-space transverse residual vanishes at fT=0'] = \
        sp.simplify(B.subs(fT, 0).doit())
    jT = -(DN / 4) * sp.trace(t3 * flux_K(gR0, gA0, h))
    out['charge current jT = -DN N1^2 dx fT'] = sp.simplify(
        jT - (-DN * N1**2 * sp.diff(fT, x)))
    return out


def build_space_inhom(Delta, fL, fT):
    """Static, real Delta(x); fL/fT(E,x), including spectral completion."""
    W = sp.sqrt(E**2 - Delta**2)
    N1, N2 = E / W, Delta / W
    L0 = E * t3 + I_ * Delta * t2
    gR0 = N1 * t3 + I_ * N2 * t2
    gA0 = -gR0
    h = fL * I2 + fT * t3
    gK = gR0 * h - h * gA0
    out = {}
    out['static commutator i[L0,gK] = 0'] = sp.simplify(
        (I_ * (L0 * gK - gK * L0)).norm())

    M = hbar * DN * dx(flux_K(gR0, gA0, h))
    r0, r1, r2, r3, B, beta = channels(hmat(M, 1), N1, N2)
    thp = E * sp.diff(Delta, x) / W**2          # = dx theta, theta = artanh(Delta/E)

    out['t1 channel = 0'] = sp.simplify(r1)
    out['longitudinal: r0 = 2 DN dxx fL (no drift, no fT)'] = sp.simplify(
        r0 - 2 * DN * sp.diff(fL, x, 2))
    out['transverse kernel: B = 2 DN [dx(N1 dxfT) + N2 dx(thp fT)]'] = sp.simplify(
        B - 2 * DN * (sp.diff(N1 * sp.diff(fT, x), x) + N2 * sp.diff(thp * fT, x)))
    out['inhomogeneous transverse residual vanishes at fT=0'] = \
        sp.simplify(B.subs(fT, 0).doit())
    out['slaved m channel: beta = -2 DN N1 dx(thp) fT'] = sp.simplify(
        beta - (-2 * DN * N1 * sp.diff(thp, x) * fT))
    out['tau3-trace = 2 DN dx(N1^2 dxfT) (current divergence)'] = sp.simplify(
        2 * r3 - 4 * DN * sp.diff(N1**2 * sp.diff(fT, x), x))

    # Same-order spectral completion of the local-BCS ansatz.  The correction
    # is O(hbar * second spatial gradient); its contribution to the spatial
    # current divergence starts beyond the order retained here.
    m0 = N2 * t3 + I_ * N1 * t2
    eta = hbar * DN * sp.diff(thp, x) / (2 * W)
    delta_gR = I_ * eta * m0
    delta_gA = delta_gR
    delta_gK = delta_gR * h - h * delta_gA
    delta_gK_exp = -2 * I_ * eta * N1 * fT * t1

    out['spectral correction preserves normalization'] = sp.simplify(
        (gR0 * delta_gR + delta_gR * gR0).norm())
    out['delta gK = -2 i eta N1 fT tau1'] = sp.simplify(
        (delta_gK - delta_gK_exp).norm())

    retarded_residual = (
        hbar * DN * dx(gR0 * dx(gR0))
        + I_ * (L0 * delta_gR - delta_gR * L0)
    )
    out['retarded local-BCS curvature residual is cancelled'] = sp.simplify(
        hmat(retarded_residual, 1).norm())

    correction_K = I_ * (L0 * delta_gK - delta_gK * L0)
    rc0, rc1, rc2, rc3, Bc, betac = channels(hmat(correction_K, 1), N1, N2)
    out['spectral completion has zero longitudinal kernel projection'] = \
        sp.simplify(rc0)
    out['spectral completion has zero transverse kernel projection'] = \
        sp.simplify(Bc)
    out['local-current and spectral image residuals cancel'] = sp.simplify(
        beta + betac)

    hatD = -I_ * Delta * t2
    Rgrad = I_ * sp.trace(hatD * (delta_gR + delta_gA)) / 4
    out['gradient relaxation trace R_grad = -Delta N1 eta'] = sp.simplify(
        Rgrad + Delta * N1 * eta)
    out['tau3 correction trace = -2 R_grad fT'] = sp.simplify(
        sp.trace(t3 * correction_K) / 4 + 2 * Rgrad * fT)

    dDL = -sp.trace(delta_gR * gA0 + gR0 * delta_gA) / 4
    dDT = -sp.trace(
        delta_gR * t3 * gA0 * t3 + gR0 * t3 * delta_gA * t3
    ) / 4
    out['no linear-eta correction to D_L'] = sp.simplify(dDL)
    out['no linear-eta correction to D_T'] = sp.simplify(dDT)
    return out


def check(out, label, subs=None):
    print(f"---- {label} ----")
    ok = True
    for name, expr in out.items():
        if subs:
            dval = sp.N(sp.sympify(expr).subs(subs))
            val = abs(complex(dval))
            good = val < 1e-9
            print(f"  [{'PASS' if good else 'FAIL'}] max|.|={val:.1e}  {name}")
        else:
            good = sp.simplify(expr) == 0
            print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good
    return ok


if __name__ == "__main__":
    Dgen = sp.Function('Delta', positive=True)(t)
    fLg = sp.Function('f_L')(E, t)
    fTg = sp.Function('f_T')(E, t)

    a = check(build_time(Dgen, fLg, fTg), "(a) time sector, symbolic generic")
    print()

    Dnum = 1 + sp.Rational(3, 10) * t + sp.Rational(1, 10) * t**2
    b = check(build_time(Dnum, sp.exp(-E) * sp.cos(t), sp.sin(E) * sp.exp(-t)),
              "(b) time sector, numeric guard", {E: 2, t: sp.Rational(1, 2), hbar: 1})
    print()

    # (c) collisionless frozen-shell solutions on the moving spectral shells
    print("---- (c) collisionless solutions on moving shells ----")
    Wn = sp.sqrt(E**2 - Dgen**2)
    N1n, N2n = E / Wn, Dgen / Wn
    Ddn = sp.diff(Dgen, t)
    G, F = sp.Function('G'), sp.Function('F')
    A_val = -2 * (sp.diff(N1n * G(Wn), t) + sp.diff(Ddn * N2n * G(Wn), E))
    fT_sol = F(Wn) / E
    B_val = sp.diff(fT_sol, t) + (Dgen * Ddn / E) * sp.diff(fT_sol, E) \
        + (Dgen * Ddn / E**2) * fT_sol
    okA = sp.simplify(A_val) == 0
    okB = sp.simplify(B_val) == 0
    print(f"  [{'PASS' if okA else 'FAIL'}] fL = G(xi) solves the longitudinal equation")
    print(f"  [{'PASS' if okB else 'FAIL'}] fT = F(xi)/E solves the transverse equation")
    c = okA and okB
    print()

    fLs = sp.Function('f_L')(E, x, t)
    fTs = sp.Function('f_T')(E, x, t)
    d = check(build_space_uniform(Dgen, fLs, fTs), "(d) spatial sector, uniform gap")
    print()

    Dx = sp.Function('Delta', positive=True)(x)
    e = check(build_space_inhom(Dx, sp.Function('f_L')(E, x), sp.Function('f_T')(E, x)),
              "(e) spatial sector, static inhomogeneous gap")
    print()

    all_ok = a and b and c and d and e
    print("ALL PASS" if all_ok else "SOME FAILED -- inspect above")
    raise SystemExit(0 if all_ok else 1)
