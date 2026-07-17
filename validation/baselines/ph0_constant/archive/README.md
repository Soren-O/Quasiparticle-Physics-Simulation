# Archived Fischer Fig. 7 regression

`fischer_fig7_paper_loose_newton_20260716.{csv,pdf}` is the superseded
Windows/Python 3.14.3 pin generated with the former Fig. 7 inner-Newton
backward-error limit of `1e-6` and Picard relative limit of `1e-7`.

The pin is retained only as audit evidence. An exact Linux repeat showed that
the loose contract admitted platform-dependent approximations to the same root even though
all states passed its `1e-5` independent balance certificate. Tightening the
inner Newton and outer Picard contracts collapsed the two worst meaningful
Windows/Linux discrepancies, and the exact 48-point tight Windows run moved
the old QP-loss curve by as much as 3.45%. Do not use these archived files as
current validation or paper-parity evidence.

`fischer_fig7_paper_tight_linux_f9014d99_20260717.{csv,pdf}` is the exact
48-point Linux tight-contract predecessor. It was superseded only because the
conservative solve-contract digest covers the complete `qpsim` source tree and
later N31/N33 integration fixes outside the Fig. 7 call path advanced that
digest. A frozen-final-source Windows repeat passed the established
cross-platform gates and produced the active canonical. The Linux pair remains
portable numerical evidence, not an active pin.

SHA-256:

- CSV: `f62c2d05826a65db43bade00bdb4efaf394e6fc95f250c9ddcdd50b4b1d65af7`
- PDF: `cabcabddf00bf5552f929c3d615c2d24c516cb74eb3cc93fb0e3779c4fc47a19`
- tight Linux predecessor CSV: `f197e5044e34b8d98b8d18d6c75c41a448816c5c007e98bf49f803d7772d7312`
- tight Linux predecessor PDF: `4bb18fa5a70bc381421de4f3d7ee4dc7cfef50400883ca148e1515cab85831de`
