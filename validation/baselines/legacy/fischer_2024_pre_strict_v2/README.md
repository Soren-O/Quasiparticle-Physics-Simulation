# Fischer 2024 pre-strict-v2 artifacts

These files preserve the Fischer 2024 artifacts that occupied the canonical
paths before strict-v2 certification. They are retained as audit evidence,
not as accepted numerical regressions.

The CSV files predate the qpsim artifact schema, dependency fingerprint,
independent steady-state certificates, and certified-payload hash. In
addition, the old paper-topology Fig. 8 table contains unchanged low-
temperature thermal seeds at its two weakest drives. Current readers must
reject every file in this directory as legacy.

The canonical files in `../../ph0_constant/` were regenerated from live
production-grid solves after commensurate-grid refinement checks. They are
certified qpsim-native regressions at paper topology; they are not claims of
paper parity.
