"""Local web frontend for qpsim (optional extra ``qpsim[ui]``).

A small FastAPI application that drives the shipped engine surfaces —
0-D T3 steady state and transients, the 1D spatial strip, and the M25
junction moment layer — from a browser, with named-setup persistence,
background runs with progress/cancel, and server-rendered matplotlib
plots.

Nothing in the core library imports this package; the UI dependencies
(``fastapi``, ``uvicorn``, ``matplotlib``) install via the ``ui``
optional extra and are only imported when the frontend itself runs.
Launch with ``qpsim-ui`` (console script) or ``python -m qpsim.webui``.
"""
