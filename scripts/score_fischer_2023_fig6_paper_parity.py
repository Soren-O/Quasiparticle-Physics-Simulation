"""Build or verify the downstream Fischer-2023 Fig. 6 paper-parity score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from validation.fischer_2023.fig6_paper_parity import (  # noqa: E402
    DEFAULT_SCORE,
    DEFAULT_SPEC,
    score_bytes,
    verify_checked_score,
)
from validation.paper_parity import PaperParityError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical score JSON exclusively; default prints to stdout",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the checked score instead of building one",
    )
    args = parser.parse_args(argv)
    if args.verify:
        if args.output is not None:
            parser.error("--output and --verify are mutually exclusive")
        score = verify_checked_score(DEFAULT_SCORE, args.spec)
        result = score["result"]
        if not isinstance(result, dict):
            raise PaperParityError("Fig. 6 score result is malformed.")
        print(
            "Fig. 6 paper-parity score is current: "
            f"status={result['status']}, gate_eligible={result['gate_eligible']}"
        )
        return 0

    payload = score_bytes(args.spec)
    if args.output is None:
        sys.stdout.buffer.write(payload)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise PaperParityError(f"Refusing to overwrite existing score {args.output}.") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
