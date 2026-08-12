"""`orch` entrypoint. R1's is `orch-workers`; this one is R2's."""

import sys

from .graders.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
