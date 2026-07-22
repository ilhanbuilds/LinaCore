import sys

from lina_core.cli import main

raise SystemExit(main(["dataset", "validate", *sys.argv[1:]]))
