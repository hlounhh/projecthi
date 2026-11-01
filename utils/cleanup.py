"""Release every resource even if another resource fails during shutdown."""

import sys


def cleanup_resources(actions):
    failed = False
    for name, action in actions:
        try:
            if action() is False:
                failed = True
                print(f"Shutdown incomplete: {name}.", file=sys.stderr)
        except Exception as exc:
            failed = True
            print(f"Shutdown failed: {name} ({type(exc).__name__}).", file=sys.stderr)
    return not failed
