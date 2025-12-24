import sys
from .rollback import set_current_version

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.pneumonia_system.mlops.promote v2")
        raise SystemExit(1)

    new_version = sys.argv[1].strip()
    set_current_version(new_version)
    print("Promoted:", new_version)
