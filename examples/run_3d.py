"""Compatibility CLI; see python -m porewise --help."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from porewise.__main__ import main
if __name__ == '__main__':
    sys.exit(main(['--dimension','3',*sys.argv[1:]]))
