"""Compatibility CLI; see python -m lbm_permeability --help."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lbm_permeability.__main__ import main
if __name__ == '__main__':
    sys.exit(main(['--dimension','3',*sys.argv[1:]]))
