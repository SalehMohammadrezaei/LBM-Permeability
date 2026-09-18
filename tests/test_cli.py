import json
import numpy as np
from lbm_permeability.__main__ import main


def test_cli_failure_saved(tmp_path):
    out=tmp_path/'failure'
    code=main(['--demo','--backend','numpy','--dx','1e-6','--steps','1','--output',str(out)])
    assert code==2
    result=json.loads((out/'result.json').read_text())
    assert result['termination_reason']=='max_steps' and result['k_m2'] is None


def test_cli_input_failure(tmp_path):
    out=tmp_path/'failure'
    code=main(['--demo','--backend','numpy','--dx','-1','--output',str(out)])
    assert code==2 and (out/'error.json').exists()


def test_cli_missing_file_saved(tmp_path):
    out=tmp_path/'failure'
    code=main([str(tmp_path/'missing.npy'),'--backend','numpy','--dx','1e-6','--output',str(out)])
    assert code==2
    error=json.loads((out/'error.json').read_text())
    assert 'missing.npy' in error['error']
