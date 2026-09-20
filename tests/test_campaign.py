import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('campaign',Path(__file__).resolve().parents[1]/'benchmarks/campaign.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_reuse_identity_tracks_geometry_source_and_settings():
    original=dict(crop=[[1,5]]*3,dataset_sha256='abc',source_sha256={'solver.py':'123'},
                  settings={'conv_tol':1e-6},output='/a',gpu_pool_limit_bytes=1)
    fingerprint=m.case_fingerprint(original)
    assert m.case_fingerprint(dict(original,output='/b',gpu_pool_limit_bytes=2))==fingerprint
    for key,value in [('crop',[[2,6]]*3),('dataset_sha256','changed'),
                      ('source_sha256',{'solver.py':'changed'}),('settings',{'conv_tol':1e-8})]:
        assert m.case_fingerprint(dict(original,**{key:value}))!=fingerprint


def test_exit_status_reports_failed_tasks_and_reports():
    exit_status = m.exit_status
    assert exit_status(False, [], 0) == 0
    assert exit_status(True, [], 0) == 2
    assert exit_status(False, [dict(case='baseline_x', execution='timeout', returncode=-15)], 0) == 2
    assert exit_status(False, [], 1) == 2
