"""Optional host tooling must not prevent solving or exporting results."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from porewise import io, memory
from porewise.backends import HAS_GPU


@pytest.mark.parametrize('error', [FileNotFoundError(), PermissionError(),
                                 subprocess.TimeoutExpired('git', 5)])
def test_git_execution_unavailable(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(io.subprocess, 'run', fail)
    result = io.provenance()
    assert result['commit'] is None and result['working_tree_status'] is None
    assert result['git_provenance_status']['status'] == 'unavailable'
    assert type(error).__name__ in result['git_provenance_status']['reason']
    assert result['source_sha256'] and result['packages']['numpy']


@pytest.mark.parametrize('failed_call', [1, 2])
def test_git_nonzero_exit_has_no_partial_metadata(monkeypatch, failed_call):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=128 if len(calls) == failed_call else 0,
                               stdout='a-commit', stderr='not a git repository')
    monkeypatch.setattr(io.subprocess, 'run', run)
    result = io.provenance()
    assert result['commit'] is None and result['working_tree_status'] is None
    assert '128' in result['git_provenance_status']['reason']


def test_git_available_preserves_clean_worktree(monkeypatch):
    def run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='abc123\n' if command[1] == 'rev-parse' else '')
    monkeypatch.setattr(io.subprocess, 'run', run)
    result = io.provenance()
    assert result['commit'] == 'abc123' and result['working_tree_status'] == ''
    assert result['git_provenance_status'] == dict(status='available', reason=None)


@pytest.mark.parametrize('platform, expected', [('linux', 2048 * 1024), ('darwin', 2048)])
def test_rss_units(monkeypatch, platform, expected):
    monkeypatch.setattr(memory.sys, 'platform', platform)
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=2048)))
    report = memory.process_memory_report()
    assert report['process_peak_rss_bytes'] == expected
    assert report['process_peak_rss_status'] == 'available'
    assert 'not a per-case peak' in report['process_peak_rss_scope']


@pytest.mark.parametrize('failure', ['import', 'system-call', 'invalid-value', 'unsupported-platform'])
def test_rss_unavailable(monkeypatch, failure):
    monkeypatch.setattr(memory.sys, 'platform', 'win32' if failure == 'unsupported-platform' else 'linux')
    def getrusage(who):
        if failure == 'system-call':
            raise OSError('measurement unavailable')
        return SimpleNamespace(ru_maxrss=float('nan'))
    monkeypatch.setitem(sys.modules, 'resource', None if failure == 'import' else
                        SimpleNamespace(RUSAGE_SELF=0, getrusage=getrusage))
    report = memory.process_memory_report()
    assert report['process_peak_rss_bytes'] is None
    assert report['process_peak_rss_status'] == 'unavailable'
    assert report['process_peak_rss_note']
    assert json.loads(json.dumps(report))['process_peak_rss_bytes'] is None


def test_fresh_process_without_git_or_resource(tmp_path):
    # No Git executable can be found, and importing resource always fails.
    # This also exercises package import (including the pressure module), a
    # converged CPU solve, normal CLI result export and its exception handler.
    script = r'''
import builtins, json, sys
from pathlib import Path
original_import = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError('resource deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_resource
import numpy as np
from porewise.__main__ import main
root = Path(sys.argv[1])
mask = np.ones((8, 12), dtype=bool)
mask[2:6] = False
np.save(root/'mask.npy', mask)
common = ['--backend', 'numpy', '--dx', '1e-6']
assert main([str(root/'mask.npy'), *common, '--steps', '3000', '--check-every', '25',
             '--fields', '--output', str(root/'accepted')]) == 0
result = json.loads((root/'accepted/result.json').read_text())
assert result['valid_for_permeability'] and result['k_m2'] > 0
assert result['memory']['process_peak_rss_bytes'] is None
assert result['memory']['process_peak_rss_status'] == 'unavailable'
assert result['memory']['process_peak_rss_note']
assert (root/'accepted/convergence.csv').is_file()
with np.load(root/'accepted/fields.npz', allow_pickle=False) as fields:
    assert fields['ux'].shape == mask.shape and np.isfinite(fields['ux']).all()
assert main([str(root/'missing.npy'), *common, '--output', str(root/'error')]) == 2
error = json.loads((root/'error/error.json').read_text())
assert 'missing.npy' in error['error']
for p in (result['provenance'], result['metadata']['geometry_provenance'], error['provenance']):
    assert p['commit'] is None and p['working_tree_status'] is None
    assert p['git_provenance_status']['status'] == 'unavailable'
    assert 'FileNotFoundError' in p['git_provenance_status']['reason']
'''
    env = os.environ.copy()
    env['PATH'] = str(tmp_path)
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not HAS_GPU, reason='requires real CUDA')
def test_pressure_export_without_memory_measurement(monkeypatch, tmp_path):
    from porewise.d2q9_pressure import lbm_stokes_2d_pressure
    monkeypatch.setitem(sys.modules, 'resource', None)
    result = lbm_stokes_2d_pressure(np.zeros((8, 12), bool), walls_y=True,
                                    n_steps_max=1, verbose=False, return_fields=False)
    io.save_result(tmp_path, result)
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['termination_reason'] == 'max_steps'
    assert saved['memory']['process_peak_rss_bytes'] is None
    assert saved['memory']['process_peak_rss_status'] == 'unavailable'
    assert saved['memory']['gpu_pool_reserved_peak_sampled_bytes'] > 0


def test_former_package_name_still_imports():
    import importlib, sys, warnings
    sys.modules.pop('lbm_permeability', None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        old = importlib.import_module('lbm_permeability')
    import porewise
    assert old.lbm_stokes is porewise.lbm_stokes
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
