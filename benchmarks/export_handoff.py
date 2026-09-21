"""Create a compact review bundle; raw volumes remain in the evidence directory."""
import hashlib,json,subprocess,tarfile,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
root=repo/(sys.argv[1] if len(sys.argv)>1 else 'results/2026-09-18-pilot')
final=root/'final';final.mkdir(exist_ok=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True)
(final/'commit.txt').write_text(git('rev-parse','HEAD'))
(final/'working_tree_status.txt').write_text(git('status','--porcelain=v1'))
(final/'tracked_changes.patch').write_text(git('diff','--binary'))
paths=[]
for name in git('ls-files','-co','--exclude-standard').splitlines():
 p=Path(name)
 if p.parts[0] in ('porewise','tests','benchmarks','examples','validation','.github') or name in (
  'README.md','AUDIT.md','CODE_AND_BENCHMARK_HANDOFF.md','pyproject.toml','LICENSE','LICENSE.md',
  'docs/numerical_limits.md','docs/reference_scope.md','docs/benchmark_reproduction.md','docs/gpu_convergence_review.md'):
  paths.append(p)
# This small deterministic mask is git-ignored but required by the pressure case.
paths.append(Path('benchmarks/configs/open_channel.npy'))
source_archive=final/'source_snapshot.tar.gz'
with tarfile.open(source_archive,'w:gz') as tar:
 for p in sorted(set(paths)):
  if (repo/p).is_file():tar.add(repo/p,arcname=str(p))
source_hashes={str(p):hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(set(paths)) if (repo/p).is_file()}
(final/'source_sha256.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
large_artifacts={}
for path in sorted(root.rglob('*')):
 if path.is_file() and path.suffix in ('.npy','.npz','.raw','.bin'):
  with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  large_artifacts[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest)
(final/'excluded_binary_artifacts.json').write_text(json.dumps(large_artifacts,indent=2)+'\n')
bundle=final/'technical_evidence.tar.gz'
with tarfile.open(bundle,'w:gz') as tar:
 tar.add(repo/'CODE_AND_BENCHMARK_HANDOFF.md',arcname='CODE_AND_BENCHMARK_HANDOFF.md')
 tar.add(source_archive,arcname='source_snapshot.tar.gz')
 for wheel in sorted((repo/'dist').glob('*.whl')):tar.add(wheel,arcname='wheels/'+wheel.name)
 for path in sorted(root.rglob('*')):
  if not path.is_file() or path in (bundle,source_archive) or path.name=='artifact_checksums.json':continue
  if path.suffix not in ('.json','.csv','.jsonl','.xml','.log','.png','.pdf','.patch','.txt','.gz') and path.name!='solver_before.py':continue
  if 'external' in path.parts and path.name!='sources.json':continue
  tar.add(path,arcname=str(path.relative_to(root)))
artifacts={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_archive,bundle,*sorted((repo/'dist').glob('*.whl'))]}
(final/'artifact_checksums.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(bundle,round(bundle.stat().st_size/1024**2,2),'MiB')
