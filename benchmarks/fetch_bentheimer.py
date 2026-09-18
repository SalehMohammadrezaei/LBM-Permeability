"""Retrieve the exact public DRP-29 entries; no credentials or archive-wide download."""
import argparse,hashlib,json,time,sys,urllib.request,urllib.parse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lbm_permeability.io import write_json
BASE='https://web.corral.tacc.utexas.edu/digitalporousmedia'
IDS={'ct':'38f6b4fe-4a55-4188-ac28-06cf03c0d509','segmentation':'f4d09c8b-9067-4378-868f-f1f6c09394c3',
     'pressure':'b23e5db4-a8bb-4a59-aaa1-8e91bbf21d53','velocity':'4ecb6b63-db08-4e09-8d9c-f21c702e4b9f'}

def fetch(url,path):
    if path.exists(): return
    tmp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as f:
        while chunk:=response.read(4*1024*1024):f.write(chunk)
    tmp.rename(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/2026-09-18-pilot/dataset');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    metadata=root/'DRP-29_metadata.json';fetch(BASE+'/archive/DRP-29/DRP-29_metadata.json',metadata)
    graph=json.loads(metadata.read_text());files=[]
    manifest=dict(source_record='https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-29',
                  doi='10.17612/P7BC78',authors=['Rodolfo Victor','Masa Prodanovic'],institution='Petrobras',
                  license='ODC-BY 1.0 (public project metadata)',files=files,decision='application example only',
                  shape=[500,500,500],voxel_size_m=5e-6,
                  array_axes=['z','y','x'],storage_order='C: width=x fastest, height=y, image stack=z; physical handedness unverified',
                  crop_policy={'128':[[0,128]]*3,'256':[[0,256]]*3,'morphology':[[0,64]]*3},
                  missing_reference_requirements=['forcing/pressure drop','viscosity/density','velocity/pressure units',
                    'all boundary conditions and reservoirs','reference convergence','physical orientation and handedness'],
                  correspondence_note='same sample and origin labels and enclosing CT path; digitalDataset UUID in migrated analysis metadata differs from enclosing CT node, retained for investigation')
    for category in ('segmentation','ct','velocity','pressure'):
        node=next(n for n in graph['nodes'] if n['id'].endswith(IDS[category]))
        selected=[f for f in node['value']['fileObjs'] if f['name'].endswith('.raw') and f['value'].get('isAdvancedImageFile')]
        for record in selected:
            url=BASE+'/DRP-29/'+urllib.parse.quote(record['path']);path=root/record['name']
            print('fetch',category,url,flush=True);t=time.perf_counter();fetch(url,path)
            sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            files.append(dict(category=category,entry_id=IDS[category],url=url,local_file=path.name,
                              sha256=sha,bytes=path.stat().st_size,download_s=time.perf_counter()-t,
                              retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),record=record,
                              entry_description=node['value'].get('description'),digitalDataset=node['value'].get('digitalDataset')))
            write_json(root/'dataset_manifest.json',manifest)
            print('saved',path.name,path.stat().st_size,sha,flush=True)
    write_json(root/'dataset_manifest.json',manifest)

if __name__=='__main__':main()
