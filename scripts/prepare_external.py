"""Prepare external-source inputs before national calibration, without a cyclic dependency."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mrds-only',action='store_true');a=p.parse_args()
 from s17_mrds_validation import load_mrds
 m=load_mrds()
 cols=['dep_id','site_name','latitude','longitude','state','commod1','dev_stat','prod_size','commod']
 (ROOT/'data_derived').mkdir(exist_ok=True)
 m[cols].to_parquet(ROOT/'data_derived/mrds_conus.parquet')
 print('Prepared MRDS rows:',len(m))
 if a.mrds_only:return
 import fitz
 folder=ROOT/'data_external/epa_smelting'
 target=folder/'406370_attached.xlsx'
 if not target.exists():
  with fitz.open(folder/'406370.pdf') as doc:
   names=doc.embfile_names()
   matches=[n for n in names if n.lower().endswith('.xlsx')]
   if len(matches)!=1:raise RuntimeError(f'Expected one embedded workbook; found {matches}')
   target.write_bytes(doc.embfile_get(matches[0]))
 from s23_external_sources import read_smelting,read_coc,fetch_frs,FRS_JSON
 FRS_JSON.parent.mkdir(parents=True,exist_ok=True)
 if not FRS_JSON.exists():
  sm,coc=read_smelting(),read_coc()
  ids=set(sm.loc[sm.Latitude.isna()|sm.Longitude.isna(),'EPA ID'])|set(coc.loc[coc.Latitude.isna()|coc.Longitude.isna(),'EPA ID'])
  fetch_frs(ids)
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'scripts/s23_external_sources.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
