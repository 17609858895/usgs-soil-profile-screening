"""Run explicit analysis stages. Downloads are always separate, deliberate commands."""
from pathlib import Path
import os,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parent
STAGES={
 'prepare':[['s01_build_dataset.py']],
 'background':[['s07_inheritance_spectrum.py'],['s10_dual_anchor.py']],
 'injection':[['s12_injection_v3.py']],
 'reference':[['s20_clean_variants.py'],['s21_label_noise.py']],
 'national':[['prepare_external.py'],['s14_national_crossconformal.py'],['s22_evidence_labels.py'],['s16_national_variants.py'],['s17_mrds_validation.py']],
 'vertical':[['s18_depth_attenuation.py']],
 'landuse':[['s08_dml_v2.py'],['s11_dml_lithogenic_index.py']],
 'benchmark':[['s05_benchmark20.py','Pb,As,Cu,Zn']],
 'censoring':[['s13_censored_anchor.py']],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=list(STAGES)+['all']);a=p.parse_args()
 env=os.environ.copy()
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env.setdefault(key,'1')
 env.setdefault('PYTHONUTF8','1')
 selected=list(STAGES) if a.stage=='all' else [a.stage]
 for stage in selected:
  for command in STAGES[stage]:
   print('Running',stage,command,flush=True)
   subprocess.run([sys.executable,str(ROOT/'scripts'/command[0]),*command[1:]],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
