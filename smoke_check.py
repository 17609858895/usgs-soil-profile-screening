"""Small deterministic checks; does not rerun the complete soil analysis."""
from pathlib import Path
import argparse,ast,sys,json,importlib,importlib.metadata,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all-imports',action='store_true',help='Also import optional analysis modules.');p.add_argument('--fit-model',action='store_true',help='Also fit a small synthetic LightGBM model.');a=p.parse_args()
 print('Checking syntax',flush=True)
 syntax=[]
 for f in sorted((ROOT/'scripts').glob('*.py')):ast.parse(f.read_text(encoding='utf8'));syntax.append(f.name)
 for f in ['run_analysis.py','smoke_check.py']:ast.parse((ROOT/f).read_text(encoding='utf8'))
 print('Checking conformal ranks and BH',flush=True)
 from pacd import conf_p,loo_p,bh,clean_reference
 np.testing.assert_allclose(conf_p([1,2,2,3],[1,2,4]),[1,.8,.2])
 np.testing.assert_allclose(loo_p([1,2,2,3]),[1,.75,.75,.25])
 np.testing.assert_allclose(loo_p([2,2,2]),[1,1,1])
 np.testing.assert_array_equal(bh([.001,.02,.04,.2],.05),[True,True,False,False])
 rng=np.random.default_rng(20261107)
 cal=rng.exponential(size=1000)
 keep=clean_reference(cal,use_evt=False,q=.01)
 assert keep.dtype==bool and keep.all()
 versions={}
 for line in (ROOT/'requirements.txt').read_text().splitlines():
  if not line or line.startswith('#'):continue
  name,expected_version=line.split('==');versions[name]=importlib.metadata.version(name)
  assert versions[name]==expected_version,(name,versions[name],expected_version)
 r2=None
 if a.fit_model:
  print('Fitting synthetic background model',flush=True)
  from lightgbm import LGBMRegressor
  from sklearn.metrics import r2_score
  x=rng.normal(size=(240,4));y=1.7*x[:,0]-.5*x[:,1]+rng.normal(0,.05,len(x))
  model=LGBMRegressor(n_estimators=80,num_leaves=7,min_child_samples=8,learning_rate=.08,verbosity=-1,n_jobs=1,random_state=20261107)
  model.fit(x[:160],y[:160]);pred=model.predict(x[160:]);r2=float(r2_score(y[160:],pred))
  assert np.isfinite(pred).all() and r2>.5
 print('Checking frozen national counts',flush=True)
 d=pd.read_csv(ROOT/'frozen_results/national_discoveries.csv')
 expected={'Sb':16,'Pb':7,'Zn':6,'Cd':3,'Se':1}
 assert len(d)==33 and d.SiteID.nunique()==31 and d.element.value_counts().to_dict()==expected
 v=pd.read_csv(ROOT/'frozen_results/s16_national_variants.csv')
 v=v[(v.variant=='cleaned_labelled')&v.score.isin(expected)]
 assert v.groupby('score').n_disc.sum().to_dict()==dict(sorted(expected.items()))
 imported=[]
 if a.all_imports:
  for f in sorted((ROOT/'scripts').glob('*.py')):
   print('Importing',f.stem,flush=True);importlib.import_module(f.stem);imported.append(f.stem)
 report={'status':'passed','seed':20261107,'syntax_checked_modules':syntax,'dependency_versions':versions,'optional_imports':imported,'synthetic_validation_r2':r2,'frozen_element_site_pairs':len(d),'frozen_unique_sites':int(d.SiteID.nunique()),'frozen_element_counts':expected,'scope':'Core-function, dependency, syntax and frozen-summary checks. A non-null synthetic_validation_r2 additionally records the optional model-fit check. Complete analyses not rerun.'}
 (ROOT/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
