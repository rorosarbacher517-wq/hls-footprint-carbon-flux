import argparse,json,os,sys,subprocess,shutil,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=['FAT','Uniform_Transformer','Preaggregation_Transformer','RF']
SCRIPTS={'FAT':'Pro_FAT_HLS_carbon_CV5_ablation_fixed_fairfix_remove_selected_sites.py','Uniform_Transformer':'Pro_FAT_HLS_carbon_CV5_ablation_fixed_fairfix_remove_selected_sites.py','Preaggregation_Transformer':'Pro_FAT_HLS_carbon_CV5_preagg_strict.py','RF':'Pro_RF_HLS_carbon_auto_merge3_stable_pred_layout_fixed_fairfix_remove_selected_sites.py'}
DEFAULT_RESULTS='./data/external/results_v6_tf215'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODELS+['All'],default='All')
    p.add_argument('--fold',type=str,default='-1',help="-1 = all five folds; or one/several folds, e.g. 3 or 0,1")
    p.add_argument('--input-root',type=Path)
    p.add_argument('--results-root',type=Path,default=Path(DEFAULT_RESULTS))
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true',help='shortened-epoch test run; results root must contain "smoke"')
    p.add_argument('--epochs',type=int,help='only with --smoke')
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--check-runtime',action='store_true')
    a=p.parse_args();a.results_root=a.results_root.resolve();c=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    epochs=int(c['dl']['epochs'])
    if a.smoke:
        if a.epochs is None or a.epochs<1:raise RuntimeError('--smoke requires --epochs N (N>=1)')
        if 'smoke' not in str(a.results_root).lower():raise RuntimeError('Smoke results root must contain "smoke" so it can never be mistaken for production.')
        epochs=a.epochs
    elif a.epochs is not None:raise RuntimeError('--epochs is allowed only together with --smoke; production uses config.json (30).')
    elif 'smoke' in str(a.results_root).lower():raise RuntimeError('Production run must not use a smoke results root.')
    inp=a.input_root or (Path(os.environ['INPUTDATA_PATH']) if os.environ.get('INPUTDATA_PATH') else (Path(c['input_root']) if c['input_root'] else None))
    if inp is None and not a.check_runtime:
        raise RuntimeError('Provide --input-root /PATH/TO/INPUT_DATA (or set INPUTDATA_PATH).')
    try:
        fold_ids=[int(x) for x in str(a.fold).split(',')]
    except ValueError:
        raise RuntimeError('--fold must be -1 or a comma list of 0..4, got '+str(a.fold))
    if fold_ids==[-1]:fold_ids=list(range(5))
    if not fold_ids or any(k<0 or k>4 for k in fold_ids) or len(set(fold_ids))!=len(fold_ids):raise RuntimeError('--fold must be -1 or distinct folds in 0..4, got '+str(a.fold))
    a.fold=','.join(map(str,fold_ids)) if len(fold_ids)<5 else '-1'
    names=MODELS if a.model=='All' else [a.model]
    if a.child or a.check_runtime:
        os.environ['TF_DETERMINISTIC_OPS']='0'
        os.environ.pop('TF_CUDNN_DETERMINISTIC',None)
        from numpy_pickle_compat import install_numpy_pickle_compat
        install_numpy_pickle_compat()
        import numpy as np,tensorflow as tf
        if tf.__version__!='2.15.1':raise RuntimeError('V6 requires TensorFlow 2.15.1 (Keras 2, native AdamW) for every model.')
        if not hasattr(tf.optimizers,'AdamW'):raise RuntimeError('Native AdamW required; TF2.8/TFA substitution is not permitted.')
        if a.model!='RF':
            from distributed_runtime import get_strategy
            get_strategy()
        import sklearn,pandas,keras
        versions={'python':platform.python_version(),'numpy':np.__version__,'tensorflow':tf.__version__,'keras':keras.__version__,'scikit-learn':sklearn.__version__,'pandas':pandas.__version__,'floatx':tf.keras.backend.floatx(),'precision_policy':tf.keras.mixed_precision.global_policy().name}
        versions['distribution']={'replicas':2,'global_batch':16,'synchronized_batchnorm':True,'predict_global_batch':16,'small_last_batch_rule':'split_16+r_when_r<replicas'}
        versions['backend_flags']={k:os.environ.get(k) for k in ['TF_DETERMINISTIC_OPS','TF_CUDNN_DETERMINISTIC','XLA_FLAGS','TF_XLA_FLAGS','TF_ENABLE_ONEDNN_OPTS']}
        versions['tf32_enabled']=tf.config.experimental.tensor_float_32_execution_enabled()
        if a.check_runtime:
            print(json.dumps(versions,indent=2));return
        contract=Path(a.results_root)/'environment.json'
        if contract.exists() and json.loads(contract.read_text())!=versions:raise RuntimeError('Models must run in one matching software environment: '+str(contract))
        contract.write_text(json.dumps(versions,indent=2),encoding='utf-8')
        from verify import verify,sha
        verify()
        identity={'release':c['version'],'code_lock_sha256':sha(ROOT/'audit/code_lock.json'),'input_manifest_sha256':sha(ROOT/'reference/input_manifest.json'),'epochs':epochs,'smoke':bool(a.smoke),'environment':versions}
        identity_path=Path(os.environ['TRAIN_DATA_ROOT'])/'execution_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise RuntimeError('Cannot mix results from different code/data/epoch releases: '+str(identity_path))
        identity_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
        os.environ['STRICT_FAIR_RELEASE']=c['version']
        if a.smoke:os.environ['V6_SMOKE']='1'
        import runpy
        sys.path.insert(0,str(ROOT/a.model));sys.path.insert(1,str(ROOT))
        mode={'FAT':0,'Uniform_Transformer':1,'Preaggregation_Transformer':'preagg_dynamic_ffp','RF':0}[a.model]
        d=c['dl'];q=c['rf']
        params=[q['n_estimators'],q['max_depth'],q['min_samples_leaf']] if a.model=='RF' else [epochs,d['learning_rate'],d['batch_size'],d['weight_decay'],d['units'],d['embedding_layers'],d['transformer_layers']]
        sys.argv=[str(ROOT/a.model/SCRIPTS[a.model])]+list(map(str,params+[a.fold,mode,c['version']]))
        runpy.run_path(sys.argv[0],run_name='__main__');return
    from verify import verify
    verify()
    for name in names:
        output=Path(a.results_root)/name
        command=[sys.executable,'-B',str(ROOT/'run.py'),'--child','--model',name,'--fold',str(a.fold),'--results-root',str(a.results_root),'--input-root',str(inp.resolve())]
        if a.smoke:command+=['--smoke','--epochs',str(epochs)]
        print(json.dumps({'model':name,'input':str(inp),'output':str(output),'batch':None if name=='RF' else 16,'epochs':None if name=='RF' else epochs,'smoke':bool(a.smoke),'fold':a.fold,'command':command},indent=2),flush=True)
        if not a.execute:continue
        from verify import verify_inputs
        verify_inputs(inp)
        selected=fold_ids
        for k in selected:
            if output.exists() and list(output.glob('**/fold%d'%k)):
                raise RuntimeError('Selected fold already exists (possibly incomplete); preserve it before retry: '+str(output)+' fold'+str(k))
        subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--model',name,'--check-runtime','--input-root',str(inp.resolve()),'--results-root',str(a.results_root)]+(['--smoke','--epochs',str(epochs)] if a.smoke else []),check=True)
        output.mkdir(parents=True,exist_ok=True)
        if not (output/'split_kfold').exists():shutil.copytree(ROOT/'reference/split_kfold',output/'split_kfold')
        env=os.environ.copy();env.update(INPUTDATA_PATH=str(inp.resolve()),TRAIN_DATA_ROOT=str(output),REFERENCE_SPLIT_DIR=str(ROOT/'reference/split_kfold'),PYTHONHASHSEED='42',TF_DETERMINISTIC_OPS='0')
        env.pop('TF_CUDNN_DETERMINISTIC',None)
        if name=='RF':env['CUDA_VISIBLE_DEVICES']='-1'
        subprocess.run(command,cwd=str(ROOT/name),env=env,check=True)
        from result_contract import certify
        certify(ROOT,name,list(selected),a.results_root,None if name=='RF' else epochs)
    if not a.execute:print('PLAN ONLY. Add --execute to train, in this console.')
if __name__=='__main__':main()
