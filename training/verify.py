"""No training: source provenance, change boundaries and input identity checks."""
import ast,hashlib,json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=['FAT','Uniform_Transformer','Preaggregation_Transformer','RF']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def tree(s):return ast.dump(ast.parse(s),include_attributes=False)
def check_runtime_defaults():
    c=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['dl']
    expected={
        'EPOCH': int(c['epochs']),
        'LEARNING_RATE': float(c['learning_rate']),
        'BATCH_SIZE': int(c['batch_size']),
        'L2': float(c['weight_decay']),
    }
    scripts=[
        ROOT/'FAT/Pro_FAT_HLS_carbon_CV5_ablation_fixed_fairfix_remove_selected_sites.py',
        ROOT/'Uniform_Transformer/Pro_FAT_HLS_carbon_CV5_ablation_fixed_fairfix_remove_selected_sites.py',
        ROOT/'Preaggregation_Transformer/Pro_FAT_HLS_carbon_CV5_preagg_strict.py',
    ]
    for p in scripts:
        tree_ast=ast.parse(p.read_text(encoding='utf-8'))
        found={}
        for node in ast.walk(tree_ast):
            if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name in expected and isinstance(node.value,ast.Constant): found[name]=node.value.value
        for name,val in expected.items():
            assert name in found and found[name]==val, f'DL standalone default mismatch: {p.relative_to(ROOT)} {name}={found.get(name)!r}, expected {val!r}'

def check_history_against_original():
    """Model_history may differ from the original FAT only by: balanced_loss compile, host NumPy data, replica-safe fit."""
    orig_src=(ROOT/'reference/original_code/FAT/Model_history.py').read_text(encoding='utf-8')
    cur_src=(ROOT/'FAT/Model_history.py').read_text(encoding='utf-8')
    o=ast.parse(orig_src);c=ast.parse(cur_src)
    def other(m):return [ast.dump(n) for n in m.body if not (isinstance(n,ast.FunctionDef) and n.name=='my_train_1schedule')]
    assert other(o)==other(c),'Model_history: non-training code changed'
    fo=next(n for n in o.body if isinstance(n,ast.FunctionDef) and n.name=='my_train_1schedule')
    fc=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='my_train_1schedule')
    assert ast.dump(fo.args)==ast.dump(fc.args)
    def compile_index(f):
        for i,s in enumerate(f.body):
            if isinstance(s,ast.Expr) and "attr='compile'" in ast.dump(s):return i
        raise AssertionError('no compile')
    io=compile_index(fo);ic=compile_index(fc)
    # everything before compile (schedule, optimizer choice, AdamW(weight_decay=decay,...)) must be identical
    pre_o=[ast.dump(s) for s in fo.body[:io]];pre_c=[ast.dump(s) for s in fc.body[:ic] if not (isinstance(s,ast.ImportFrom) and s.module=='distributed_runtime')]
    assert pre_o==pre_c,'Model_history: optimizer/schedule changed'
    assert 'balanced_loss(loss)' in cur_src and 'fit_with_safe_batches' in cur_src and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in ast.walk(fc))

def verify():
    check_runtime_defaults()
    config=json.loads((ROOT/'config.json').read_text())
    assert config['dl']=={'epochs':30,'learning_rate':.0001,'batch_size':16,'weight_decay':.1,'units':64,'embedding_layers':2,'transformer_layers':4,'heads':4,'dropout':.1,'validation_split':.04}
    assert [config['rf'][k] for k in ['n_estimators','max_depth','min_samples_leaf']]==[200,10,10]
    source=json.loads((ROOT/'reference/source_manifest.json').read_text())
    for rel,info in source.items():
        assert sha(ROOT/'reference/original_code'/rel)==info['sha256'],rel
    for file in ['Model_build.py','Model_build_fairfix.py']:
        actual=(ROOT/'FAT'/file).read_text(encoding='utf-8')
        expected=(ROOT/'reference/original_code/FAT'/file).read_text(encoding='utf-8').replace('layers.BatchNormalization()','layers.BatchNormalization(synchronized=True)')
        assert tree(actual)==tree(expected),'Only synchronized BatchNorm may differ from original: '+file
    assert sha(ROOT/'FAT/train_test.py')==sha(ROOT/'reference/original_code/FAT/train_test.py'),'train_test.py'
    check_history_against_original()
    for name in ['Uniform_Transformer','RF','shared']:
        for file in ['Model_build.py','Model_build_fairfix.py','Model_history.py','train_test.py']:
            assert sha(ROOT/name/file)==sha(ROOT/'FAT'/file),name+'/'+file
    assert sha(ROOT/'Preaggregation_Transformer/Model_build_fairfix.py')==sha(ROOT/'FAT/Model_build_fairfix.py')
    for name in MODELS:
        assert sha(ROOT/name/'myloss.py')==sha(ROOT/'FAT/myloss.py')
    import batching
    assert batching.batch_sizes(19905,16,2)[-2:]==[9,8] and sum(batching.batch_sizes(19905,16,2))==19905 and min(batching.batch_sizes(19905,16,2))>=2
    assert batching.batch_sizes(19846,16,2)[-1]==6 and batching.batch_sizes(18000,16,2)[-1]==16
    assert config['sensor_encoding']=='raw_0_1'
    assert config['dl']['validation_split']==0.04
    ug=config['uniform_geometry']
    assert ug['time_resolution_minutes']==30 and ug['timesteps']==48 and ug['no_record_median'] is True
    # Pure-NumPy smoke check: Uniform must produce independent [N,T,H,W] weights.
    sys_path_added=False
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0,str(ROOT));sys_path_added=True
    from height_uniform import height_uniform_weights_numpy as _huw
    _f=np.zeros((2,48,103),np.float32);_f[:,:,96]=1.0
    _f[1,:,96]=np.linspace(.45,2.50,48,dtype=np.float32)
    _v=np.ones((2,50,50),dtype=bool)
    _uw=_huw(_v,_f)
    assert _uw.shape==(2,48,50,50)
    assert np.allclose(_uw.sum(axis=(2,3)),1.0,atol=1e-6)
    assert not np.array_equal(_uw[1,0],_uw[1,-1]), 'Uniform geometry is still record-level; expected timestep-specific weights.'
    assert config['footprint_normalization']=='max_scaled_exact_sum_no_epsilon'
    for name in MODELS:
        runner=next((ROOT/name).glob('Pro_*.py'))
        nodes=ast.walk(ast.parse(runner.read_text(encoding='utf-8')))
        calls=[n for n in nodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='normalize_flux_train_test']
        assert len(calls)==1,name
        kw={v.arg:v.value for v in calls[0].keywords}
        assert isinstance(kw['use_sensor'],ast.Constant) and kw['use_sensor'].value is False,name
    for name in ['Preaggregation_Transformer','RF','shared']:
        assert sha(ROOT/name/'model_heads_ffp.py')==sha(ROOT/'FAT/model_heads_ffp.py'),name
    # Only a finite zero subgradient was added: positive loss values are unchanged.
    original=(ROOT/'reference/original_code/FAT/myloss.py').read_text(encoding='utf-8')
    current=(ROOT/'FAT/myloss.py').read_text(encoding='utf-8')
    oc={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    cc={n.name:n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)}
    for name,node in oc.items():
        if name=='mask_loss_physical_gpp':
            segment=ast.get_source_segment(current,cc[name]).replace('_sqrt_with_zero_subgradient(', 'tf.sqrt(')
            assert tree(segment)==tree(ast.get_source_segment(original,node))
        else:assert ast.dump(node)==ast.dump(cc[name]),name
    for name in MODELS:
        assert sha(ROOT/name/'myloss.py')==sha(ROOT/'FAT/myloss.py')
    fat_script='Pro_FAT_HLS_carbon_CV5_ablation_fixed_fairfix_remove_selected_sites.py'
    assert sha(ROOT/'FAT'/fat_script)==sha(ROOT/'Uniform_Transformer'/fat_script)
    assert tree((ROOT/'Preaggregation_Transformer/Model_build_preagg_fairfix.py').read_text(encoding='utf-8'))==tree((ROOT/'reference/original_code/Preaggregation_Transformer/Model_build_preagg_fairfix.py').read_text(encoding='utf-8').replace('BatchNormalization()','BatchNormalization(synchronized=True)')),'Preagg builder may differ only by synchronized BatchNorm'
    for name in MODELS:
        assert sha(ROOT/name/'train_test.py')==sha(ROOT/'FAT/train_test.py')
    f=(ROOT/'FAT/Model_history.py').read_text(encoding='utf-8')
    p=(ROOT/'Preaggregation_Transformer/Model_history_preagg.py').read_text(encoding='utf-8')
    expected=f.replace('model, HLS_train_norm1,FLUX_train_norm1,FFP_train_tensor,','model, HLS_train_norm1,FLUX_train_norm1,').replace('[HLS_train_norm1, FLUX_train_norm1, FFP_train_tensor]','[HLS_train_norm1, FLUX_train_norm1]')
    assert tree(p)==tree(expected),'PREAGG training differs beyond input interface'
    # Uniform hook must leave every original function except build_weight_head intact.
    fa=ast.parse((ROOT/'FAT/model_heads_ffp.py').read_text(encoding='utf-8'))
    un=ast.parse((ROOT/'Uniform_Transformer/model_heads_ffp.py').read_text(encoding='utf-8'))
    udf={n.name:n for n in un.body if isinstance(n,ast.FunctionDef)}
    for n in fa.body:
        if isinstance(n,ast.FunctionDef) and n.name!='build_weight_head':assert ast.dump(n)==ast.dump(udf[n.name]),n.name
    for name in MODELS:
        for file in (ROOT/name).glob('*.py'):ast.parse(file.read_text(encoding='utf-8-sig'))
    lock=ROOT/'audit/code_lock.json'
    assert lock.exists(),'Missing audit/code_lock.json: fair-critical code must be hash-locked.'
    locked=json.loads(lock.read_text(encoding='utf-8'))
    active={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.py') if 'results' not in p.relative_to(ROOT).parts and 'original_code' not in p.relative_to(ROOT).parts}
    assert active.issubset(set(locked)), 'Unreviewed active Python files: '+str(sorted(active-set(locked)))
    for rel,digest in locked.items():
        path=ROOT/rel
        assert path.is_file(), 'Missing locked file: '+rel
        assert sha(path)==digest,'Changed after audit: '+rel
    assert 'verify.py' in locked and 'run.py' in locked and 'config.json' in locked, 'Core audit/control files are not locked.'
    print('CODE CONSISTENCY PASS (V6): FAT network/loss/optimizer preserved; sync-BN, host-data and replica-safe batching are the only permitted changes; four-model controls and code lock verified. See audit/RELEASE_REPORT.md for validation limits.',flush=True)
    return True

def verify_inputs(root):
    ref=json.loads((ROOT/'reference/input_manifest.json').read_text())
    for name,info in ref.items():
        p=Path(root)/name
        print('Verifying input SHA256: '+name,flush=True)
        assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],'Input identity mismatch: '+str(p)
    print('PASS: all three complete input files match.',flush=True)
if __name__=='__main__':verify()
