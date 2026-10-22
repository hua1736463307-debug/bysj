import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from run_experiments import config_hash
from summarize_experiments import read_group,compare,stats


def protocol():
    return {'config':{'seeds':[42,43,44],'class_map':{'0':'cell'},'dataset_version_id':'v'},
        'build':{'code_sha':'a'*40,'requirements_lock_sha256':'b'*64,'ml_lock_sha256':'c'*64},
        'base':{'model_id':'base','sha256':'d'*64},'version':{'manifest_sha256':'e'*64},
        'split_asset_sha256':{'val':'val','test':'test'},'effective_requests':[{'seed':42,'params':{'seed':42}}]}


def test_no_finished_seeds_are_missing_not_zero(tmp_path):
    p=protocol();root=tmp_path/config_hash(p);root.mkdir();(root/'config.frozen.json').write_text(json.dumps(p))
    g=read_group(root)
    assert g['status']=='partial' and g['aggregate']['map50']['mean'] is None
    assert [r['status'] for r in g['seeds']]==['not_run']*3
    assert compare({'E1':g,'E2':g})['paired_seeds']==[]


def test_quality_warning_is_retained_separately_from_numerical_completeness(tmp_path):
    p=protocol();root=tmp_path/config_hash(p);root.mkdir();(root/'config.frozen.json').write_text(json.dumps(p))
    (root/'BLOCKED.json').write_text(json.dumps({'reason':'cross-split candidate'}))
    assert read_group(root)['validity_warning']['reason']=='cross-split candidate'


@pytest.mark.parametrize('change',['test','build','params'])
def test_incomparable_protocols_are_rejected(change):
    a={'protocol':protocol(),'completed':[]};b=copy.deepcopy(a)
    if change=='test':b['protocol']['split_asset_sha256']['test']='changed'
    if change=='build':b['protocol']['build']['code_sha']='changed'
    if change=='params':b['protocol']['effective_requests'][0]['params']['seed']=43
    with pytest.raises(ValueError):compare({'E1':a,'E2':b})


def test_sample_standard_deviation_and_empty_values():
    assert stats([])['mean'] is None and stats([1])['std'] is None
    assert stats([1,2,3])['mean']==2 and stats([1,2,3])['std']==1
