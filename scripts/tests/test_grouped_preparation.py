from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from prepare_grouped_bccd import grouped_sources


def test_clusters_are_transitive_and_preserve_prior_groups_without_using_split():
    assets=[{'relative_path':n,'source_group':g,'sha256':n*64,'split':s} for n,g,s in [('a','old','test'),('b','old','train'),('c','c','val'),('d','d','train')]]
    audit={'decoded_identical_pairs':[],'candidate_pairs':[{'a':'b','b':'c'}]}
    groups=grouped_sources(assets,audit)
    assert groups['a']==groups['b']==groups['c'] and groups['d']=='d'
    assert grouped_sources(list(reversed(assets)),audit)==groups


def test_unknown_audit_image_is_rejected():
    with pytest.raises(ValueError,match='unknown image'):
        grouped_sources([{'relative_path':'a','source_group':'a','sha256':'a'*64}],{'decoded_identical_pairs':[],'candidate_pairs':[{'a':'a','b':'missing'}]})
