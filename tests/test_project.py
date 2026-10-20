import numpy as np
import pandas as pd
import pytest
from latentlab.images import extract_arrays, TARGET, CHEAP_FEATURES
from latentlab.pipeline import make_demo, analyze


def test_extract_3d_arrays():
    pre=np.ones((12,12,12),dtype=np.float32)
    post=np.ones_like(pre)*1.2
    pre[2:8,2:8,2:8]+=np.indices((6,6,6))[0]*0.03
    post[2:8,2:8,2:8]+=np.indices((6,6,6))[0]*0.08
    mask=np.zeros_like(pre);mask[2:8,2:8,2:8]=1
    f=extract_arrays(pre,post,mask,spacing=(.8,.7,1.2))
    assert np.isclose(f['tumor_volume_ml'],216*.8*.7*1.2/1000)
    assert f[TARGET]>0 and all(k in f for k in CHEAP_FEATURES)


def test_reject_misalignment():
    x=np.ones((6,6,6));mask=np.ones_like(x)
    with pytest.raises(ValueError): extract_arrays(x,x[:5],mask)


def test_reject_duplicate_patients(tmp_path):
    p=tmp_path/'toy.csv';make_demo(p,60)
    d=pd.read_csv(p);d.loc[1,'patient_id']=d.loc[0,'patient_id'];d.to_csv(p,index=False)
    with pytest.raises(ValueError,match='duplicated'):
        analyze(p,tmp_path/'results',epochs=3,skip_umap=True)


def test_end_to_end_synthetic_smoke(tmp_path):
    p=tmp_path/'toy.csv';make_demo(p,60)
    result=analyze(p,tmp_path/'results',heldout='DUKE',epochs=4,skip_umap=True)
    assert result['n_train']==45 and result['n_test']==15
    assert 'surrogate_holdout_mae' in result
    assert (tmp_path/'results'/'surrogate.png').is_file()
    assert (tmp_path/'results'/'embeddings.csv').is_file()


def test_indexer(tmp_path):
    from latentlab.indexer import build_manifest
    pid="ISPY1_1221"
    folder=tmp_path/"images"/pid;folder.mkdir(parents=True)
    seg=tmp_path/"segmentations"/"expert";seg.mkdir(parents=True)
    for p in [folder/f"{pid}_0000.nii.gz",folder/f"{pid}_0001.nii.gz",seg/f"{pid}.nii.gz"]:
        p.write_bytes(b"fixture-not-an-image")
    report=build_manifest(tmp_path,tmp_path/"manifest.csv")
    assert report["indexed"]==1
    assert pd.read_csv(tmp_path/"manifest.csv").iloc[0]["cohort"]=="ISPY1"
