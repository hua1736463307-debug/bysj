import sys
from types import SimpleNamespace
import pytest
from app.ml.train_cli import run


@pytest.mark.parametrize('failure',[None,'loader','nonfinite'])
def test_post_training_validation_is_bounded_and_must_succeed(tmp_path,monkeypatch,failure):
    best=tmp_path/'original.pt';best.write_bytes(b'test-weight')
    calls={}
    class Model:
        def __init__(self,*a,**k):self.trainer=SimpleNamespace(best=best,epoch=0,args=SimpleNamespace(workers=0))
        def train(self,**kwargs):calls['train']=kwargs;return SimpleNamespace(results_dict={})
        def val(self,**kwargs):
            calls['val']=kwargs
            if failure=='loader':raise RuntimeError('loader unavailable')
            return SimpleNamespace(box=SimpleNamespace(mp=.7,mr=.8,map50=.6,map=float('nan') if failure=='nonfinite' else .5))
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Model,__version__='test'))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setattr('app.ml.yolo_adapter.environment_snapshot',lambda:{'test':True})
    request={'base_model_path':str(best),'workdir':str(tmp_path),'dataset_yaml':'unused.yaml',
             'params':{'epochs':1,'batch':8,'imgsz':640,'device':'cpu'}}
    if failure:
        with pytest.raises(RuntimeError):run(request)
    else:
        result=run(request);assert result['metrics']['map50_95']==.5
    assert calls['val']['workers']==0 and calls['val']['batch']==8
