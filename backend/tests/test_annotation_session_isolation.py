import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException
from PIL import Image

from database import Base, DatasetVersion, DatasetAsset, AnnotationDraft, AnnotationDraftAsset, JobTask
from annotation_session_classes import draft_dataset_view, isolated_draft, normalize_class_context
import yolo_api as api


class AnnotationSessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'datasets/source'
        self.root.mkdir(parents=True)
        self.engine = create_engine('sqlite:///:memory:',connect_args={'check_same_thread':False},poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.original = {'annotation_classes': ['person', 'badge', 'mobilephone', 'tv', 'cell phone'],
                         'annotation_inactive_class_ids': [2, 3], 'annotation_class_labels_zh': {'cell phone': '旧手机'},
                         'annotation_model_path': '/models/old.pt'}
        self.version = DatasetVersion(id='source-v1', dataset_id='source', dataset_role='original',
            status='ready', root_path=str(self.root), manifest_path=str(self.root/'manifest.json'),
            class_map_json={}, metadata_json=copy.deepcopy(self.original), content_hash='source-hash')
        self.db.add(self.version)
        for index in range(3):
            Image.new('RGB', (100, 100)).save(self.root/f'{index}.png')
            self.db.add(DatasetAsset(id=index+1, dataset_version_id=self.version.id,
                relative_path=f'{index}.png', basename=f'{index}.png', width=100,height=100,
                content_hash=str(index), annotations_json=[{'yolo_line':'4 0.5 0.5 0.2 0.2'}]))
        self.db.commit()
        self.user = {'id': 1, 'username': 'owner', 'role': 'admin'}
        self.http = SimpleNamespace(state=SimpleNamespace(current_user=self.user))
        self.weight = self.root/'best.pt';self.weight.write_bytes(b'fixture')
        self.record = SimpleNamespace(id='model-fpi',path=str(self.weight),name='best.pt',source='trained')
        self.record.to_dict=lambda: {'id':self.record.id, 'path':str(self.weight), 'source':'trained',
                                   'model_family':'yolo','base_model':'yolo26m.pt'}
        self.patches = [patch.object(api,'require_dataset_manage'),patch.object(api,'require_dataset_read'),
                        patch.object(api,'_validate_annotation_model_selection',return_value=self.record),
                        patch.object(api,'resolve_registered_model_path',return_value=self.weight),
                        patch.object(api,'checkpoint_class_names',return_value=['phone']),
                        patch.object(api,'build_dataset_info_from_metadata',return_value={'name':'source'})]
        for item in self.patches:item.start()

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.db.close();self.engine.dispose();self.tmp.cleanup()

    def create(self, kind='yolo', classes=None, **extra):
        with patch.object(api,'checkpoint_class_names',return_value=classes or ['phone']):
            return asyncio.run(api.create_annotation_draft('source',kind,
                {'model_path':str(self.weight),'annotation_geometry':'rect','trained_class_names':['untrusted'],**extra},
                self.http,self.db))

    def save(self, response, **overrides):
        payload={'revision':response['revision'],'model_path':str(self.weight),
                 'class_names':response['class_names'],'class_bindings':response['class_bindings'],'assets':[],**overrides}
        return asyncio.run(api.put_annotation_draft('source',response['annotation_kind'],payload,self.http,self.db))

    def test_new_fpi_uses_weight_names_and_never_imports_source_retirement(self):
        result=self.create()
        self.assertEqual(result['class_names'],['phone'])
        self.assertEqual(result['class_scope'],'model_session')
        self.assertEqual(result['class_context']['inactive_class_ids'],[])
        self.db.refresh(self.version);self.assertEqual(self.version.metadata_json,self.original)

    def test_new_images_are_blank_even_when_original_has_old_class_four(self):
        self.create()
        result=asyncio.run(api.list_dataset_images('source',self.http,page=1,page_size=2,
            include_dimensions=False,search='',dataset_role='original',draft_kind='yolo',db=self.db))
        self.assertEqual(result['draft_class_names'],['phone'])
        self.assertEqual(result['annotation_classes'],['phone'])
        self.assertEqual([r['annotations'] for r in result['images']],[[],[]])
        self.assertTrue(result['has_more'])

    def test_resume_and_page_two_keep_draft_classes_labels_and_annotations(self):
        result=self.create()
        saved=self.save(result,class_context={'labels_zh':{'phone':'手机'},'inactive_class_ids':[]},
            assets=[{'relative_path':'2.png','width':100,'height':100,'annotations':[self.box(0)]}])
        resumed=asyncio.run(api.get_annotation_draft_metadata('source','yolo',self.http,self.db))
        self.assertEqual(resumed['class_context']['labels_zh'],{'phone':'手机'})
        images=asyncio.run(api.list_dataset_images('source',self.http,page=2,page_size=2,
            include_dimensions=False,search='',dataset_role='original',draft_kind='yolo',db=self.db))
        self.assertEqual(images['images'][0]['annotations'][0]['class'],0)
        self.assertEqual(images['draft_class_scope'],'model_session')
        self.assertEqual(saved['revision'],resumed['revision'])

    @staticmethod
    def box(class_id):return {'class':class_id,'x':10,'y':10,'width':20,'height':20,'type':'rect'}

    def test_two_annotation_kinds_have_independent_classes(self):
        first=self.create()
        second=self.create('multimodal',['person','badge'])
        first_draft=self.db.get(AnnotationDraft,first['id']);second_draft=self.db.get(AnnotationDraft,second['id'])
        self.assertEqual(first_draft.class_names_json,['phone'])
        self.assertEqual(second_draft.class_names_json,['person','badge'])
        self.assertEqual(self.version.metadata_json,self.original)

    def test_existing_draft_requires_explicit_replacement(self):
        self.create()
        with self.assertRaises(HTTPException) as raised:self.create()
        self.assertEqual(raised.exception.status_code,409)
        self.assertEqual(self.db.query(AnnotationDraft).count(),1)

    def test_failed_model_selection_preserves_old_draft(self):
        first=self.create()
        with patch.object(api,'checkpoint_class_names',side_effect=ValueError('invalid')):
            with self.assertRaises(HTTPException):
                asyncio.run(api.create_annotation_draft('source','yolo',{'model_path':str(self.weight),
                    'replace_draft_id':first['id'],'replace_revision':first['revision']},self.http,self.db))
        self.assertEqual(self.db.get(AnnotationDraft,first['id']).class_names_json,['phone'])

    def test_confirmed_replacement_creates_distinct_id_and_preserves_other_kind(self):
        first=self.create();other=self.create('multimodal',['person','badge'])
        second=self.create(classes=['new-class'],replace_draft_id=first['id'],replace_revision=first['revision'])
        self.assertNotEqual(first['id'],second['id'])
        self.assertIsNone(self.db.get(AnnotationDraft,first['id']))
        self.assertEqual(self.db.get(AnnotationDraft,other['id']).class_names_json,['person','badge'])
        self.assertEqual(self.version.metadata_json,self.original)

    def test_bad_checkpoint_does_not_create_draft_or_change_source(self):
        with patch.object(api,'checkpoint_class_names',side_effect=ValueError('no classes')):
            with self.assertRaises(HTTPException):
                asyncio.run(api.create_annotation_draft('source','yolo',{'model_path':str(self.weight)},self.http,self.db))
        self.assertEqual(self.db.query(AnnotationDraft).count(),0)
        self.assertEqual(self.version.metadata_json,self.original)

    def test_model_is_locked_for_draft_lifetime(self):
        result=self.create()
        with self.assertRaises(HTTPException) as raised:self.save(result,model_path='/models/different.pt')
        self.assertEqual(raised.exception.status_code,409)

    def test_old_source_class_id_cannot_be_saved_into_new_phone_session(self):
        result=self.create()
        with self.assertRaises(HTTPException):
            self.save(result,assets=[{'relative_path':'0.png','annotations':[self.box(4)]}])
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),0)

    def test_active_old_session_job_blocks_replacement(self):
        first=self.create()
        self.db.add(JobTask(id='active-annotation',type='annotation',status='running',owner_username='owner',
                           input_json={'annotation_draft_id':first['id']}));self.db.commit()
        with self.assertRaises(HTTPException):
            self.create(replace_draft_id=first['id'],replace_revision=first['revision'])
        self.assertIsNotNone(self.db.get(AnnotationDraft,first['id']))

    def test_stale_revision_cannot_overwrite_saved_draft(self):
        result=self.create();self.save(result)
        with self.assertRaises(HTTPException) as raised:self.save(result)
        self.assertEqual(raised.exception.status_code,409)

    def test_source_version_drift_still_blocks_isolated_resume(self):
        result=self.create();draft=self.db.get(AnnotationDraft,result['id']);draft.source_version_id='stale';self.db.commit()
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(api.get_annotation_draft_metadata('source','yolo',self.http,self.db))
        self.assertEqual(raised.exception.status_code,409)

    def test_other_owner_cannot_read_or_infer_with_session(self):
        result=self.create();other=SimpleNamespace(state=SimpleNamespace(current_user={'id':2,'username':'other'}))
        with self.assertRaises(HTTPException):asyncio.run(api.get_annotation_draft_metadata('source','yolo',other,self.db))
        with self.assertRaises(HTTPException):api._annotation_request_draft(self.db,'source',result['id'],{'id':2},str(self.weight))

    def test_class_retirement_only_affects_current_session(self):
        first=self.create(classes=['phone','other']);second=self.create('multimodal',['person','badge'])
        first=self.save(first,assets=[{'relative_path':'0.png','width':100,'height':100,
            'annotations':[self.box(0),self.box(1)]}])
        self.save(first,class_context={'labels_zh':{},'inactive_class_ids':[1]})
        row=self.db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id==first['id']).one()
        self.assertEqual([r['class'] for r in row.annotations_json],[0])
        self.assertEqual(self.db.get(AnnotationDraft,second['id']).class_names_json,['person','badge'])
        self.db.refresh(self.version);self.assertEqual(self.version.metadata_json,self.original)

    def test_draft_view_never_marks_source_orm_row_dirty(self):
        result=self.create();draft=self.db.get(AnnotationDraft,result['id']);view=draft_dataset_view(self.version,draft)
        self.assertEqual(view.id,self.version.id)
        self.assertEqual(api.get_dataset_annotation_classes(view)[0],['phone'])
        self.assertEqual(api.get_inactive_annotation_class_ids(view),[])
        self.assertEqual(self.version.metadata_json,self.original)
        self.assertNotIn(self.version,self.db.dirty)

    def test_export_starts_blank_and_uses_only_current_draft_labels(self):
        result=self.create()
        self.save(result,assets=[{'relative_path':'0.png','width':100,'height':100,'annotations':[self.box(0)]}])
        updates,classes=api.build_annotation_export_updates(self.db,self.version,{
            'annotation_draft_id':result['id'],'owner_id':1,'classes':['phone'],'images':[]})
        self.assertEqual(classes,['phone'])
        self.assertEqual(set(updates),{'0.png','1.png','2.png'})
        self.assertTrue(updates['0.png']['annotations'][0]['yolo_line'].startswith('0 '))
        self.assertEqual(updates['1.png']['annotations'],[])
        self.assertEqual(updates['2.png']['annotations'],[])

    def test_export_rejects_old_dataset_class_list(self):
        result=self.create()
        with self.assertRaises(ValueError):api.build_annotation_export_updates(self.db,self.version,{
            'annotation_draft_id':result['id'],'owner_id':1,'classes':['person','badge','cell phone']})

    def test_legacy_drafts_keep_original_retirement_protection(self):
        result=self.create();draft=self.db.get(AnnotationDraft,result['id']);draft.class_context_json=None;self.db.commit()
        with self.assertRaises(HTTPException):self.save(result)
        self.assertFalse(isolated_draft(draft))

    def test_legacy_dataset_view_keeps_exact_source_object(self):
        self.assertIs(draft_dataset_view(self.version,None),self.version)

    def test_session_cannot_reenable_retired_id(self):
        with self.assertRaises(ValueError):normalize_class_context(['phone','other'],{'inactive_class_ids':[]},{'inactive_class_ids':[1]})

    def test_replaced_session_id_cannot_read_page_or_metadata(self):
        self.create()
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(api.list_dataset_images('source',self.http,page=1,page_size=2,
                include_dimensions=False,search='',dataset_role='original',draft_kind='yolo',draft_id='old-id',db=self.db))
        self.assertEqual(raised.exception.status_code,409)
        with self.assertRaises(HTTPException):
            asyncio.run(api.get_annotation_draft_metadata('source','yolo',self.http,self.db,draft_id='old-id'))

    def test_replaced_session_id_cannot_autosave_over_current_draft(self):
        result=self.create()
        with self.assertRaises(HTTPException):self.save(result,draft_id='old-id')
        self.assertEqual(self.db.get(AnnotationDraft,result['id']).class_names_json,['phone'])

    def test_read_view_overrides_nested_model_binding(self):
        self.version.metadata_json={**self.original,'annotation_model':{'path':'/models/old.pt','geometry':'polygon'}}
        self.db.commit();result=self.create();draft=self.db.get(AnnotationDraft,result['id'])
        view=draft_dataset_view(self.version,draft)
        self.assertEqual(api.get_dataset_annotation_model(self.db,view,allow_source_fallback=False)['path'],str(self.weight))
        self.assertEqual(self.version.metadata_json['annotation_model']['path'],'/models/old.pt')

    def test_class_edits_preserve_native_model_vocabulary(self):
        result=self.create()
        saved=self.save(result,class_names=['phone','manual class'])
        self.assertEqual(saved['class_context']['native_classes'],['phone'])
        self.assertEqual(self.version.metadata_json,self.original)

    def test_open_vocabulary_placeholder_ids_are_not_shown_as_semantic_classes(self):
        with patch.object(api,'annotation_model_supports_dynamic_text_prompts',return_value=True):
            result=self.create('multimodal',['0','1','2'])
        self.assertEqual(result['class_names'],['object'])
        self.assertEqual(result['class_context']['initialization_mode'],'dynamic_prompt')

    def test_source_retirement_does_not_change_draft_class_zero(self):
        result=self.create()
        self.version.metadata_json={**self.original,'annotation_inactive_class_ids':[0,2,3]}
        self.db.commit()
        self.save(result,assets=[{'relative_path':'1.png','width':100,'height':100,'annotations':[self.box(0)]}])
        images=asyncio.run(api.list_dataset_images('source',self.http,page=1,page_size=3,
            include_dimensions=False,search='',dataset_role='original',draft_kind='yolo',db=self.db))
        self.assertEqual(images['images'][1]['annotations'][0]['class'],0)

    def test_augmentation_copy_uses_session_classes_and_never_copies_old_labels(self):
        import dataset_augmentation_jobs as jobs
        first=self.create()
        first=self.save(first,assets=[{'relative_path':'0.png','width':100,'height':100,'annotations':[self.box(0)]}])
        session=self.db.get(AnnotationDraft,first['id'])
        task=JobTask(id='augmentation-isolation',type='dataset_augmentation',status='pending',
            owner_username='owner',input_json={'dataset_id':'source','source_version_id':self.version.id,
            'original_dataset_id':'source','create_new_dataset':True,'temporary_dataset':True,
            'output_name':'isolated-augmentation','config':{'mirror':'horizontal'},
            'source_relative_paths':['0.png'],'annotation_overrides':[],
            'annotation_draft_id':first['id'],'annotation_draft_owner_id':1,
            'draft_source_version_id':self.version.id,'annotation_draft_revision':first['revision'],
            'owner_id':1,'owner_username':'owner'})
        self.db.add(task);self.db.commit()
        with tempfile.TemporaryDirectory() as output:
            output=Path(output);factory=sessionmaker(bind=self.engine,expire_on_commit=False)
            next_id=[100]
            def assign_sqlite_asset_ids(session, *_):
                for row in session.new:
                    if isinstance(row,DatasetAsset) and row.id is None:
                        row.id=next_id[0];next_id[0]+=1
            event.listen(factory.class_,'before_flush',assign_sqlite_asset_ids)
            with patch.object(jobs,'SessionLocal',factory),patch.object(jobs,'DATA_ROOT',output), \
                 patch.object(jobs.constants,'DATASETS_DIR',str(self.root.parent)), \
                 patch.object(jobs,'is_cancel_requested',return_value=False), \
                 patch.object(jobs,'_update_progress'),patch.object(jobs,'update_job'):
                result=jobs.execute_dataset_augmentation_job(task.id)
            version=self.db.query(DatasetVersion).filter(DatasetVersion.dataset_id==result['dataset_id']).one()
            self.assertEqual(version.class_map_json,{'0':'phone'})
            rows=self.db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id==version.id).all()
            untouched=[row for row in rows if row.relative_path in ['1.png','2.png']]
            self.assertEqual([row.annotations_json for row in untouched],[[],[]])
            self.assertTrue(any(row.annotations_json for row in rows))
            # The new labels belong to this session and must remain visible/exportable.
            self.db.expire_all()
            page=asyncio.run(api.list_dataset_images('source',self.http,page=1,page_size=100,
                include_dimensions=False,search='',dataset_role='original',draft_kind='yolo',db=self.db))
            generated=[row for row in page['images'] if row['relative_path'] not in ['0.png','1.png','2.png']]
            self.assertTrue(generated)
            self.assertTrue(all(row['annotations'] for row in generated),
                [(row['relative_path'],row['annotations']) for row in generated])
            self.assertAlmostEqual(generated[0]['annotations'][0]['x'],.7)
            self.assertAlmostEqual(generated[0]['annotations'][0]['y'],.1)
            updates,classes=api.build_annotation_export_updates(self.db,version,{
                'annotation_draft_id':first['id'],'owner_id':1,'classes':['phone']})
            self.assertTrue(all(updates[row['relative_path']]['annotations'] for row in generated))
            transformed=updates[generated[0]['relative_path']]['annotations'][0]
            self.assertAlmostEqual(transformed['x'],.8)
            self.assertAlmostEqual(transformed['y'],.2)
            self.db.refresh(self.version);self.assertEqual(self.version.metadata_json,self.original)
            originals=self.db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id==self.version.id).all()
            self.assertTrue(all(row.annotations_json[0]['yolo_line'].startswith('4 ') for row in originals))

    def test_http_create_save_page_and_resume_stay_in_one_session(self):
        from fastapi import FastAPI
        import requests,socket,threading,time,uvicorn
        app=FastAPI()
        for route in api.app.routes:
            if getattr(route,'path','') in ['/datasets/{dataset_id}/annotation-drafts/{kind}', '/datasets/{dataset_id}/images']:
                app.add_api_route(route.path,route.endpoint,methods=list(route.methods))
        @app.middleware('http')
        async def owner(request,call_next):
            request.state.current_user=self.user
            return await call_next(request)
        def database():yield self.db
        app.dependency_overrides[api.get_db]=database
        sock=socket.socket();sock.bind(('127.0.0.1',0))
        base='http://127.0.0.1:'+str(sock.getsockname()[1])
        server=uvicorn.Server(uvicorn.Config(app,lifespan='off',log_level='error',access_log=False))
        thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
        deadline=time.monotonic()+5
        while not server.started and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(server.started)
        client=SimpleNamespace(**{method:(lambda path,method=method,**kwargs:
            requests.request(method,base+path,timeout=3,**kwargs)) for method in ['post','put','get']})
        try:
            created=client.post('/datasets/source/annotation-drafts/yolo',json={'model_path':str(self.weight),'trained_class_names':['person','badge']})
            self.assertEqual(created.status_code,200,created.text)
            draft=created.json();self.assertEqual(draft['class_names'],['phone'])
            saved=client.put('/datasets/source/annotation-drafts/yolo',json={
                'draft_id':draft['id'],'revision':draft['revision'],'class_names':['phone'],
                'assets':[{'relative_path':'0.png','width':100,'height':100,'annotations':[self.box(0)]}]})
            self.assertEqual(saved.status_code,200,saved.text)
            page=client.get('/datasets/source/images',params={'draft_kind':'yolo','draft_id':draft['id']})
            self.assertEqual(page.status_code,200,page.text)
            self.assertEqual(page.json()['images'][0]['annotations'][0]['class'],0)
            self.assertEqual(page.json()['images'][1]['annotations'],[])
            resumed=client.get('/datasets/source/annotation-drafts/yolo',params={'draft_id':draft['id']})
            self.assertEqual(resumed.json()['class_scope'],'model_session')
            stale=client.get('/datasets/source/images',params={'draft_kind':'yolo','draft_id':'old-id'})
            self.assertEqual(stale.status_code,409)
        finally:
            server.should_exit=True;thread.join(timeout=5);sock.close()
        self.db.refresh(self.version);self.assertEqual(self.version.metadata_json,self.original)


if __name__=='__main__':unittest.main()
