import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sqlalchemy import BigInteger, create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from database import Base, AnnotationDraft, AnnotationDraftAsset, DatasetAsset, DatasetVersion, JobEvent, JobTask
from annotation_draft_batches import import_draft_batch, latest_draft_batch


@compiles(BigInteger, 'sqlite')
def sqlite_integer_primary_key(element, compiler, **kwargs):
    return 'INTEGER'


class DraftCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.version = DatasetVersion(id='version', dataset_id='source', dataset_role='original',
            status='ready', root_path='/datasets/source', manifest_path='/tmp/manifest.json',
            content_hash='unchanged', class_map_json={}, metadata_json={})
        self.draft = AnnotationDraft(id='draft', owner_user_id=1, owner_username='owner',
            dataset_id='source', source_version_id='version', annotation_kind='yolo', model_path='model.pt',
            model_name='model', annotation_geometry='rect', class_names_json=['phone'],
            class_bindings_json=[], revision=1)
        self.task = JobTask(id='task', type='annotation', status='completed', owner_id='1', owner_username='owner',
            finished_at=datetime.now()-timedelta(seconds=1), input_json={'annotation_draft_id':'draft',
            'dataset_id':'source', 'source_version_id':'version', 'model_path':'model.pt',
            'annotation_classes':['phone'], 'annotation_class_bindings':[], 'annotation_mode':'replace'}, result_json={})
        self.db.add_all([self.version,self.draft,self.task]); self.db.commit()

    def tearDown(self):
        self.db.close(); self.engine.dispose()

    def fixture(self, count=250, **item_overrides):
        for i in range(count):
            relative=f'nested/{i}.jpg'
            self.db.add(DatasetAsset(id=i+1,dataset_version_id='version',relative_path=relative,
                basename=f'{i}.jpg',width=100,height=200,content_hash=str(i),
                annotations_json=[{'yolo_line':'0 0.5 0.5 0.2 0.2'}]))
            self.db.add(JobEvent(id=i+1,task_id='task',sequence=i+1,event_type='annotation_result',
                payload_json={'item':{'path':f'/datasets/source/{relative}', 'filename':f'{i}.jpg',
                'status':'success','annotations':[{'class':0,'x':0.1,'y':0.2,'width':0.3,'height':0.4}],
                **item_overrides}}))
        self.db.commit()

    def sync(self, **kwargs):
        result=import_draft_batch(self.db,self.draft,self.version,self.version,'task',**kwargs)
        self.db.commit(); return result

    def test_all_250_images_survive_reopen_without_loading_pages(self):
        self.fixture()
        self.assertTrue(self.sync()['has_more'])
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),200)
        self.db.expire_all()  # New request resumes the persisted cursor.
        self.assertFalse(self.sync()['has_more'])
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),250)
        self.assertEqual(self.db.query(DatasetAsset).first().annotations_json,[{'yolo_line':'0 0.5 0.5 0.2 0.2'}])
        self.assertEqual(latest_draft_batch(self.db,self.draft)['synced'],True)

    def test_append_is_idempotent_and_keeps_original_labels(self):
        self.fixture(1);self.task.input_json={**self.task.input_json,'annotation_mode':'append'};self.db.commit()
        self.sync(); self.sync()
        annotations=self.db.query(AnnotationDraftAsset).one().annotations_json
        self.assertEqual(len(annotations),2)
        self.assertAlmostEqual(annotations[0]['x'],0.4)

    def test_replace_empty_result_is_persisted(self):
        self.fixture(1,annotations=[]);self.sync()
        self.assertEqual(self.db.query(AnnotationDraftAsset).one().annotations_json,[])

    def test_failed_image_does_not_erase_original_annotations(self):
        self.fixture(1,status='failed',annotations=[]);self.sync()
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),0)

    def test_manual_edits_after_task_finish_are_preserved(self):
        self.fixture(1)
        self.db.add(AnnotationDraftAsset(draft_id='draft',relative_path='nested/0.jpg',width=100,height=200,
            annotations_json=[{'class':0,'x':0.9}],updated_at=datetime.now()))
        self.db.commit();self.sync()
        self.assertEqual(self.db.query(AnnotationDraftAsset).one().annotations_json,[{'class':0,'x':0.9}])

    def test_transaction_rollback_does_not_advance_cursor(self):
        self.fixture(1)
        import_draft_batch(self.db,self.draft,self.version,self.version,'task')
        self.db.rollback()
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),0)
        self.assertNotIn('draft_sync',self.db.get(JobTask,'task').result_json)
        self.sync();self.assertEqual(self.db.query(AnnotationDraftAsset).count(),1)

    def test_owner_draft_model_source_and_classes_must_match(self):
        self.fixture(1)
        mutations=[{'annotation_draft_id':'other'},{'source_version_id':'old'},
                   {'model_path':'other.pt'},{'annotation_classes':['other']},
                   {'annotation_class_bindings':[{'page_index':0,'model_index':9}]}]
        original=dict(self.task.input_json)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.task.input_json={**original,**mutation};self.db.commit()
                with self.assertRaises(ValueError):self.sync()
                self.db.rollback()
        self.task.input_json=original;self.task.owner_id='2';self.db.commit()
        with self.assertRaises(ValueError):self.sync()

    def test_duplicate_filenames_use_full_nested_path(self):
        self.fixture(1)
        self.db.add(DatasetAsset(id=2,dataset_version_id='version',relative_path='other/0.jpg',basename='0.jpg',
            width=100,height=100,content_hash='other',annotations_json=[]));self.db.commit()
        self.sync();self.assertEqual(self.db.query(AnnotationDraftAsset).one().relative_path,'nested/0.jpg')

    def test_polygon_geometry_survives_server_import(self):
        self.fixture(1);self.draft.annotation_geometry='polygon';self.db.commit();self.sync()
        shape=self.db.query(AnnotationDraftAsset).one().annotations_json[0]
        self.assertEqual(shape['type'],'polygon');self.assertEqual(len(shape['points']),4)

    def test_cancelled_job_preserves_completed_results(self):
        self.fixture(2);self.task.status='cancelled';self.db.commit();self.sync()
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),2)

    def test_legacy_task_without_mode_is_not_automatically_reassociated(self):
        self.fixture(1)
        data=dict(self.task.input_json);data.pop('annotation_mode')
        self.task.input_json=data;self.db.commit()
        self.assertIsNone(latest_draft_batch(self.db,self.draft))
        with self.assertRaises(ValueError):self.sync()

    def test_isolated_append_does_not_inherit_original_annotations(self):
        self.fixture(1)
        self.draft.class_context_json={'scope':'model_session','inactive_class_ids':[]}
        self.task.input_json={**self.task.input_json,'annotation_mode':'append'}
        self.db.commit();self.sync()
        self.assertEqual(len(self.db.query(AnnotationDraftAsset).one().annotations_json),1)

    def test_isolated_retired_ids_cannot_reappear_from_batch(self):
        self.fixture(1)
        self.draft.class_context_json={'scope':'model_session','inactive_class_ids':[0]}
        self.db.commit();self.sync()
        self.assertEqual(self.db.query(AnnotationDraftAsset).one().annotations_json,[])

    def test_augmented_working_version_keeps_original_source_identity(self):
        self.fixture(1)
        working=DatasetVersion(id='working',dataset_id='temporary',dataset_role='original',
            status='ready',root_path='/datasets/temporary',manifest_path='/tmp/temporary.json',
            content_hash='temporary',class_map_json={},metadata_json={})
        self.db.add(working)
        self.db.add(DatasetAsset(id=900,dataset_version_id='working',relative_path='nested/0.jpg',
            basename='0.jpg',width=100,height=200,content_hash='temporary',annotations_json=[]))
        self.task.input_json={**self.task.input_json,'source_version_id':'working',
            'annotation_draft_source_version_id':'version'}
        self.db.commit()
        import_draft_batch(self.db,self.draft,self.version,working,'task');self.db.commit()
        self.assertEqual(self.db.query(AnnotationDraftAsset).count(),1)

    def test_api_commits_assets_and_checkpoint_and_restores_task(self):
        import yolo_api as api
        self.fixture(250)
        http=SimpleNamespace(state=SimpleNamespace(current_user={'id':1,'username':'owner','role':'admin'}))
        with patch.object(api,'require_dataset_manage'),patch.object(api,'require_dataset_read'), \
             patch.object(api,'_validate_annotation_model_selection'), \
             patch.object(api,'_get_latest_original_dataset_version',return_value=self.version), \
             patch.object(api,'_draft_current_version_or_conflict',return_value=self.version), \
             patch.object(api,'_draft_working_version',return_value=self.version):
            metadata=asyncio.run(api.get_annotation_draft_metadata('source','yolo',http,self.db))
            self.assertEqual(metadata['annotation_batch']['task_id'],'task')
            response=asyncio.run(api.put_annotation_draft('source','yolo',{'draft_id':'draft','revision':1,
                'model_path':'model.pt','class_names':['phone'],'class_bindings':[],
                'annotation_task_id':'task','assets':[]},http,self.db))
            self.assertTrue(response['batch_sync']['has_more'])
            response=asyncio.run(api.put_annotation_draft('source','yolo',{'draft_id':'draft',
                'revision':response['revision'],'model_path':'model.pt','class_names':['phone'],
                'class_bindings':[],'annotation_task_id':'task','assets':[]},http,self.db))
            self.assertFalse(response['batch_sync']['has_more'])
            self.assertEqual(response['saved_image_count'],250)

    def test_start_binds_draft_identity_mode_and_class_snapshot(self):
        import yolo_api as api
        self.fixture(1)
        http=SimpleNamespace(state=SimpleNamespace(current_user={'id':1,'username':'owner','role':'admin'}))
        def close_task(coroutine):
            coroutine.close()
            return SimpleNamespace()
        with patch.object(api,'require_dataset_manage'), \
             patch.object(api,'_resolve_annotation_request_model',side_effect=lambda db,path,*args:(path,None,{})), \
             patch.object(api,'list_annotation_image_paths_for_batch',return_value=[Path('/datasets/source/nested/0.jpg')]), \
             patch.object(api.asyncio,'create_task',side_effect=close_task), \
             patch.object(api,'build_annotation_batch_status_payload',side_effect=lambda task_id,*args:{'task_id':task_id}):
            response=asyncio.run(api.start_ai_assist_batch_annotation({'dataset_id':'source',
                'annotation_draft_id':'draft','annotation_draft_model_path':'model.pt','model_path':'model.pt',
                'annotation_classes':['phone'],'annotation_class_bindings':[],'annotation_mode':'append'},http,self.db))
        job=self.db.get(JobTask,response['task_id'])
        self.assertEqual(job.input_json['annotation_draft_id'],'draft')
        self.assertEqual(job.input_json['annotation_draft_class_bindings'],[])
        self.assertEqual(job.input_json['annotation_mode'],'append')
        api.annotation_batch_task_handles.pop(job.id,None)

    def test_clear_all_preserves_explicit_new_annotations_in_same_request(self):
        import yolo_api as api
        self.fixture(2)
        http=SimpleNamespace(state=SimpleNamespace(current_user={'id':1,'username':'owner','role':'admin'}))
        with patch.object(api,'require_dataset_manage'),patch.object(api,'_validate_annotation_model_selection'), \
             patch.object(api,'_get_latest_original_dataset_version',return_value=self.version), \
             patch.object(api,'_draft_working_version',return_value=self.version):
            asyncio.run(api.put_annotation_draft('source','yolo',{'draft_id':'draft','revision':1,
                'model_path':'model.pt','class_names':['phone'],'class_bindings':[],
                'clear_all_annotations':True,'assets':[{'relative_path':'nested/0.jpg',
                'annotations':[{'class':0,'x':0.1,'y':0.2,'width':0.3,'height':0.4}]}]},http,self.db))
        rows={row.relative_path:row.annotations_json for row in self.db.query(AnnotationDraftAsset)}
        self.assertEqual(len(rows['nested/0.jpg']),1);self.assertEqual(rows['nested/1.jpg'],[])


if __name__ == '__main__':unittest.main()
