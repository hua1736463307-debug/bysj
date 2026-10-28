import hashlib
import json
from pathlib import Path
import sys
import uuid

import pytest
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from legacy_schema import legacy_metadata
from migrate_legacy_snapshot import copy_snapshot, digest
from app.core.auth import hash_password
from app.core.config import get_settings
from app.db.models import DatasetVersion, DatasetAsset, AnnotationDraft, AnnotationDraftAsset, ModelRegistry


@pytest.fixture
def legacy_source(pg_engine, tmp_path):
    metadata, source_hash = legacy_metadata(Path(__file__).parents[2] / "backend/database.py")
    schema = "bysj_legacy_" + uuid.uuid4().hex
    with pg_engine.begin() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    url = pg_engine.url.update_query_dict({"options": f"-csearch_path={schema}"})
    engine = sa.create_engine(url)
    source_data = tmp_path / "legacy-data"
    source_models = tmp_path / "legacy-models"
    source_data.mkdir(); source_models.mkdir()
    resource_id, version_id, draft_id, job_id, model_id = [str(uuid.uuid4()) for _ in range(5)]
    root = source_data / "original"
    root.mkdir()
    annotations = [{"class_id": 0, "type": "bbox", "x": .5, "y": .5, "width": .2, "height": .4}]
    assets = []
    from PIL import Image
    for index, split in enumerate(("train", "val", "test"), 1):
        relative = f"{split}/images/{index}.jpg"
        path = root / relative; path.parent.mkdir(parents=True)
        Image.new("RGB", (40, 30), (index*50, 20, 30)).save(path)
        assets.append({"id": index+100, "dataset_version_id": version_id, "relative_path": relative, "basename": path.name,
            "size": path.stat().st_size, "width": 40, "height": 30, "source_group": f"original:{index}",
            "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(), "annotations_json": annotations, "split": split})
    manifest_path = source_data / "old-manifest.json"
    manifest = {"schema_version":1,"dataset_id":"legacy-bccd","dataset_role":"training","version_id":version_id,
        "class_names":["RBC"],"assets":[{**{k:v for k,v in a.items() if k not in {"id","dataset_version_id","annotations_json"}},"annotations":annotations} for a in assets]}
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    weight = source_models / "best.pt";weight.write_bytes(b"legacy-test-weight-not-a-torch-model")
    weight_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(metadata.tables["users"].insert().values(id=7,username="legacy-researcher",password_hash=hash_password("migration-test-password"),role="user"))
        conn.execute(metadata.tables["dataset_resources"].insert().values(id=resource_id,dataset_id="legacy-bccd",dataset_role="training",owner_user_id=7,owner_username="legacy-researcher"))
        conn.execute(metadata.tables["dataset_versions"].insert().values(id=version_id,dataset_id="legacy-bccd",dataset_role="training",root_path=str(root),manifest_path=str(manifest_path),class_map_json={"0":"RBC"},split_config_json={"seed":0},content_hash="a"*64,created_by="legacy-researcher"))
        for asset in assets:
            conn.execute(metadata.tables["dataset_assets"].insert().values(**{k:v for k,v in asset.items() if k!="split"}))
        conn.execute(metadata.tables["annotation_drafts"].insert().values(id=draft_id,owner_user_id=7,owner_username="legacy-researcher",dataset_id="legacy-bccd",source_version_id=version_id,annotation_kind="manual",model_path="",model_name="",revision=7,class_names_json=["RBC"]))
        conn.execute(metadata.tables["annotation_draft_assets"].insert().values(id=91,draft_id=draft_id,relative_path=assets[0]["relative_path"],width=40,height=30,annotations_json=annotations))
        conn.execute(metadata.tables["job_tasks"].insert().values(id=job_id,type="training",status="completed",owner_id="7",owner_username="legacy-researcher",input_json={"dataset_version_id":version_id},result_json={"model_id":model_id}))
        conn.execute(metadata.tables["job_events"].insert().values(id=81,task_id=job_id,sequence=1,event_type="completed",message="legacy complete"))
        conn.execute(metadata.tables["model_registry"].insert().values(id=model_id,name="legacy-best",path=str(weight),sha256=weight_hash,size=weight.stat().st_size,source="training",task_id=job_id,dataset_version_id=version_id,created_by="legacy-researcher"))
    try:
        yield {"engine":engine,"metadata":metadata,"source_hash":source_hash,"data":source_data,"models":source_models,
               "resource":resource_id,"version":version_id,"draft":draft_id,"job":job_id,"model":model_id,"assets":assets,"weight_hash":weight_hash}
    finally:
        engine.dispose()
        with pg_engine.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))


#: 旧库 9 张共享表之外、由当前 ORM/Alembic 创建的表。
#: 新增表必须同步更新此清单——该断言的作用就是阻止"加了表却没更新迁移测试"。
EXPECTED_NEW_TABLES = {
    "artifact_records",
    "auth_sessions",
    "dataset_imports",
    "evaluation_runs",
    "experiment_runs",
    "feedback_exports",
    "inference_items",
    "inference_predictions",
    "inference_runs",
    "resource_leases",
    "review_items",
    "review_records",
    # CXR 五标签（Alembic 4a6b1222f003）
    "cxr_samples",
    "cxr_analysis_batches",
    "cxr_analyses",
    "cxr_reviews",
    "cxr_reports",
    # 皮肤镜方向（Alembic 4a6b1222f004）
    "isic_samples",
}


def test_offline_legacy_copy_preserves_identity_files_and_api(sessions, legacy_source, tmp_path):
    old = legacy_source
    settings = get_settings()
    before = {str(p.relative_to(old["data"])):hashlib.sha256(p.read_bytes()).hexdigest() for p in old["data"].rglob('*') if p.is_file()}
    with old["engine"].connect().execution_options(isolation_level="REPEATABLE READ") as source, sessions() as target:
        source.execute(sa.text("SET TRANSACTION READ ONLY"))
        report = copy_snapshot(source,target,settings,expected_metadata=old["metadata"],source_data_root=old["data"],source_model_root=old["models"])
        assert report["source_counts"] == report["target_counts"]
        assert report["new_tables"] == sorted(EXPECTED_NEW_TABLES)
        version = target.get(DatasetVersion,old["version"])
        assert version.dataset_id == old["resource"] and version.content_hash == "a"*64
        assert version.state == "frozen" and version.manifest_sha256
        manifest=json.loads(Path(version.manifest_path).read_text())
        assert manifest["split_seed"] == 0
        assert {a["split"] for a in manifest["assets"]} == {"train","val","test"}
        assert target.get(AnnotationDraft,old["draft"]).revision == 7
        assert target.get(AnnotationDraftAsset,91).annotations_json[0]["w"] == .2
        assert target.get(ModelRegistry,old["model"]).sha256 == old["weight_hash"]
        assert target.get(DatasetAsset,101).annotations_json[0]["h"] == .4
        # Serial sequences must advance beyond preserved integer IDs.
        from app.db.models import User
        created=User(username="after-migration",password_hash=hash_password("another-test-password"))
        target.add(created);target.commit();assert created.id>7
        report["legacy_source_sha256"]=old["source_hash"]
        report["source_unchanged"] = before == {str(p.relative_to(old["data"])):hashlib.sha256(p.read_bytes()).hexdigest() for p in old["data"].rglob('*') if p.is_file()}
        assert report["source_unchanged"]
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app(settings)) as client:
        login=client.post('/api/v1/auth/login',json={"username":"legacy-researcher","password":"migration-test-password"})
        assert login.status_code==200,login.text
        headers={"Authorization":"Bearer "+login.json()["data"]["token"]}
        assert client.get('/api/v1/dataset-versions/'+old["version"],headers=headers).json()["data"]["state"]=="frozen"
        annotation=client.get('/api/v1/annotations/101',headers=headers).json()["data"]
        assert annotation["revision"]==7 and annotation["annotations"][0]["cx"]==.5
        image=client.get('/api/v1/assets/101/image',headers=headers)
        assert image.status_code==200 and hashlib.sha256(image.content).hexdigest()==old["assets"][0]["content_hash"]
        weight=client.get('/api/v1/models/'+old["model"]+'/download',headers=headers)
        assert weight.status_code==200 and hashlib.sha256(weight.content).hexdigest()==old["weight_hash"]
        assert client.get('/api/v1/jobs/'+old["job"],headers=headers).json()["data"]["status"]=="completed"
    report["status"]="synthetic_schema_and_api_pass_real_dump_not_verified"
    output=Path('artifacts/legacy-migration/summary.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def test_unrecognized_schema_rejected_before_copy(sessions,legacy_source):
    old=legacy_source
    with old["engine"].begin() as conn:
        conn.execute(sa.text("ALTER TABLE users ADD COLUMN unexpected_runtime_field TEXT"))
    with old["engine"].connect() as source,sessions() as target:
        with pytest.raises(ValueError,match="unrecognized legacy columns"):
            copy_snapshot(source,target,get_settings(),expected_metadata=old["metadata"],source_data_root=old["data"],source_model_root=old["models"])
        assert target.query(DatasetVersion).count()==0


def test_active_legacy_job_rejected_before_copy(sessions,legacy_source):
    old=legacy_source
    with old["engine"].begin() as conn:
        conn.execute(old["metadata"].tables["job_tasks"].update().values(status="running"))
    with old["engine"].connect() as source,sessions() as target:
        with pytest.raises(ValueError,match="quiesced"):
            copy_snapshot(source,target,get_settings(),expected_metadata=old["metadata"],source_data_root=old["data"],source_model_root=old["models"])
        assert target.query(DatasetVersion).count()==0
