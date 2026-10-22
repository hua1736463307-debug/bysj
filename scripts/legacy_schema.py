"""Extract only legacy table declarations, never import the old application's engine.

The retained source hash and DDL are migration evidence. This helper does not run
module-level startup, create_engine, migrations, default users or filesystem code.
"""
from __future__ import annotations
import ast
from datetime import datetime
import hashlib
from pathlib import Path
import uuid
import os
import subprocess

import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

SHARED_CLASSES = {"User", "DatasetResource", "DatasetVersion", "DatasetAsset", "AnnotationDraft", "AnnotationDraftAsset", "JobTask", "JobEvent", "ModelRegistry"}
BASELINE = "113181f498bac6d16988f97261ad628c9cdc8ea8"
REPOSITORY = Path(__file__).resolve().parents[1]


def read_legacy_source(relative):
    """Read the fixed Git baseline even after runtime cleanup.

    Offline container tests may use an exported fixture directory plus its Git
    object hash manifest. No legacy application source enters the runtime image.
    """
    relative = Path(relative).as_posix()
    if relative.startswith('/') or '..' in Path(relative).parts:
        raise ValueError("invalid legacy source path")
    fixture = os.environ.get('BYSJ_LEGACY_FIXTURES')
    if fixture:
        root = Path(fixture)
        import json
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['baseline'] != BASELINE:
            raise ValueError('legacy fixture baseline mismatch')
        source = (root/relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != manifest['files'][relative]:
            raise ValueError('legacy fixture hash mismatch')
        return source.decode('utf-8-sig')
    return subprocess.check_output(['git','-C',str(REPOSITORY),'show',f'{BASELINE}:{relative}']).decode('utf-8-sig')


def legacy_metadata(path: Path):
    source = read_legacy_source('backend/database.py').encode('utf-8')
    tree = ast.parse(source.decode("utf-8-sig"))
    namespace = {name: getattr(sa, name) for name in ("BigInteger", "Boolean", "Column", "DateTime", "Float", "ForeignKey", "Index", "Integer", "JSON", "String", "Text", "UniqueConstraint")}
    namespace.update(Base=declarative_base(), datetime=datetime, uuid=uuid)
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in SHARED_CLASSES:
            node.body = [item for item in node.body if isinstance(item, ast.Assign)]
            node.decorator_list = []
            definitions.append(node)
    if {node.name for node in definitions} != SHARED_CLASSES:
        raise ValueError("legacy source does not contain the nine expected shared models")
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Base"].metadata, hashlib.sha256(source).hexdigest()


def structure(connection, tables=None):
    inspector = sa.inspect(connection)
    result = {}
    for name in sorted(tables or inspector.get_table_names()):
        result[name] = {
            "columns": [{"name": c["name"], "type": str(c["type"].compile(dialect=connection.dialect)), "nullable": c["nullable"], "default": c.get("default")} for c in inspector.get_columns(name)],
            "primary_key": inspector.get_pk_constraint(name)["constrained_columns"],
            "unique_constraints": sorted([u["column_names"] for u in inspector.get_unique_constraints(name)]),
            "foreign_keys": [{k: f[k] for k in ("constrained_columns", "referred_table", "referred_columns", "options")} for f in inspector.get_foreign_keys(name)],
            "indexes": [{"name": i["name"], "columns": i["column_names"], "unique": i["unique"], "duplicates_constraint": i.get("duplicates_constraint")} for i in inspector.get_indexes(name)],
        }
    return result
