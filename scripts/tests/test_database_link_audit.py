import sqlite3
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from audit_database_links import orphan_query, audit_sql, RELATIONS


def test_missing_reference_detected_but_null_and_existing_are_allowed():
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE parents(id INTEGER PRIMARY KEY); CREATE TABLE children(parent_id INTEGER); INSERT INTO parents VALUES(7); INSERT INTO children VALUES(7),(NULL),(9);')
    query=orphan_query('children','parent_id','parents')
    assert db.execute(query).fetchone()[0]==1
    db.execute('INSERT INTO parents VALUES(9)')
    assert db.execute(query).fetchone()[0]==0
    db.close()


def test_schema_identifiers_cannot_inject_sql():
    with pytest.raises(ValueError):orphan_query('children; DROP TABLE parents','parent_id','parents')


def test_checks_cover_unique_relations_in_read_only_snapshot():
    assert len(RELATIONS)==len(set(RELATIONS))==48
    text=audit_sql()
    assert text.startswith('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;') and text.endswith('COMMIT;')
