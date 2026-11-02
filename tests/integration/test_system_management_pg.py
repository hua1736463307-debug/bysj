from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from app.core.errors import ApiError
from app.core.config import get_settings
from app.db.models import User, AuthSession, SystemDepartment
from app.domains import auth_service, system_service


def test_case_insensitive_user_creation_is_atomic(sessions):
    barrier = Barrier(2)
    def create(name):
        with sessions() as db:
            barrier.wait(timeout=10)
            try:
                auth_service.create_user(db, username=name, password='integration-password')
                return 'created'
            except ApiError as error:
                return error.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, ['ConcurrentUser', 'concurrentuser']))
    assert sorted(results) == ['created', 'username_taken']


def test_department_cycle_is_rejected_under_concurrent_updates(sessions):
    with sessions() as db:
        a = system_service.save_department(db, {'name': 'A', 'parent_id': None, 'status': 'active', 'sort_order': 0})
        b = system_service.save_department(db, {'name': 'B', 'parent_id': None, 'status': 'active', 'sort_order': 0})
        identifiers = [(a.id, b.id), (b.id, a.id)]
    barrier = Barrier(2)
    def move(pair):
        with sessions() as db:
            barrier.wait(timeout=10)
            try:
                system_service.save_department(db, {'name': 'moved', 'parent_id': pair[1], 'status': 'active', 'sort_order': 0}, pair[0])
                return 'saved'
            except ApiError as error:
                return error.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(move, identifiers))
    assert sorted(results) == ['department_cycle', 'saved']


def test_disable_serializes_with_login(sessions):
    with sessions() as db:
        user = auth_service.create_user(db, username='login-race', password='integration-password')
        identifier = user.id
    def login():
        with sessions() as db:
            try:
                auth_service.authenticate(db, get_settings(), username='login-race', password='integration-password')
                return 'logged-in'
            except ApiError as error:
                return error.code
    with ThreadPoolExecutor(max_workers=1) as executor:
        with sessions() as db:
            db.query(User).filter_by(id=identifier).with_for_update().one()
            future = executor.submit(login)
            auth_service.update_user(db, identifier, {'status': 'disabled'})
        assert future.result(timeout=15) == 'user_disabled'
    with sessions() as db:
        assert db.query(AuthSession).count() == 0
