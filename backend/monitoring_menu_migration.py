"""One-time migration from the legacy inference menu to monitoring management."""

MONITORING_MIGRATION_KEY = "monitoring_management_v1"
MONITORING_CHILDREN = (
    ("监控配置", "/monitoring/config", "Setting", 1),
    ("实时监控", "/monitoring/live", "Monitor", 2),
)
MONITORING_EMBEDDED_ALGORITHM_KEY = "monitoring_algorithm_embedded_v1"


def _create_menu(db, Menu, **values):
    """Supply an id only for SQLite, whose BIGINT primary keys do not autoincrement."""
    if db.bind.dialect.name == "sqlite":
        last_row = db.query(Menu.id).order_by(Menu.id.desc()).first()
        values["id"] = int(last_row[0] if last_row else 0) + 1
    return Menu(**values)


def _create_role_menu(db, RoleMenu, **values):
    """Supply an id only for SQLite, whose BIGINT primary keys do not autoincrement."""
    if db.bind.dialect.name == "sqlite":
        last_row = db.query(RoleMenu.id).order_by(RoleMenu.id.desc()).first()
        values["id"] = int(last_row[0] if last_row else 0) + 1
    return RoleMenu(**values)


def _one(query, label):
    rows = query.all()
    if len(rows) > 1:
        raise RuntimeError(f"监控管理菜单迁移冲突：{label} 存在多条记录")
    return rows[0] if rows else None


def _grant_like_legacy(db, RoleMenu, legacy, targets):
    role_ids = [link.role_id for link in db.query(RoleMenu).filter(RoleMenu.menu_id == legacy.id).all()]
    for role_id in role_ids:
        for target in targets:
            exists = db.query(RoleMenu).filter(
                RoleMenu.role_id == role_id, RoleMenu.menu_id == target.id
            ).first()
            if not exists:
                db.add(_create_role_menu(db, RoleMenu, role_id=role_id, menu_id=target.id))


def migrate_monitoring_management(db, Menu, RoleMenu, SystemSeedState):
    """Migrate an existing exact legacy menu once; caller owns the commit."""
    if db.query(SystemSeedState).filter(SystemSeedState.key == MONITORING_MIGRATION_KEY).first():
        return False

    with db.begin_nested():
        legacy = _one(
            db.query(Menu).filter(Menu.path == "/inference-service", Menu.menu_type == "menu"),
            "旧版推理服务",
        )
        if legacy is None:
            db.add(SystemSeedState(key=MONITORING_MIGRATION_KEY))
            return False

        group = _one(db.query(Menu).filter(Menu.path == "/monitoring"), "监控管理目录")
        if group and group.menu_type != "directory":
            raise RuntimeError("监控管理路径已被非目录菜单占用")
        if group is None:
            group = _create_menu(
                db,
                Menu,
                parent_id=None,
                name="监控管理",
                path="/monitoring",
                icon="View",
                menu_type="directory",
                sort_order=legacy.sort_order,
                status=legacy.status,
                visible=legacy.visible,
            )
            db.add(group)
            db.flush()

        children = []
        for name, path, icon, sort_order in MONITORING_CHILDREN:
            child = _one(db.query(Menu).filter(Menu.path == path), name)
            if child and (child.menu_type != "menu" or child.parent_id not in (None, group.id)):
                raise RuntimeError(f"监控管理路径冲突：{path}")
            if child is None:
                child = _create_menu(
                    db,
                    Menu,
                    parent_id=group.id,
                    name=name,
                    path=path,
                    icon=icon,
                    menu_type="menu",
                    sort_order=sort_order,
                    status=legacy.status,
                    visible=legacy.visible,
                )
                db.add(child)
            elif child.parent_id is None:
                child.parent_id = group.id
            children.append(child)

        db.flush()
        legacy.parent_id = group.id
        legacy.name = "推理服务（旧版）"
        legacy.sort_order = 3
        _grant_like_legacy(db, RoleMenu, legacy, [group, *children])
        db.add(SystemSeedState(key=MONITORING_MIGRATION_KEY))
        db.flush()
    return True


def migrate_embedded_algorithm_config(db, Menu, RoleMenu, SystemSeedState):
    """Move legacy algorithm-page grants to monitoring config and remove its menu row."""
    if db.query(SystemSeedState).filter(SystemSeedState.key == MONITORING_EMBEDDED_ALGORITHM_KEY).first():
        return False

    changed = False
    with db.begin_nested():
        config_menu = _one(db.query(Menu).filter(Menu.path == "/monitoring/config"), "监控配置")
        algorithm_menu = _one(db.query(Menu).filter(Menu.path == "/monitoring/algorithms"), "算法配置")
        live_menu = _one(db.query(Menu).filter(Menu.path == "/monitoring/live"), "实时监控")
        legacy_menu = _one(db.query(Menu).filter(Menu.path == "/inference-service"), "旧版推理服务")

        if algorithm_menu is not None:
            if config_menu is not None:
                role_ids = [
                    link.role_id
                    for link in db.query(RoleMenu).filter(RoleMenu.menu_id == algorithm_menu.id).all()
                ]
                for role_id in role_ids:
                    exists = db.query(RoleMenu).filter(
                        RoleMenu.role_id == role_id,
                        RoleMenu.menu_id == config_menu.id,
                    ).first()
                    if not exists:
                        db.add(_create_role_menu(
                            db,
                            RoleMenu,
                            role_id=role_id,
                            menu_id=config_menu.id,
                        ))
            db.query(RoleMenu).filter(
                RoleMenu.menu_id == algorithm_menu.id
            ).delete(synchronize_session=False)
            db.delete(algorithm_menu)
            changed = True

        if config_menu is not None:
            config_menu.sort_order = 1
        if live_menu is not None:
            live_menu.sort_order = 2
        if legacy_menu is not None and legacy_menu.parent_id is not None:
            legacy_menu.sort_order = 3

        db.add(SystemSeedState(key=MONITORING_EMBEDDED_ALGORITHM_KEY))
        db.flush()
    return changed
