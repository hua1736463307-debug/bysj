"""One-time migration that aligns persisted menus with the current navigation layout."""

MENU_STRUCTURE_KEY = "menu_structure_20260921_v1"


def _one(db, Menu, path, menu_type, label):
    rows = (
        db.query(Menu)
        .filter(Menu.path == path, Menu.menu_type == menu_type)
        .order_by(Menu.id.asc())
        .all()
    )
    if len(rows) > 1:
        raise RuntimeError(f"菜单结构迁移冲突：{label} 存在多条记录")
    return rows[0] if rows else None


def _ensure_menu(
    db,
    Menu,
    *,
    path,
    menu_type,
    name,
    icon,
    sort_order,
    status,
    parent_id=None,
):
    from monitoring_menu_migration import _create_menu

    item = _one(db, Menu, path, menu_type, name)
    created = item is None
    if created:
        item = _create_menu(
            db,
            Menu,
            parent_id=parent_id,
            name=name,
            path=path,
            icon=icon,
            menu_type=menu_type,
            sort_order=sort_order,
            status=status,
            visible="visible",
        )
        db.add(item)
        db.flush()
    else:
        item.parent_id = parent_id
        item.name = name
        item.icon = icon
        item.sort_order = sort_order
        item.status = status
        item.visible = "visible"
    return item, created


def _copy_grants(db, RoleMenu, source, target):
    if source is None or target is None:
        return
    from monitoring_menu_migration import _create_role_menu

    role_ids = [
        row.role_id
        for row in db.query(RoleMenu).filter(RoleMenu.menu_id == source.id).all()
    ]
    for role_id in role_ids:
        exists = (
            db.query(RoleMenu)
            .filter(RoleMenu.role_id == role_id, RoleMenu.menu_id == target.id)
            .first()
        )
        if not exists:
            db.add(
                _create_role_menu(
                    db,
                    RoleMenu,
                    role_id=role_id,
                    menu_id=target.id,
                )
            )


def _remove_obsolete_menu(db, Menu, RoleMenu, path, replacement=None):
    rows = (
        db.query(Menu)
        .filter(Menu.path == path, Menu.menu_type == "menu")
        .order_by(Menu.id.asc())
        .all()
    )
    changed = False
    for item in rows:
        _copy_grants(db, RoleMenu, item, replacement)
        db.query(RoleMenu).filter(RoleMenu.menu_id == item.id).delete(
            synchronize_session=False
        )
        db.delete(item)
        changed = True
    return changed


def migrate_current_menu_structure(db, Menu, RoleMenu, SystemSeedState):
    """Make existing databases match the current menu-management layout once."""
    if (
        db.query(SystemSeedState)
        .filter(SystemSeedState.key == MENU_STRUCTURE_KEY)
        .first()
    ):
        return False

    changed = False
    with db.begin_nested():
        monitoring, created = _ensure_menu(
            db,
            Menu,
            path="/monitoring",
            menu_type="directory",
            name="监控管理",
            icon="View",
            sort_order=7,
            status="active",
            parent_id=None,
        )
        changed = changed or created

        config, created = _ensure_menu(
            db,
            Menu,
            path="/monitoring/config",
            menu_type="menu",
            name="监控配置",
            icon="Setting",
            sort_order=1,
            status="active",
            parent_id=monitoring.id,
        )
        changed = changed or created

        _, created = _ensure_menu(
            db,
            Menu,
            path="/monitoring/live",
            menu_type="menu",
            name="实时监控",
            icon="Monitor",
            sort_order=2,
            status="active",
            parent_id=monitoring.id,
        )
        changed = changed or created

        _, created = _ensure_menu(
            db,
            Menu,
            path="/inference-service",
            menu_type="menu",
            name="推理服务（旧版）",
            icon="View",
            sort_order=3,
            status="disabled",
            parent_id=monitoring.id,
        )
        changed = changed or created

        alarm_center, created = _ensure_menu(
            db,
            Menu,
            path="/alarm-center",
            menu_type="menu",
            name="告警中心",
            icon="Bell",
            sort_order=4,
            status="active",
            parent_id=monitoring.id,
        )
        changed = changed or created

        algorithm_config, created = _ensure_menu(
            db,
            Menu,
            path="/system/algorithms",
            menu_type="menu",
            name="算法配置",
            icon="DataAnalysis",
            sort_order=6,
            status="active",
            parent_id=monitoring.id,
        )
        changed = changed or created

        behavior_group, created = _ensure_menu(
            db,
            Menu,
            path="/behavior",
            menu_type="directory",
            name="行为分析",
            icon="VideoCamera",
            sort_order=8,
            status="active",
            parent_id=None,
        )
        changed = changed or created

        _, created = _ensure_menu(
            db,
            Menu,
            path="/behavior",
            menu_type="menu",
            name="行为分析",
            icon="VideoCamera",
            sort_order=1,
            status="disabled",
            parent_id=behavior_group.id,
        )
        changed = changed or created

        algorithm_training, created = _ensure_menu(
            db,
            Menu,
            path="/algorithm-training",
            menu_type="menu",
            name="算法优化",
            icon="DataAnalysis",
            sort_order=3,
            status="active",
            parent_id=behavior_group.id,
        )
        changed = changed or created

        platform, created = _ensure_menu(
            db,
            Menu,
            path="/system",
            menu_type="directory",
            name="平台管理",
            icon="Setting",
            sort_order=9,
            status="active",
            parent_id=None,
        )
        changed = changed or created

        changed = (
            _remove_obsolete_menu(
                db,
                Menu,
                RoleMenu,
                "/behavior-review",
                replacement=alarm_center,
            )
            or changed
        )
        changed = (
            _remove_obsolete_menu(
                db,
                Menu,
                RoleMenu,
                "/action-training",
                replacement=algorithm_training,
            )
            or changed
        )
        changed = (
            _remove_obsolete_menu(
                db,
                Menu,
                RoleMenu,
                "/monitoring/algorithms",
                replacement=config,
            )
            or changed
        )

        # Keep the variable referenced so future edits do not accidentally
        # treat the moved algorithm menu as unused migration state.
        _ = algorithm_config, platform

        db.add(SystemSeedState(key=MENU_STRUCTURE_KEY))
        db.flush()

    return changed
