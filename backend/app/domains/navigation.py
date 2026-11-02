"""Stable application routes with configurable presentation."""
from app.db.models import SystemMenu

ROUTES = [
    ("/cxr/dashboard", "工作台"), ("/cxr/data", "胸片数据"), ("/cxr/training", "训练与校准"),
    ("/cxr/analysis", "病例分析"), ("/cxr/review", "人工复核"), ("/cxr/reports", "分析报告"),
    ("/raw-dataset", "原始数据"), ("/dataset-annotation", "数据标注"), ("/train-dataset", "训练数据"),
    ("/model-training", "模型训练"), ("/model-hub", "模型管理"), ("/inference-service", "推理验证"),
    ("/isic/workspace", "皮肤镜研究"), ("/system/user", "账户管理"), ("/system/dept", "部门管理"),
    ("/system/role", "角色与模块授权"), ("/system/menu", "菜单管理"), ("/system/online", "在线会话"),
    ("/system/log/logininfor", "登录日志"), ("/system/log/operlog", "操作日志"), ("/system/log/faults", "系统故障日志"),
]


def list_navigation(db):
    overrides = {row.path: row for row in db.query(SystemMenu).all()}
    result = []
    for index, (path, label) in enumerate(ROUTES):
        row = overrides.get(path)
        result.append({"path": path, "label": row.label if row else label,
                       "sort_order": row.sort_order if row else index * 10,
                       "visible": row.visible if row else True})
    return sorted(result, key=lambda row: (row["sort_order"], row["path"]))
