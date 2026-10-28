"""ISIC 测试夹具：复用 CXR 的合成数据构造器（跨目录按路径加载）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CXR_SUPPORT = Path(__file__).resolve().parents[1] / "cxr" / "cxr_support.py"
_spec = importlib.util.spec_from_file_location("cxr_support_for_isic", _CXR_SUPPORT)
cxr_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cxr_support)


@pytest.fixture()
def cxr_dataset(client, admin_headers, settings, tmp_path):
    """建一个已冻结的 CXR 数据集，用于验证跨方向拒绝。"""
    return cxr_support.ready_dataset(client, admin_headers, settings, tmp_path, patients=40)
