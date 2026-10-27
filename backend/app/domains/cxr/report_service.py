"""CXR 报告：不可变快照、草稿/最终态与导出。

规则（02 契约 H）：
- 报告只引用输入、五项评分、阈值/校准、模型版本、真实审核意见与限制；
- 无批准复核只出"待复核研究草稿"；
- 最终报告在锁定复核 revision 时生成快照，后续变化出新版本并引用 supersedes_id；
- 导出只读快照，不实时读取会变动的评分/复核；
- 不填真实姓名/病历号，不生成病史、确诊、药物或疗程。
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir
from app.db.base import utcnow
from app.db.models import CxrAnalysis, CxrReport, CxrReview
from app.domains.cxr import service as cxr_service
from app.domains.cxr.review_service import label_matrix
from app.schemas.cxr import CXR_LABEL_SCHEMA, CXR_PREPROCESS_SCHEMA, LABEL_KEYS

REPORT_SCHEMA = "cxr-report-v1"

LIMITATIONS = [
    "本报告为研究原型输出，不是临床诊断结论，必须由具备资质的人员复核。",
    "五项结果为模型观察评分（含影像征象），不等同于疾病确诊。",
    "未超过阈值不代表排除该征象，也不代表排除其他疾病。",
    "系统未做外部机构验证，未取得临床有效性证据。",
    "模型关注区域只说明模型依据，不是病灶定位真值。",
]


def _case_id(analysis: CxrAnalysis) -> str:
    return f"CASE-{analysis.id[:8].upper()}"


def _snapshot(db: Session, analysis: CxrAnalysis, review: CxrReview, *, version: int, supersedes_id: str | None) -> dict:
    from app.db.models import DatasetAsset, DatasetVersion

    source: dict = {"kind": "upload", "sha256": analysis.image_sha256}
    if analysis.asset_id is not None:
        asset = db.get(DatasetAsset, analysis.asset_id)
        if asset is not None:
            version_row = db.get(DatasetVersion, asset.dataset_version_id)
            source = {
                "kind": "controlled_asset",
                "asset_id": asset.id,
                "sha256": asset.content_hash,
                "dataset_version_id": asset.dataset_version_id,
                "manifest_sha256": version_row.manifest_sha256 if version_row else None,
                "split": asset.split,
            }
    labels = label_matrix(review)
    state = "final" if review.state == "approved" else "draft"
    return {
        "schema": REPORT_SCHEMA,
        "task": analysis.task,
        "label_schema": analysis.label_schema or CXR_LABEL_SCHEMA,
        "preprocess_schema": CXR_PREPROCESS_SCHEMA,
        "state": state,
        "version": version,
        "supersedes_id": supersedes_id,
        "title": "胸部X光五标签辅助分析报告" if state == "final" else "待复核研究草稿",
        "case": {
            "case_id": _case_id(analysis),
            "analysis_id": analysis.id,
            "batch_id": analysis.batch_id,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        },
        "input": {
            **source,
            "width": analysis.image_width,
            "height": analysis.image_height,
            "quality": analysis.quality_json or {},
        },
        "model": {
            "model_id": analysis.model_id,
            "sha256": analysis.model_sha256,
            "backend": analysis.backend,
            "calibration_status": analysis.calibration_status,
            "threshold_source": analysis.threshold_source,
            "calibration_artifact_id": analysis.calibration_artifact_id,
        },
        "labels": labels,
        "score_semantics": "五路独立评分，不要求和为 1；可为多项或零项超过阈值；不输出确诊。",
        "explanation": {
            "available": bool(analysis.explanation_artifact_id),
            "artifact_id": analysis.explanation_artifact_id,
            "definition": "模型关注区域；无定位真值，不计算定位准确度。",
        },
        "review": {
            "review_id": review.id,
            "revision": review.revision,
            "state": review.state,
            "reviewer": review.reviewer_username or None,
            "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
            "decided_at": review.decided_at.isoformat() if review.decided_at else None,
            "decided_by": review.decided_by_username,
            "reason": review.reason,
            "items": [
                {
                    "label_key": item["label_key"],
                    "opinion": item.get("opinion"),
                    "reason": item.get("reason"),
                }
                for item in (review.items_json or [])
            ],
        },
        "limitations": LIMITATIONS,
        "issued_by": review.decided_by_username if state == "final" else None,
        "issued_at": utcnow().isoformat() if state == "final" else None,
        "disclaimer": "研究用途；需人工复核；不得作为诊断、用药或治疗依据。",
    }


def _sha256(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def create_report(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> CxrReport:
    analysis = cxr_service.get_analysis(db, payload["analysis_id"], user)
    if analysis.status != "completed":
        raise ApiError(409, "cxr_analysis_not_ready", "只有已完成的分析可以出报告")
    review = db.get(CxrReview, payload["review_id"])
    if review is None or review.analysis_id != analysis.id:
        raise ApiError(404, "not_found", "复核记录不存在或不属于该分析")
    if not user.is_admin and review.owner_id != user.id:
        raise ApiError(404, "not_found", "复核记录不存在")

    latest = (
        db.query(CxrReport)
        .filter(CxrReport.analysis_id == analysis.id)
        .order_by(CxrReport.version.desc())
        .first()
    )
    if (
        latest is not None
        and latest.review_id == review.id
        and latest.review_revision == review.revision
        and latest.state == ("final" if review.state == "approved" else "draft")
    ):
        return latest  # 幂等：同复核 revision 同状态不重复签发

    if latest is not None and review.state != "approved" and latest.state == "final":
        raise ApiError(
            409,
            "cxr_report_supersede_requires_final",
            "已签发最终报告后不能回退为草稿；复核变更需重新提交并审批",
        )

    version = (latest.version + 1) if latest is not None else 1
    snapshot = _snapshot(db, analysis, review, version=version, supersedes_id=latest.id if latest else None)
    digest = _sha256(snapshot)
    report = CxrReport(
        analysis_id=analysis.id,
        review_id=review.id,
        review_revision=review.revision,
        state=snapshot["state"],
        version=version,
        supersedes_id=latest.id if latest else None,
        snapshot_json=snapshot,
        snapshot_sha256=digest,
        owner_id=user.id,
        issued_by_id=user.id if snapshot["state"] == "final" else None,
        issued_by_username=user.username if snapshot["state"] == "final" else None,
        issued_at=utcnow() if snapshot["state"] == "final" else None,
    )
    db.add(report)
    db.flush()

    target_dir = ensure_dir(Path(settings.artifact_root) / "cxr" / "reports" / analysis.id)
    json_path = target_dir / f"report-v{version}.json"
    html_path = target_dir / f"report-v{version}.html"
    atomic_write_bytes(json_path, json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    atomic_write_bytes(html_path, render_html(snapshot).encode("utf-8"))
    record = cxr_service.record_artifact(
        db, settings, owner_id=user.id, resource_type="cxr_report",
        resource_id=report.id, path=html_path, mime="text/html",
    )
    report.artifact_id = record.id
    db.commit()
    db.refresh(report)
    return report


def render_html(snapshot: dict) -> str:
    """自包含可打印 HTML；所有动态文本转义，不引用外部资源。"""
    escape = html.escape
    labels = snapshot.get("labels") or []
    rows = []
    for item in labels:
        raw = item.get("raw_score")
        calibrated = item.get("calibrated_score")
        threshold = item.get("threshold")
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('display_name') or item.get('label_key')))}</td>"
            f"<td class='num'>{'—' if raw is None else f'{float(raw):.4f}'}</td>"
            f"<td class='num'>{'—' if calibrated is None else f'{float(calibrated):.4f}'}</td>"
            f"<td class='num'>{'—' if threshold is None else f'{float(threshold):.4f}'}</td>"
            f"<td>{escape(_flag_text(item.get('flag')))}</td>"
            f"<td>{escape(_opinion_text(item.get('opinion')))}</td>"
            f"<td>{escape(str(item.get('opinion_reason') or ''))}</td>"
            "</tr>"
        )
    review = snapshot.get("review") or {}
    model = snapshot.get("model") or {}
    case = snapshot.get("case") or {}
    source = snapshot.get("input") or {}
    limitations = "".join(f"<li>{escape(str(item))}</li>" for item in snapshot.get("limitations") or [])
    banner = "最终报告" if snapshot.get("state") == "final" else "待复核研究草稿（未经人工批准，不可作为诊断依据）"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{escape(str(snapshot.get('title')))} · {escape(str(case.get('case_id')))}</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2937; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .banner {{ padding: 8px 12px; border-radius: 4px; margin: 12px 0 20px;
             background: {'#ecfdf5' if snapshot.get('state') == 'final' else '#fef3c7'};
             border: 1px solid {'#a7f3d0' if snapshot.get('state') == 'final' else '#fcd34d'}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }}
  th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
  th {{ background: #f3f4f6; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .meta td:first-child {{ width: 160px; color: #4b5563; }}
  .muted {{ color: #6b7280; font-size: 12px; }}
  @media print {{ body {{ margin: 12mm; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<h1>{escape(str(snapshot.get('title')))}</h1>
<div class="muted">报告编号 {escape(str(case.get('case_id')))} · 版本 v{escape(str(snapshot.get('version')))} · 生成时间 {escape(str(case.get('created_at') or ''))}</div>
<div class="banner">{escape(banner)}</div>
<table class="meta">
  <tr><td>任务 / 标签契约</td><td>{escape(str(snapshot.get('task')))} / {escape(str(snapshot.get('label_schema')))}</td></tr>
  <tr><td>模型版本</td><td>{escape(str(model.get('model_id')))}（sha256 {escape(str(model.get('sha256'))[:16])}…，后端 {escape(str(model.get('backend')))}）</td></tr>
  <tr><td>校准状态</td><td>{escape(str(model.get('calibration_status')))}；阈值来源 {escape(str(model.get('threshold_source')))}</td></tr>
  <tr><td>输入影像</td><td>sha256 {escape(str(source.get('sha256'))[:16])}…，尺寸 {escape(str(source.get('width')))}×{escape(str(source.get('height')))}，来源 {escape(str(source.get('kind')))}</td></tr>
  <tr><td>复核状态</td><td>{escape(str(review.get('state')))}（revision {escape(str(review.get('revision')))}，审核者 {escape(str(review.get('reviewer') or '未指定'))}）</td></tr>
  <tr><td>复核说明</td><td>{escape(str(review.get('reason') or '—'))}</td></tr>
</table>
<h2 style="font-size:16px">五项模型评分</h2>
<table>
  <thead><tr><th>观察项</th><th>原始评分</th><th>校准评分</th><th>阈值</th><th>提示</th><th>人工意见</th><th>意见理由</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<p class="muted">{escape(str(snapshot.get('score_semantics')))}</p>
<h2 style="font-size:16px">解释信息</h2>
<p class="muted">{escape(str((snapshot.get('explanation') or {}).get('definition')))}</p>
<h2 style="font-size:16px">限制与免责</h2>
<ul class="muted">{limitations}</ul>
<p class="muted">{escape(str(snapshot.get('disclaimer')))}</p>
</body>
</html>
"""


def _flag_text(flag: str | None) -> str:
    return {
        "above_threshold": "高于阈值",
        "below_threshold": "低于阈值",
        "borderline": "接近阈值",
        "not_available": "无可用评分",
    }.get(str(flag), "—")


def _opinion_text(opinion: str | None) -> str:
    return {
        "supported": "支持",
        "not_supported": "不支持",
        "indeterminate": "无法判断",
    }.get(str(opinion), "未填写")


def export(db: Session, settings: Settings, report: CxrReport, fmt: str) -> tuple[bytes, str, str]:
    """导出只读快照；不重新计算评分或复核。"""
    snapshot = report.snapshot_json or {}
    if fmt == "json":
        body = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        return body, f"{report.id}.json", "application/json"
    if fmt == "html":
        return render_html(snapshot).encode("utf-8"), f"{report.id}.html", "text/html"
    raise ApiError(422, "cxr_export_format", "format 只支持 json 或 html")
