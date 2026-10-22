"""医学测试夹具：程序生成图像与标签，不使用真实病人数据。"""
from __future__ import annotations

import io
import zipfile

from PIL import Image, ImageDraw


def image_bytes(color: tuple[int, int, int], size=(96, 96), fmt: str = "JPEG") -> bytes:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255, 255, 255), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def yolo_txt(annotations: list[tuple[int, float, float, float, float]]) -> str:
    return "\n".join(
        f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for class_id, cx, cy, w, h in annotations
    )


def build_zip(entries: dict[str, bytes], symlinks: dict[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in entries.items():
            bundle.writestr(name, payload)
        for name, target in (symlinks or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o120777 & 0xFFFF) << 16
            bundle.writestr(info, target)
    return buffer.getvalue()


def import_dataset(
    client,
    headers,
    *,
    files: list[tuple[str, bytes]],
    form: dict | None = None,
    field: str = "files",
):
    payload = {"name": "测试数据集", "class_map": '{"0":"RBC","1":"WBC","2":"Platelets"}'}
    payload.update(form or {})
    return client.post(
        "/api/v1/datasets/import",
        headers=headers,
        files=[(field, (name, data, "application/octet-stream")) for name, data in files],
        data=payload,
    )


def freeze(client, headers, dataset_id: str, *, version_id: str | None = None, seed: int = 42):
    body = {"split_seed": seed}
    if version_id:
        body["base_version_id"] = version_id
    return client.post(f"/api/v1/datasets/{dataset_id}/versions", headers=headers, json=body)


def make_dataset(client, headers, *, count: int = 12, classes=(0, 1, 2), seed_prefix: str = "img"):
    files: list[tuple[str, bytes]] = []
    for index in range(count):
        color = ((index * 37) % 255, (index * 61) % 255, (index * 91) % 255)
        name = f"{seed_prefix}-{index:03d}.jpg"
        files.append((name, image_bytes(color)))
        labels = [(class_id, 0.5, 0.5, 0.3, 0.3) for class_id in classes]
        files.append((f"{seed_prefix}-{index:03d}.txt", yolo_txt(labels).encode()))
    response = import_dataset(client, headers, files=files)
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    assert data["status"] == "completed", data
    return data["result"]["dataset_id"], data["result"]["version_id"]
