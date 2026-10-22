"""PR-02：上传与解压安全边界。"""
from __future__ import annotations

import support


def _job(response) -> dict:
    assert response.status_code == 202, response.text
    return response.json()["data"]


def _errors(job: dict) -> list[dict]:
    return ((job.get("result") or {}).get("error") or {}).get("report", {}).get("errors", [])


def test_zip_traversal_is_rejected(client, researcher_headers, settings):
    payload = support.build_zip({"../evil.jpg": support.image_bytes((1, 2, 3))})
    job = _job(support.import_dataset(client, researcher_headers, files=[("bundle.zip", payload)]))
    assert job["status"] == "failed"
    assert job["error"]["code"] == "dataset_invalid"
    assert any(err["code"] == "invalid_path" for err in _errors(job))
    assert not (settings.data_root.parent / "evil.jpg").exists()


def test_zip_absolute_path_is_rejected(client, researcher_headers):
    payload = support.build_zip({"/abs/evil.jpg": support.image_bytes((1, 2, 3))})
    job = _job(support.import_dataset(client, researcher_headers, files=[("bundle.zip", payload)]))
    assert job["status"] == "failed"
    assert any(err["code"] == "invalid_path" for err in _errors(job))


def test_zip_symlink_is_rejected(client, researcher_headers):
    payload = support.build_zip({}, symlinks={"link.jpg": "/etc/passwd"})
    job = _job(support.import_dataset(client, researcher_headers, files=[("bundle.zip", payload)]))
    assert job["status"] == "failed"
    assert any(err["code"] == "zip_symlink" for err in _errors(job))


def test_out_of_range_label_is_reported_with_line(client, researcher_headers):
    label = support.yolo_txt([(0, 0.5, 0.5, 0.2, 0.2), (0, 0.95, 0.5, 0.5, 0.5)]).encode()
    job = _job(
        support.import_dataset(
            client,
            researcher_headers,
            files=[("a.jpg", support.image_bytes((9, 9, 9))), ("a.txt", label)],
        )
    )
    assert job["status"] == "failed"
    errors = _errors(job)
    assert errors[0]["line"] == 2
    assert errors[0]["code"] == "label_out_of_range"
    assert errors[0]["file"].endswith("a.txt")


def test_zero_area_and_unknown_class_labels_are_reported(client, researcher_headers):
    label = "0 0.5 0.5 0.0 0.2\n9 0.5 0.5 0.2 0.2\n"
    job = _job(
        support.import_dataset(
            client,
            researcher_headers,
            files=[("a.jpg", support.image_bytes((4, 5, 6))), ("a.txt", label.encode())],
        )
    )
    codes = {err["code"] for err in _errors(job)}
    assert {"label_zero_area", "label_class"} <= codes


def test_empty_label_allowed_by_default(client, researcher_headers):
    job = _job(
        support.import_dataset(
            client, researcher_headers, files=[("a.jpg", support.image_bytes((7, 7, 7)))]
        )
    )
    assert job["status"] == "completed", job
    assert job["result"]["report"]["counts"]["images_imported"] == 1


def test_empty_label_can_be_rejected(client, researcher_headers):
    job = _job(
        support.import_dataset(
            client,
            researcher_headers,
            files=[("a.jpg", support.image_bytes((8, 8, 8)))],
            form={"allow_empty_labels": "false"},
        )
    )
    assert job["status"] == "failed"
    assert _errors(job)[0]["code"] == "label_empty"


def test_invalid_image_is_reported(client, researcher_headers):
    job = _job(
        support.import_dataset(client, researcher_headers, files=[("broken.jpg", b"not-an-image")])
    )
    assert job["status"] == "failed"
    assert _errors(job)[0]["code"] == "image_decode"


def test_pixel_limit_is_enforced(client, researcher_headers, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_image_pixels", 100)
    job = _job(
        support.import_dataset(client, researcher_headers, files=[("a.jpg", support.image_bytes((1, 1, 1)))])
    )
    assert job["status"] == "failed"
    assert _errors(job)[0]["code"] == "image_pixels"
