import io
import json
import zipfile
import support


def test_asset_image_authorized_and_export_download(client, researcher_headers):
    dataset_id, version_id = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f"/api/v1/datasets/{dataset_id}/assets", headers=researcher_headers).json()["data"]["items"][0]
    url = f"/api/v1/assets/{asset['asset_id']}/image"
    assert client.get(url).status_code == 401
    response = client.get(url, headers=researcher_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    for fmt in ["json", "yolo-txt"]:
        job = client.post("/api/v1/annotations/export", headers=researcher_headers,
            json={"dataset_id": dataset_id, "version_id": version_id, "format": fmt}).json()["data"]
        assert job["status"] == "completed", job
        result = job["result"]
        download = client.get(result["download_url"], headers=researcher_headers)
        assert download.status_code == 200
        if fmt == "json":
            assert len(download.json()["assets"]) == 3
        else:
            with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
                assert len([name for name in archive.namelist() if name.endswith(".txt")]) == 3


def test_dataset_version_mismatch_is_rejected(client, researcher_headers):
    dataset_id, _ = support.make_dataset(client, researcher_headers, count=3)
    _, other_version = support.make_dataset(client, researcher_headers, count=4)
    assert client.get(f"/api/v1/datasets/{dataset_id}/assets?version_id={other_version}", headers=researcher_headers).status_code == 404
    assert client.post(f"/api/v1/datasets/{dataset_id}/versions", headers=researcher_headers,
        json={"base_version_id": other_version}).status_code == 404
    assert client.post("/api/v1/annotations/export", headers=researcher_headers,
        json={"dataset_id": dataset_id, "version_id": other_version}).status_code == 404
