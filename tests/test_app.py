"""Offline tests: model calls are mocked, so no API keys are needed.

Run with:  python -m pytest -q
"""
import os
import sys
from io import BytesIO
from unittest import mock

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ai_utils  # noqa: E402
import storage  # noqa: E402


def png_bytes(color=(255, 0, 0), fmt="PNG"):
    buf = BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(storage, "META_FILE", str(tmp_path / "metadata.json"))
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def test_ensure_setup_repairs_empty_metadata(tmp_data):
    (tmp_data / "metadata.json").write_text("")
    storage.ensure_setup()
    assert storage.list_images() == []


def test_versions_and_search_ranking():
    storage.ensure_setup()
    a = storage.new_image_slot("a")
    b = storage.new_image_slot("b")
    storage.save_version_from_bytes(a, png_bytes())
    storage.update_caption_and_embedding(a, "cat", [1.0, 0.0])
    storage.update_caption_and_embedding(b, "car", [0.0, 1.0])
    assert storage.search_by_embedding([0.9, 0.1])[0][0] == a
    assert storage.search_by_embedding([]) == []
    assert storage.search_by_embedding([1.0, 0.0, 0.0]) == []  # dimension mismatch is skipped
    storage.save_version_from_bytes(a, png_bytes((0, 255, 0)), prompt="make green")
    assert [v["version"] for v in storage.get_versions(a)] == [2, 1]


def test_jpeg_is_normalised_to_png():
    out = ai_utils.to_png_bytes(png_bytes(fmt="JPEG"))
    assert Image.open(BytesIO(out)).format == "PNG"


def test_edit_raises_when_model_returns_no_image(tmp_data):
    path = tmp_data / "in.png"
    path.write_bytes(png_bytes())
    text_part = mock.Mock(inline_data=None, text="I can't edit that.")
    response = mock.Mock(candidates=[mock.Mock(content=mock.Mock(parts=[text_part]))])
    client = mock.Mock()
    client.models.generate_content.return_value = response
    with mock.patch.object(ai_utils, "_gemini", return_value=client):
        with pytest.raises(ai_utils.ImageEditError, match="can't edit"):
            ai_utils.edit_image(str(path), "remove background")


def test_eval_metrics():
    import eval_search

    storage.ensure_setup()
    a = storage.new_image_slot("cat-photo")
    b = storage.new_image_slot("car-photo")
    storage.update_caption_and_embedding(a, "cat", [1.0, 0.0])
    storage.update_caption_and_embedding(b, "car", [0.0, 1.0])
    vectors = {"a cat": [1.0, 0.1], "a car": [1.0, 0.2]}  # second query ranks the car 2nd
    queries = [{"query": "a cat", "relevant": ["cat-photo"]}, {"query": "a car", "relevant": ["car-photo"]}]
    metrics, misses = eval_search.evaluate(queries, [1, 3], embed=vectors.get)
    assert metrics["recall@1"] == 0.5 and metrics["recall@3"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert misses == [("a car", 2)]


def test_streamlit_open_navigates_to_detail(tmp_data):
    from streamlit.testing.v1 import AppTest

    storage.ensure_setup()
    img = storage.new_image_slot("sunset")
    storage.save_version_from_bytes(img, png_bytes())
    storage.update_caption_and_embedding(img, "a sunset", [1.0])

    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.button(key=f"lib_{img}").click().run()
    assert at.session_state["page"] == "Image Detail"
    assert any("sunset" in s.value for s in at.subheader)
    assert not at.exception
