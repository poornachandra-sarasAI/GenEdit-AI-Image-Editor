# storage.py
import os
import json
import uuid
import datetime
import numpy as np

DATA_DIR = "data"
META_FILE = os.path.join(DATA_DIR, "metadata.json")

def ensure_setup():
    """Create the data folder and a valid metadata file (also repairs an empty one)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    needs_init = not os.path.exists(META_FILE) or os.path.getsize(META_FILE) == 0
    if not needs_init:
        try:
            with open(META_FILE, "r") as f:
                needs_init = "images" not in json.load(f)
        except json.JSONDecodeError:
            needs_init = True
    if needs_init:
        _save_meta({"images": {}})

def _load_meta():
    with open(META_FILE, "r") as f:
        return json.load(f)

def _save_meta(meta):
    # Write to a temp file first so a crash mid-write can't corrupt the library
    tmp = META_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, META_FILE)

def new_image_slot(title=None):
    ensure_setup()
    meta = _load_meta()
    img_id = str(uuid.uuid4())
    meta["images"][img_id] = {
        "title": title or "",
        "caption": "",
        "embedding": [],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": []
    }
    _save_meta(meta)
    os.makedirs(os.path.join(DATA_DIR, img_id), exist_ok=True)
    return img_id

def save_version_from_bytes(image_id, image_bytes, prompt="original"):
    meta = _load_meta()
    img = meta["images"].get(image_id)
    if not img:
        raise ValueError("Image ID not found.")
    folder = os.path.join(DATA_DIR, image_id)
    os.makedirs(folder, exist_ok=True)
    v = len(img["versions"]) + 1
    filename = f"v{v}.png"
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    entry = {
        "version": v,
        "filename": filename,
        "prompt": prompt,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    img["versions"].append(entry)
    _save_meta(meta)
    return filepath, v

def get_latest_version_file(image_id):
    meta = _load_meta()
    img = meta["images"].get(image_id)
    if not img or not img["versions"]:
        return None
    latest = img["versions"][-1]
    return os.path.join(DATA_DIR, image_id, latest["filename"])

def list_images():
    meta = _load_meta()
    images = []
    for img_id, data in meta["images"].items():
        images.append({
            "id": img_id,
            "title": data.get("title", ""),
            "caption": data.get("caption", ""),
            "created_at": data.get("created_at", "")
        })
    images.sort(key=lambda x: x["created_at"], reverse=True)
    return images

def get_versions(image_id):
    meta = _load_meta()
    img = meta["images"].get(image_id)
    if not img:
        return []
    versions = []
    for v in img["versions"]:
        versions.append({
            **v,
            "path": os.path.join(DATA_DIR, image_id, v["filename"])
        })
    return sorted(versions, key=lambda x: x["version"], reverse=True)

def update_caption_and_embedding(image_id, caption, embedding_vector):
    meta = _load_meta()
    if image_id not in meta["images"]:
        raise ValueError("Image not found.")
    meta["images"][image_id]["caption"] = caption
    meta["images"][image_id]["embedding"] = list(map(float, embedding_vector))
    _save_meta(meta)

def get_image_meta(image_id):
    meta = _load_meta()
    if image_id not in meta["images"]:
        return None
    return meta["images"][image_id] | {"id": image_id}

def search_by_embedding(query_emb, top_k=10):
    """Rank images by cosine similarity between the query and caption embeddings."""
    query_emb = np.asarray(query_emb, dtype=float)
    if query_emb.size == 0:
        return []
    meta = _load_meta()
    results = []
    for img_id, data in meta["images"].items():
        emb = np.array(data.get("embedding", []), dtype=float)
        if emb.size == 0 or emb.shape != query_emb.shape:
            continue
        denom = np.linalg.norm(query_emb) * np.linalg.norm(emb)
        sim = float(np.dot(query_emb, emb) / denom) if denom > 0 else 0.0
        results.append((img_id, data.get("caption", ""), sim))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]