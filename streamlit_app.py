# streamlit_app.py
import os

import streamlit as st

from ai_utils import ImageEditError, edit_image, embed_text, generate_caption, to_png_bytes
from storage import (
    ensure_setup,
    get_image_meta,
    get_latest_version_file,
    get_versions,
    list_images,
    new_image_slot,
    save_version_from_bytes,
    search_by_embedding,
    update_caption_and_embedding,
)

PAGES = ["Library", "Add Image", "Image Detail"]
PRESETS = {
    "Remove background": "Remove the background and replace it with plain white, keeping the subject intact.",
    "Add tree in background": "Add a medium-sized oak tree in the background on the right side.",
    "Make vibrant": "Increase color saturation and contrast subtly to make the image more vibrant.",
}

st.set_page_config(page_title="GenEdit — AI Image Editor", layout="wide")
st.title("GenEdit — AI Image Editor")
ensure_setup()

st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("open_image_id", "")


# ---------- navigation helpers (callbacks run before the next rerun) ----------

def open_image(image_id: str):
    st.session_state["open_image_id"] = image_id
    st.session_state["page"] = "Image Detail"


def set_prompt(text: str):
    st.session_state["edit_prompt"] = text


def index_image(image_id: str, path: str) -> str:
    """Caption an image and store its embedding so it is searchable."""
    caption = generate_caption(path)
    update_caption_and_embedding(image_id, caption, embed_text(caption))
    return caption


def image_grid(items, key_prefix: str):
    cols = st.columns(4)
    for i, (img_id, label) in enumerate(items):
        latest = get_latest_version_file(img_id)
        if latest and os.path.exists(latest):
            col = cols[i % 4]
            col.image(latest, use_container_width=True, caption=label)
            col.button("Open", key=f"{key_prefix}_{img_id}", on_click=open_image, args=(img_id,))


# ---------- pages ----------

def show_library():
    st.header("Image Library")
    with st.form("search"):
        q = st.text_input("Search (natural language)", placeholder="e.g., beach, product photo, person with dog")
        searched = st.form_submit_button("Search")

    if searched and q.strip():
        try:
            with st.spinner("Embedding query and searching..."):
                results = search_by_embedding(embed_text(q), top_k=30)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return
        if not results:
            st.info("No matches found. Try other terms or add images.")
            return
        image_grid([(i, f"{c[:60]} — score {s:.2f}") for i, c, s in results], "search")
        return

    images = list_images()
    if not images:
        st.info("No images yet. Upload one on the 'Add Image' page.")
        return
    image_grid([(img["id"], img["caption"] or img["title"] or img["id"]) for img in images], "lib")


def show_add_image():
    st.header("Add Image")
    uploaded = st.file_uploader("Choose image file", type=["png", "jpg", "jpeg"])
    title = st.text_input("Title (optional)")
    if st.button("Upload", disabled=uploaded is None):
        try:
            png = to_png_bytes(uploaded.getvalue())
        except Exception as e:
            st.error(f"Could not read that image: {e}")
            return
        image_id = new_image_slot(title=title)
        filepath, v = save_version_from_bytes(image_id, png, prompt="original upload")
        st.session_state["last_uploaded_id"] = image_id
        st.success(f"Saved image as version {v}")
        with st.spinner("Generating caption and indexing..."):
            try:
                st.write("Caption:", index_image(image_id, filepath))
            except Exception as e:
                st.warning(f"Saved, but captioning/indexing failed (image won't appear in search): {e}")

    last = st.session_state.get("last_uploaded_id")
    if last:
        st.button("Open last uploaded image", on_click=open_image, args=(last,))


def show_image_detail():
    st.header("Image Detail & Edit")
    image_id = st.text_input("Image ID", key="open_image_id", help="Set automatically when you click Open")
    if not image_id:
        st.info("Open an image from the Library, or paste an image ID.")
        return
    meta = get_image_meta(image_id.strip())
    if not meta:
        st.error("Image not found")
        return
    image_id = meta["id"]

    st.subheader(meta.get("title") or "Untitled image")
    latest = get_latest_version_file(image_id)
    if latest:
        st.image(latest, use_container_width=True)
    else:
        st.warning("No image files found for this ID")
    st.write("Caption:", meta.get("caption") or "(none)")
    st.caption(f"Created: {meta.get('created_at')}")

    st.markdown("---")
    st.subheader("Edit image (presets or natural language)")
    for col, (label, text) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        col.button(label, on_click=set_prompt, args=(text,))

    prompt = st.text_area("Custom edit prompt", key="edit_prompt", height=120)
    if st.button("Apply edit", disabled=not latest or not prompt.strip()):
        try:
            with st.spinner("Editing image..."):
                edited = edit_image(latest, prompt)
            new_path, version = save_version_from_bytes(image_id, edited, prompt=prompt)
            st.success(f"Saved new version v{version}")
            st.image(new_path, caption=f"v{version}", use_container_width=True)
        except ImageEditError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Edit failed: {e}")
            return
        try:
            index_image(image_id, new_path)
        except Exception as e:
            st.warning(f"Edit saved, but re-indexing for search failed: {e}")

    st.markdown("---")
    st.subheader("Version history")
    for v in get_versions(image_id):
        cols = st.columns([1, 4, 2])
        cols[0].write(f"v{v['version']}")
        if os.path.exists(v["path"]):
            cols[1].image(v["path"], use_container_width=True)
        else:
            cols[1].text(f"{v['filename']} (missing)")
        cols[2].write(v["prompt"] or "")
        if cols[2].button("Restore", key=f"restore_{image_id}_{v['version']}"):
            with open(v["path"], "rb") as f:
                save_version_from_bytes(image_id, f.read(), prompt=f"restored v{v['version']}")
            st.rerun()


# ---------- routing ----------
st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption("GenEdit — natural-language image edits")

{"Library": show_library, "Add Image": show_add_image, "Image Detail": show_image_detail}[st.session_state["page"]]()
