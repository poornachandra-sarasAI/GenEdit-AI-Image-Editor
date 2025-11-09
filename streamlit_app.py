# streamlit_app.py
import streamlit as st
from storage import ensure_setup, new_image_slot, save_version_from_bytes, list_images, get_latest_version_file, get_versions, update_caption_and_embedding, get_image_meta, search_by_embedding
import storage
from ai_utils import embed_text
from ai_utils import generate_caption_gemini as generate_caption
from ai_utils import edit_image_gemini as edit_image
from PIL import Image
import io
import os
import base64

st.title("GenEdit — AI Image Editor (MVP)")
ensure_setup()

# Sidebar navigation
page = st.sidebar.selectbox("Page", ["Library", "Add Image", "Image Detail (by ID)"])
st.sidebar.markdown("---")
st.sidebar.caption("GenEdit MVP — natural-language image edits")

def show_library():
    st.header("Image Library")
    q = st.text_input("Search (natural language)", placeholder="e.g., beach, product photo, person with dog")
    if st.button("Search") and q.strip():
        with st.spinner("Embedding query and searching..."):
            q_emb = embed_text(q)
            results = search_by_embedding(q_emb, top_k=30)
        if not results:
            st.info("No matches found. Try other terms or add images.")
            return
        cols = st.columns(4)
        for idx, (img_id, caption, score) in enumerate(results):
            col = cols[idx % 4]
            latest = get_latest_version_file(img_id)
            if latest and os.path.exists(latest):
                col.image(latest, use_container_width=True, caption=f"{caption[:60]} — score {score:.2f}")
                if col.button("Open", key=f"open_{img_id}"):
                    st.session_state["open_image_id"] = img_id
                    st.rerun()
    else:
        images = list_images()
        if not images:
            st.info("No images yet. Upload on 'Add Image' page.")
            return
        cols = st.columns(4)
        for i,img in enumerate(images):
            latest = get_latest_version_file(img["id"])
            col = cols[i%4]
            if latest and os.path.exists(latest):
                col.image(latest, use_container_width=True, caption=img["caption"] or img["title"] or img["id"])
                if col.button("Open", key=f"open_{img['id']}"):
                    st.session_state["open_image_id"] = img['id']
                    st.rerun()

def show_add_image():
    st.header("Add Image")
    uploaded = st.file_uploader("Choose image file", type=["png","jpg","jpeg"])
    title = st.text_input("Title (optional)")
    if st.button("Upload") and uploaded:
        image_bytes = uploaded.getvalue()
        image_id = new_image_slot(title=title)
        filepath, v = save_version_from_bytes(image_id, image_bytes, prompt="original upload")
        st.success(f"Saved image {image_id} as version {v}")
        # generate caption & embed
        with st.spinner("Generating caption and indexing..."):
            try:
                caption = generate_caption(filepath)
                emb = embed_text(caption)
                update_caption_and_embedding(image_id, caption, emb)
                st.write("Caption:", caption)
            except Exception as e:
                st.error("Caption or embedding failed: " + str(e))
        if st.button("Open image detail"):
            st.session_state["open_image_id"] = image_id
            st.rerun()

def show_image_detail(image_id=None):
    st.header("Image Detail & Edit")
    if not image_id:
        image_id = st.text_input("Enter image ID to open (or set from Library)", value=st.session_state.get("open_image_id",""))
        if not image_id:
            st.info("Select an image from the Library or enter an ID.")
            return
    meta = get_image_meta(image_id)
    if not meta:
        st.error("Image not found")
        return
    st.subheader(meta.get("title") or image_id)
    latest = get_latest_version_file(image_id)
    if latest:
        st.image(latest, use_container_width=True)
    else:
        st.warning("No image files found for this ID")
    st.write("Caption:", meta.get("caption","(none)"))
    st.write("Created:", meta.get("created_at"))

    st.markdown("---")
    st.subheader("Edit image (presets or natural language)")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    if preset_col1.button("Remove background"):
        st.session_state["edit_prompt"] = "Remove the background and make it transparent, keep the subject intact."
    if preset_col2.button("Add tree in background"):
        st.session_state["edit_prompt"] = "Add a medium-sized oak tree in the background on the right side."
    if preset_col3.button("Make vibrant"):
        st.session_state["edit_prompt"] = "Increase color saturation and contrast subtly to make the image more vibrant."

    # free text
    prompt = st.text_area("Custom edit prompt", value=st.session_state.get("edit_prompt",""), height=120)
    if st.button("Apply edit"):
        if not latest:
            st.error("No base image found")
        else:
            with st.spinner("Calling image edit API..."):
                try:
                    edited_bytes = edit_image(latest, prompt)
                    new_path, version = save_version_from_bytes(image_id, edited_bytes, prompt=prompt)
                    st.success(f"Saved new version {version}")
                    # update caption & embedding for the new version
                    caption = generate_caption(new_path)
                    emb = embed_text(caption)
                    update_caption_and_embedding(image_id, caption, emb)
                    st.image(new_path, caption=f"v{version}", use_container_width=True)
                except Exception as e:
                    st.error("Edit failed: " + str(e))

    st.markdown("---")
    st.subheader("Version history")
    versions = get_versions(image_id)
    for v in versions:
        cols = st.columns([1,4,2])
        cols[0].write(f"v{v['version']}")
        try:
            cols[1].image(v['path'], use_container_width=True)
        except Exception:
            cols[1].text(v['filename'])
        cols[2].write(v['prompt'] or "")
        if cols[2].button("Restore", key=f"restore_{image_id}_{v['version']}"):
            # restoring: copy version file to become new version
            with open(v['path'],"rb") as f:
                bytestr = f.read()
            save_version_from_bytes(image_id, bytestr, prompt=f"restored v{v['version']}")
            st.rerun()

# Main page routing
if page == "Library":
    show_library()
elif page == "Add Image":
    show_add_image()
else:
    show_image_detail()