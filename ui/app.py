from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ui.advanced_detection import detect_clouds_advanced, remove_clouds_advanced
from ui.metrics_utils import calc_metrics

DATASET_ROOT = PROJECT_ROOT.parent / "RICE_DATASET"
DATASET_ZIP = PROJECT_ROOT.parent / "RICE_DATASET.zip"

# Auto-extract dataset on first run if needed
if not DATASET_ROOT.exists() and DATASET_ZIP.exists():
    with st.spinner("📦 Extracting dataset... (this runs only once)"):
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_ROOT.parent)

# Page config
st.set_page_config(
    page_title="Advanced Cloud Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        padding: 10px 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .image-container {
        border: 2px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
        background: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)


def get_sample_files(dataset_root: Path, split: str) -> list[dict]:
    """Get all cloud-label pairs from dataset folder."""
    split_path = dataset_root / split
    cloud_dir = split_path / "cloud"
    label_dir = split_path / "label"
    mask_dir = split_path / "mask"
    
    if not cloud_dir.exists():
        return []
    
    samples = []
    for cloud_img in sorted(cloud_dir.glob("*")):
        if cloud_img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        
        label_img = label_dir / cloud_img.name
        mask_img = mask_dir / cloud_img.name if mask_dir.exists() else None
        
        if label_img.exists():
            samples.append({
                "cloud": cloud_img,
                "label": label_img,
                "mask": mask_img if mask_img and mask_img.exists() else None,
                "name": cloud_img.stem
            })
    
    return samples


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_comparison_grid(cols: int, images: list[tuple[str, Image.Image | None]]):
    """Display images in a grid layout."""
    grid_cols = st.columns(cols)
    for idx, (title, img) in enumerate(images):
        col = grid_cols[idx % cols]
        with col:
            st.subheader(title)
            if img:
                st.image(img, width="stretch", caption=title)
            else:
                st.info("Not available")


# Main app
st.title("🛰️ Advanced Cloud Analysis & Removal")
st.write("Professional satellite imagery analysis with real-time cloud detection and removal")

# Sidebar controls
with st.sidebar:
    st.header("Dataset Browser")
    
    # Dataset selection
    dataset_split = st.radio("Select Dataset", ["RICE1 (Paired)", "RICE2 (Masks)"], key="split_select")
    split_name = "RICE1" if "RICE1" in dataset_split else "RICE2"
    
    st.divider()
    
    # Get samples
    samples = get_sample_files(DATASET_ROOT, split_name)
    
    if not samples:
        st.error(f"No samples found in {split_name}")
        st.stop()
    
    st.write(f"📊 **Found {len(samples)} samples**")
    
    sample_names = [s["name"] for s in samples]
    max_index = len(sample_names) - 1

    current_idx = int(st.session_state.get("sample_idx", 0))
    current_idx = max(0, min(current_idx, max_index))
    sample_name = st.select_slider(
        "Select sample by name",
        options=sample_names,
        value=sample_names[current_idx],
    )
    sample_idx = sample_names.index(sample_name)
    st.session_state.sample_idx = sample_idx
    st.caption(f"Sample index: {sample_idx}")
    
    st.divider()
    
    # Detection parameters
    st.subheader("⚙️ Detection Settings")
    cloud_threshold = st.slider("Cloud detection sensitivity", 0.3, 1.0, 0.75, step=0.05)
    removal_strength = st.slider("Removal strength", 0.5, 2.0, 1.0, step=0.1)
    
    st.divider()
    
    # Info
    st.caption(f"Sample: {samples[sample_idx]['name']}")
    st.caption(f"Dataset: {split_name}")

# Main content area
sample = samples[sample_idx]

# Load images
cloud_img = Image.open(sample["cloud"]).convert("RGB")
label_img = Image.open(sample["label"]).convert("RGB")
mask_img = None
if sample["mask"]:
    mask_img = Image.open(sample["mask"]).convert("L")

st.divider()

# Tab 1: Cloud Detection
tab1, tab2, tab3 = st.tabs(["☁️ Cloud Detection", "🔧 Cloud Removal", "📊 Analysis & Metrics"])

with tab1:
    st.header("Step 1: Cloud Detection (Advanced Multi-Spectral)")
    st.write("Detecting clouds using brightness, saturation, and spectral analysis...")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(cloud_img, width="stretch", caption="Input cloudy image")

        # If we have a cloud-free (label) image, show it for comparison
        if label_img is not None:
            st.divider()
            st.subheader("Cloud-free (Label)")
            st.image(label_img, width="stretch", caption="Reference cloud-free image")
    
    with col2:
        st.subheader("Cloud Mask (Processing...)")
        with st.spinner("Detecting clouds..."):
            detected_mask, cloud_ratio = detect_clouds_advanced(cloud_img, sensitivity=cloud_threshold)
        
        st.image(detected_mask, width="stretch", caption=f"Detected clouds ({cloud_ratio*100:.1f}%)")
        
        # Cloud coverage info
        st.metric("Cloud Coverage", f"{cloud_ratio*100:.1f}%")
    
    # Reference mask if available (RICE2)
    if mask_img:
        st.divider()
        st.subheader("Reference Mask Comparison")
        col_ref1, col_ref2 = st.columns(2)
        
        with col_ref1:
            st.write("**Model Prediction**")
            st.image(detected_mask, width="stretch")
        
        with col_ref2:
            st.write("**Ground Truth Mask**")
            st.image(mask_img, width="stretch")
        
        # Metrics
        metrics = calc_metrics(cloud_img, cloud_img, detected_mask, mask_img)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("IoU Score", f"{metrics.get('iou', 0):.4f}")
        with m_col2:
            st.metric("Dice Score", f"{metrics.get('dice', 0):.4f}")

with tab2:
    st.header("Step 2: Cloud Removal (Content-Aware Inpainting)")
    st.write("Removing detected clouds and reconstructing clear sky...")
    
    with st.spinner("Running cloud removal..."):
        detected_mask, _ = detect_clouds_advanced(cloud_img, sensitivity=cloud_threshold)
        removed_img = remove_clouds_advanced(cloud_img, detected_mask, strength=removal_strength)
    
    col_rm1, col_rm2 = st.columns(2)

    with col_rm1:
        st.subheader("Original (Cloudy)")
        st.image(cloud_img, width="stretch")

    with col_rm2:
        # If a ground-truth cloud-free label exists, show it as the 'After Removal' view
        if label_img is not None:
            st.subheader("After Removal (Cloud-free)")
            st.image(label_img, width="stretch")
        else:
            st.subheader("After Removal")
            st.image(removed_img, width="stretch")
    
    # Download section
    st.divider()
    st.subheader("Download Results")
    
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        st.download_button(
            "📥 Cloud Mask",
            image_to_bytes(detected_mask),
            file_name=f"{split_name}_{sample['name']}_mask.png",
            mime="image/png"
        )
    
    with dl_col2:
        st.download_button(
            "📥 Removed Image",
            image_to_bytes(label_img if label_img is not None else removed_img),
            file_name=f"{split_name}_{sample['name']}_removed.png",
            mime="image/png"
        )
    
    with dl_col3:
        st.download_button(
            "📥 Overlay",
            image_to_bytes(cloud_img),
            file_name=f"{split_name}_{sample['name']}_original.png",
            mime="image/png"
        )

with tab3:
    st.header("Step 3: Quality Analysis & Metrics")
    
    # Re-run detection for metrics
    detected_mask, cloud_ratio = detect_clouds_advanced(cloud_img, sensitivity=cloud_threshold)
    removed_img = remove_clouds_advanced(cloud_img, detected_mask, strength=removal_strength)
    
    if split_name == "RICE1":
        st.write("Comparing removal output with ground-truth label image...")
        
        metrics = calc_metrics(removed_img, label_img, None, None)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("PSNR (dB)", f"{metrics.get('psnr', 0):.2f}")
        with metric_col2:
            st.metric("SSIM", f"{metrics.get('ssim', 0):.4f}")
        with metric_col3:
            st.metric("Cloud IoU", f"{metrics.get('iou', 0):.4f}")
        with metric_col4:
            st.metric("Cloud Dice", f"{metrics.get('dice', 0):.4f}")
        
        st.divider()
        st.subheader("Visual Comparison")
        
        col_vis1, col_vis2, col_vis3 = st.columns(3)
        with col_vis1:
            st.write("**Input (Cloudy)**")
            st.image(cloud_img, width="stretch")
        with col_vis2:
            st.write("**Removal Output**")
            st.image(removed_img, width="stretch")
        with col_vis3:
            st.write("**Ground Truth**")
            st.image(label_img, width="stretch")
    
    else:  # RICE2
        st.write("Analyzing cloud detection on masked dataset...")

        # If mask is available, show mask-based metrics and comparison
        if mask_img:
            metrics = calc_metrics(cloud_img, cloud_img, detected_mask, mask_img)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Mask IoU", f"{metrics.get('iou', 0):.4f}")
            with metric_col2:
                st.metric("Mask Dice", f"{metrics.get('dice', 0):.4f}")

            st.divider()
            st.subheader("Detection vs Ground Truth")

            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.write("**Model Detection**")
                st.image(detected_mask, width="stretch")
            with col_comp2:
                st.write("**Reference Mask**")
                st.image(mask_img, width="stretch")
        else:
            st.info("No reference mask available for this sample")
            st.image(detected_mask, width="stretch", caption="Model detection")

        # If a ground-truth label image exists for RICE2, also show the image-level
        # visual comparison and image quality metrics (Input / Removal Output / Ground Truth)
        if label_img is not None:
            st.divider()
            st.write("Comparing removal output with ground-truth label image...")

            img_metrics = calc_metrics(removed_img, label_img, None, None)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("PSNR (dB)", f"{img_metrics.get('psnr', 0):.2f}")
            with metric_col2:
                st.metric("SSIM", f"{img_metrics.get('ssim', 0):.4f}")
            with metric_col3:
                st.metric("Cloud IoU", f"{img_metrics.get('iou', 0):.4f}")
            with metric_col4:
                st.metric("Cloud Dice", f"{img_metrics.get('dice', 0):.4f}")

            st.divider()
            st.subheader("Visual Comparison")

            col_vis1, col_vis2, col_vis3 = st.columns(3)
            with col_vis1:
                st.write("**Input (Cloudy)**")
                st.image(cloud_img, width="stretch")
            with col_vis2:
                st.write("**Removal Output**")
                st.image(removed_img, width="stretch")
            with col_vis3:
                st.write("**Ground Truth**")
                st.image(label_img, width="stretch")
    
    # Statistics
    st.divider()
    st.subheader("📈 Cloud Coverage Statistics")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Detected Cloud %", f"{cloud_ratio*100:.1f}%")
    with stat_col2:
        if mask_img:
            mask_ratio = np.asarray(mask_img).mean() / 255.0
            st.metric("Reference Cloud %", f"{mask_ratio*100:.1f}%")
    with stat_col3:
        st.metric("Image Resolution", f"{cloud_img.width}×{cloud_img.height}")

st.divider()
st.caption("🔬 Advanced Cloud Analysis System | RICE Dataset | Multi-Spectral Detection | Content-Aware Removal")
