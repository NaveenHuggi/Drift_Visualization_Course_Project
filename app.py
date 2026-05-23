"""
Melanoma CAM Visualizer — Professional Streamlit Dashboard
A deep learning interpretability tool for melanoma classification using ResNet-18
with 8 state-of-the-art Class Activation Mapping (CAM) algorithms.
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ── CAM imports ──────────────────────────────────────────────────────────────
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Melanoma CAM Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark, premium aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root variables ── */
  :root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2230;
    --bg-hover:      #21293a;
    --accent:        #58a6ff;
    --accent-2:      #3fb950;
    --accent-warn:   #f85149;
    --accent-orange: #d29922;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --border:        #30363d;
    --border-accent: #388bfd66;
    --glow:          #58a6ff33;
    --radius:        12px;
    --shadow:        0 8px 32px rgba(0,0,0,.45);
  }

  /* ── Global resets ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

  /* ── Page header banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #1a2233 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow), 0 0 40px var(--glow);
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at top left, #58a6ff18, transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .4rem;
    letter-spacing: -.5px;
  }
  .hero-sub {
    font-size: .95rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
  }

  /* ── Cards ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: border-color .2s, box-shadow .2s;
  }
  .card:hover { border-color: var(--border-accent); box-shadow: 0 8px 32px var(--glow); }

  /* ── Section labels ── */
  .section-label {
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: .8rem;
  }

  /* ── Result badge — BENIGN ── */
  .badge-benign {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: #1a3a2a;
    border: 1px solid #3fb950;
    color: #3fb950;
    border-radius: 999px;
    padding: .4rem 1.2rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: .05em;
  }

  /* ── Result badge — MALIGNANT ── */
  .badge-malignant {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: #3a1a1a;
    border: 1px solid #f85149;
    color: #f85149;
    border-radius: 999px;
    padding: .4rem 1.2rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: .05em;
  }

  /* ── Confidence bar ── */
  .conf-bar-container {
    background: #21293a;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: .5rem;
  }
  .conf-bar-fill-benign    { background: linear-gradient(90deg, #2ea043, #3fb950); height: 100%; border-radius: 999px; }
  .conf-bar-fill-malignant { background: linear-gradient(90deg, #da3633, #f85149); height: 100%; border-radius: 999px; }

  /* ── Metric tile ── */
  .metric-tile {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label {
    font-size: .75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .2rem;
  }

  /* ── CAM chip selector ── */
  .cam-chip {
    display: inline-block;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .35rem .9rem;
    font-size: .82rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all .18s;
  }
  .cam-chip:hover, .cam-chip.active {
    background: var(--bg-hover);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color .2s !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
  }

  /* ── Selectbox & radio ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stRadio"] {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
  }

  /* ── Divider ── */
  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
    margin: 1.2rem 0;
  }

  /* ── Info box ── */
  .info-box {
    background: #162032;
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: .8rem 1rem;
    font-size: .85rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }
  .warn-box {
    background: #2d2210;
    border-left: 3px solid var(--accent-orange);
    border-radius: 0 8px 8px 0;
    padding: .8rem 1rem;
    font-size: .85rem;
    color: #e3b341;
    line-height: 1.6;
  }

  /* ── Progress / spinner override ── */
  .stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }

  /* ── Streamlit image caption ── */
  .stImage caption { color: var(--text-secondary) !important; font-size: .8rem !important; }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    transition: opacity .2s, transform .1s !important;
    box-shadow: 0 4px 15px #1f6feb44 !important;
  }
  .stButton > button:hover { opacity: .88; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "melanoma_resnet18.pth")
CLASS_NAMES  = ["Benign", "Malignant"]
IMG_SIZE     = 224
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]

CAM_ALGORITHMS = {
    "GradCAM":      GradCAM,
    "GradCAM++":    GradCAMPlusPlus,
    "HiResCAM":     HiResCAM,
    "XGradCAM":     XGradCAM,
    "EigenCAM":     EigenCAM,
    "ScoreCAM":     ScoreCAM,
    "AblationCAM":  AblationCAM,
    "FullGrad":     FullGrad,
}

CAM_DESCRIPTIONS = {
    "GradCAM":     "Uses gradient information flowing into the last conv layer to understand which pixels are important.",
    "GradCAM++":   "Improved GradCAM with better localisation for multiple objects of same class in single image.",
    "HiResCAM":    "High-resolution variant that produces sharper attention maps than standard GradCAM.",
    "XGradCAM":    "Axiom-based GradCAM offering a more principled explanation by satisfying sensitivity & conservation.",
    "EigenCAM":    "Uses the first principal component of activations — fast and requires no backpropagation.",
    "ScoreCAM":    "Score-based approach using forward passes to generate class-discriminative maps (slower, high quality).",
    "AblationCAM": "Selectively ablates activation channels to measure their contribution — robust to noise.",
    "FullGrad":    "Uses all bias layers (not just the target layer) for a complete gradient-based explanation.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> nn.Module:
    """Load fine-tuned ResNet-18 melanoma classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PRE/POST PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

_inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1.0 / s for s in STD],
)


def preprocess(pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Return (normalised tensor [1,C,H,W], float32 RGB array [H,W,3])."""
    rgb = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(rgb).astype(np.float32) / 255.0
    tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    return tensor, arr


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float, np.ndarray]:
    """Return (class_idx, confidence %, softmax probs)."""
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx  = int(np.argmax(probs))
    conf = float(probs[idx]) * 100
    return idx, conf, probs


# ─────────────────────────────────────────────────────────────────────────────
# CAM GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_cam(
    model:      nn.Module,
    tensor:     torch.Tensor,
    rgb_arr:    np.ndarray,
    cam_name:   str,
    target_cls: int,
) -> np.ndarray:
    """Generate a CAM overlay image (H,W,3) uint8."""
    cam_cls   = CAM_ALGORITHMS[cam_name]
    target_lyr = [model.layer4[-1]]
    targets    = [ClassifierOutputTarget(target_cls)]

    with cam_cls(model=model, target_layers=target_lyr) as cam:
        grayscale = cam(input_tensor=tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_arr, grayscale, use_rgb=True)
    return overlay, grayscale


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_side_by_side_fig(rgb_arr, overlay, cam_name, pred_label, conf):
    """Return a matplotlib Figure with original + overlay side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    axes[0].imshow(rgb_arr)
    axes[0].set_title("Original Scan", color="#e6edf3", fontsize=12, pad=10, fontweight=600)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"{cam_name}  ·  Predicted: {pred_label}  ({conf:.1f}%)",
        color="#e6edf3", fontsize=12, pad=10, fontweight=600,
    )
    axes[1].axis("off")

    plt.tight_layout(pad=1.5)
    return fig


def make_heatmap_only_fig(grayscale):
    """Return a figure showing the raw heatmap with a colorbar."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(grayscale, cmap="inferno")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelcolor="#8b949e", labelsize=8)
    cbar.outline.set_edgecolor("#30363d")
    plt.tight_layout()
    return fig


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: .5rem 0 1.5rem;">
          <span style="font-size:2.8rem;">🔬</span>
          <div style="font-size:1.1rem; font-weight:700; color:#e6edf3; margin-top:.4rem;">
            Melanoma Analyzer
          </div>
          <div style="font-size:.75rem; color:#8b949e; margin-top:.2rem;">
            ResNet-18 · Grad-CAM Suite
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Upload Skin Lesion Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Supported formats: JPEG, PNG, BMP, TIFF, WebP",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">CAM Algorithm</div>', unsafe_allow_html=True)
        cam_choice = st.selectbox(
            label="",
            options=list(CAM_ALGORITHMS.keys()),
            index=0,
            help="Select the Class Activation Mapping algorithm to visualize.",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Visualization Options</div>', unsafe_allow_html=True)

        show_heatmap = st.checkbox("Show raw heatmap", value=False,
                                   help="Display the raw grayscale activation map alongside the overlay.")
        target_mode  = st.radio(
            "Target class for CAM",
            ["Predicted class (auto)", "Benign", "Malignant"],
            help="Which class the CAM should highlight.",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <strong style="color:#58a6ff">About this tool</strong><br>
          Upload a dermoscopic or clinical image of a skin lesion.
          The model will classify it as <em>Benign</em> or <em>Malignant</em>
          and highlight the regions that influenced its decision using the chosen CAM method.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:2rem; text-align:center; font-size:.72rem; color:#484f58;">
          ResNet-18 · Melanoma Cancer Dataset<br>
          Grad-CAM Suite v1.5.5
        </div>
        """, unsafe_allow_html=True)

    return uploaded, cam_choice, show_heatmap, target_mode


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">🔬 Melanoma CAM Visualizer</div>
      <div class="hero-sub">
        Deep-learning classification + Class Activation Mapping for skin lesion analysis.
        Upload a lesion image, get an instant diagnosis, and see <em>exactly why</em> the model decided.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    uploaded, cam_choice, show_heatmap, target_mode = render_sidebar()

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.markdown(f"""
        <div class="warn-box">
          ⚠️  Model weights not found at <code>{MODEL_PATH}</code>.<br>
          Place <code>melanoma_resnet18.pth</code> in the same folder as <code>app.py</code>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner("Loading model …"):
        model = load_model(MODEL_PATH)

    # ── Idle state ────────────────────────────────────────────────────────────
    if uploaded is None:
        col_a, col_b, col_c = st.columns(3)
        tiles = [
            ("🧠", "ResNet-18", "Fine-tuned on Melanoma Cancer Dataset"),
            ("🎯", "8 CAM Methods", "GradCAM, ScoreCAM, EigenCAM & more"),
            ("⚡", "Real-time", "CPU & GPU inference supported"),
        ]
        for col, (icon, title, desc) in zip([col_a, col_b, col_c], tiles):
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center; padding:2rem 1rem;">
                  <div style="font-size:2.2rem; margin-bottom:.6rem;">{icon}</div>
                  <div style="font-weight:700; color:#e6edf3; margin-bottom:.3rem;">{title}</div>
                  <div style="font-size:.82rem; color:#8b949e;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:.5rem;">
          <div class="section-label">How to use</div>
          <ol style="color:#8b949e; font-size:.88rem; line-height:2; margin:0; padding-left:1.3rem;">
            <li>Upload a dermoscopic image of a skin lesion using the left sidebar.</li>
            <li>Choose a CAM algorithm from the dropdown.</li>
            <li>View the diagnosis (Benign / Malignant) and confidence score.</li>
            <li>Explore the heatmap overlay to understand the model's focus areas.</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Process image ─────────────────────────────────────────────────────────
    pil_img = Image.open(uploaded)
    tensor, rgb_arr = preprocess(pil_img)

    with st.spinner("Running inference …"):
        pred_idx, conf, probs = predict(model, tensor)

    pred_label = CLASS_NAMES[pred_idx]

    # ── Resolve target class for CAM ──────────────────────────────────────────
    if target_mode == "Predicted class (auto)":
        cam_target = pred_idx
        cam_target_label = f"{pred_label} (predicted)"
    elif target_mode == "Benign":
        cam_target = 0
        cam_target_label = "Benign (forced)"
    else:
        cam_target = 1
        cam_target_label = "Malignant (forced)"

    # ── Layout: classification result ─────────────────────────────────────────
    res_col, meta_col = st.columns([3, 2], gap="large")

    with res_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Classification Result</div>', unsafe_allow_html=True)

        badge_cls  = "badge-benign" if pred_label == "Benign" else "badge-malignant"
        badge_icon = "✅" if pred_label == "Benign" else "⚠️"
        st.markdown(f"""
        <div style="margin:.5rem 0 1rem;">
          <span class="{badge_cls}">{badge_icon} {pred_label.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        bar_cls = "conf-bar-fill-benign" if pred_label == "Benign" else "conf-bar-fill-malignant"
        st.markdown(f"""
        <div style="font-size:.82rem; color:#8b949e; margin-bottom:.3rem;">
          Confidence: <strong style="color:#e6edf3;">{conf:.1f}%</strong>
        </div>
        <div class="conf-bar-container">
          <div class="{bar_cls}" style="width:{conf:.1f}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Both-class breakdown
        st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
        for i, cname in enumerate(CLASS_NAMES):
            pct = probs[i] * 100
            bar_c = "conf-bar-fill-benign" if cname == "Benign" else "conf-bar-fill-malignant"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:.82rem; color:#8b949e; margin-bottom:.3rem;">
              <span>{cname}</span>
              <span style="color:#e6edf3; font-family:'JetBrains Mono',monospace;">{pct:.2f}%</span>
            </div>
            <div class="conf-bar-container" style="margin-bottom:.8rem;">
              <div class="{bar_c}" style="width:{pct:.1f}%"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # end card

    with meta_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Image Details</div>', unsafe_allow_html=True)

        w, h = pil_img.size
        m1, m2, m3 = st.columns(3)
        for col, val, label in zip(
            [m1, m2, m3],
            [f"{w}px", f"{h}px", DEVICE.type.upper()],
            ["Width", "Height", "Device"],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="metric-value">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">CAM Configuration</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table style="width:100%; font-size:.82rem; color:#8b949e; border-collapse:collapse;">
          <tr>
            <td style="padding:.35rem 0; color:#58a6ff;">Algorithm</td>
            <td style="padding:.35rem 0; text-align:right; color:#e6edf3;">{cam_choice}</td>
          </tr>
          <tr>
            <td style="padding:.35rem 0; color:#58a6ff;">Target</td>
            <td style="padding:.35rem 0; text-align:right; color:#e6edf3;">{cam_target_label}</td>
          </tr>
          <tr>
            <td style="padding:.35rem 0; color:#58a6ff;">Layer</td>
            <td style="padding:.35rem 0; text-align:right; color:#e6edf3;">layer4[-1]</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="divider"></div>
        <div class="info-box" style="margin-top:.5rem;">
          <strong style="color:#58a6ff;">{cam_choice}</strong><br>
          {CAM_DESCRIPTIONS.get(cam_choice, '')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # end card

    # ── Generate CAM ──────────────────────────────────────────────────────────
    st.markdown('<div class="divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-label" style="font-size:.9rem; color:#e6edf3; margin-bottom:.8rem;">
      🗺️ &nbsp;{cam_choice} Visualization &nbsp;·&nbsp;
      <span style="color:#8b949e; font-weight:400;">Target: {cam_target_label}</span>
    </div>
    """, unsafe_allow_html=True)

    slow_cams = {"ScoreCAM", "AblationCAM"}
    if cam_choice in slow_cams:
        st.markdown(f"""
        <div class="warn-box" style="margin-bottom:.8rem;">
          ⏳ <strong>{cam_choice}</strong> uses multiple forward passes and may take 20-60 seconds.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner(f"Generating {cam_choice} …"):
        overlay, grayscale = generate_cam(model, tensor, rgb_arr, cam_choice, cam_target)

    # ── Display images ────────────────────────────────────────────────────────
    if show_heatmap:
        img_col, cam_col, heat_col = st.columns([1, 1, 1], gap="medium")
        cols_config = [(img_col, "Original Scan", rgb_arr),
                       (cam_col, f"{cam_choice} Overlay", overlay),
                       (heat_col, "Raw Activation Map", None)]
    else:
        img_col, cam_col = st.columns([1, 1], gap="large")
        cols_config = [(img_col, "Original Scan", rgb_arr),
                       (cam_col, f"{cam_choice} Overlay", overlay)]

    for col, title, arr in cols_config:
        with col:
            st.markdown(f'<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
            if arr is not None:
                st.image(arr, use_container_width=True)
            else:
                # Raw heatmap with colorbar via matplotlib
                heat_fig = make_heatmap_only_fig(grayscale)
                st.pyplot(heat_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Download section ──────────────────────────────────────────────────────
    st.markdown('<div class="divider" style="margin:1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3, gap="medium")

    with dl1:
        side_fig = make_side_by_side_fig(rgb_arr, overlay, cam_choice, pred_label, conf)
        side_pil = fig_to_pil(side_fig)
        buf1 = io.BytesIO()
        side_pil.save(buf1, format="PNG")
        st.download_button(
            label="⬇  Download Side-by-Side",
            data=buf1.getvalue(),
            file_name=f"melanoma_{cam_choice}_comparison.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl2:
        overlay_pil = Image.fromarray(overlay)
        buf2 = io.BytesIO()
        overlay_pil.save(buf2, format="PNG")
        st.download_button(
            label="⬇  Download CAM Overlay",
            data=buf2.getvalue(),
            file_name=f"melanoma_{cam_choice}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl3:
        orig_buf = io.BytesIO()
        pil_img.save(orig_buf, format="PNG")
        st.download_button(
            label="⬇  Download Original",
            data=orig_buf.getvalue(),
            file_name="melanoma_original.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3rem; text-align:center; font-size:.75rem; color:#484f58; padding-bottom:2rem;">
      ⚠️ &nbsp;This tool is for <strong style="color:#8b949e;">educational and research purposes only</strong>.
      It does not constitute medical advice. Always consult a qualified dermatologist for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
