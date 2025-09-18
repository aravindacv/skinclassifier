import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    RocCurveDisplay, classification_report
)

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

import cv2  # for contour-based bbox from Grad-CAM mask

st.set_page_config(page_title="Skin Diseases Classifier (Explainable)", layout="wide")
st.title("Skin Diseases Classifier")

st.markdown("""
This demo uses your 4 classes (**glioma, meningioma, pituitary, notumour**).  
Pipeline: **ResNet-18 feature extractor (frozen)** → **Logistic Regression head** → **Grad-CAM**.  
**New:** Bounding box from Grad-CAM + **NLP report** for each image.
""")

# ---------- Sidebar ----------
st.sidebar.header("Dataset & Settings")
DEFAULT_ROOT = r"C:\Users\aravi\Downloads\brain_tumor_ai\data\brain_paper\MRI\Testing"
data_root = st.sidebar.text_input("Dataset root (ImageFolder)", DEFAULT_ROOT)

images_per_class = st.sidebar.slider("Max images per class (speed control)", 20, 1000, 200, 20)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
batch_size = st.sidebar.selectbox("Batch size (feature extraction)", [8, 16, 32, 64], index=2)

cam_thresh = st.sidebar.slider("Grad-CAM threshold (bbox)", 0.45, 0.90, 0.60, 0.01)
top_k_gallery = st.sidebar.slider("Grad-CAM gallery images", 3, 12, 6, 1)

do_train = st.sidebar.button("Load data & Train")

# ---------- Torch setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ---------- Session state ----------
for k in ["clf", "class_names", "feature_extractor", "cam_model",
          "Xte_feats", "Xte_paths"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------- Models ----------
@st.cache_resource(show_spinner=False)
def load_resnet18_feature_extractor():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(m.children())[:-1])   # (B,512,1,1)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone

def build_resnet18_for_cam(num_classes: int):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device).eval()

class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.target = self.model.layer4[-1].conv2
        self.activations = None
        self.gradients = None
        self._hook()

    def _hook(self):
        def fwd_hook(_, __, out):
            self.activations = out.detach()
        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target.register_forward_hook(fwd_hook)
        self.target.register_full_backward_hook(bwd_hook)

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        w = torch.mean(self.gradients, dim=(2,3), keepdim=True)  # (C,1,1)
        cam = torch.sum(w * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()  # (H,W), [0,1]

# ---------- Utils ----------
def stratified_subset_indices(imagefolder_dataset, per_class, seed=42):
    targets = np.array([label for _, label in imagefolder_dataset.samples])
    cls_to_idx = imagefolder_dataset.class_to_idx
    rng = np.random.RandomState(seed)
    sel = []
    for _, idx in cls_to_idx.items():
        idxs = np.where(targets == idx)[0]
        rng.shuffle(idxs)
        sel.extend(idxs[:per_class])
    return np.array(sel, dtype=int)

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y

def extract_features(backbone, dataloader):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, ys in dataloader:
            imgs = imgs.to(device)
            f = backbone(imgs).view(imgs.size(0), -1).cpu().numpy()
            feats.append(f)
            labels.append(ys.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)

def ensure_trained(mini=False):
    if st.session_state.clf is not None and st.session_state.class_names is not None:
        return True
    if not os.path.isdir(data_root):
        st.error("Dataset root does not exist. Check the path in the sidebar.")
        return False
    try:
        st.session_state.feature_extractor = load_resnet18_feature_extractor()
        full_ds = datasets.ImageFolder(root=data_root, transform=TFM)
        class_names = full_ds.classes
        if len(class_names) < 2:
            st.error("Found fewer than 2 classes. Verify folder structure.")
            return False

        per_class = min(60, images_per_class) if mini else images_per_class
        idxs = stratified_subset_indices(full_ds, per_class=per_class, seed=random_state)
        paths = [full_ds.samples[i][0] for i in idxs]
        ys    = np.array([full_ds.samples[i][1] for i in idxs])

        Xtr, Xte, ytr, yte = train_test_split(paths, ys, test_size=test_size,
                                              random_state=random_state, stratify=ys)
        train_items = [(p, int(ytr[i])) for i, p in enumerate(Xtr)]
        test_items  = [(p, int(yte[i])) for i, p in enumerate(Xte)]

        train_dl = DataLoader(PathDataset(train_items, TFM), batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(PathDataset(test_items,  TFM), batch_size=batch_size, shuffle=False)

        Xtr_f, ytr_f = extract_features(st.session_state.feature_extractor, train_dl)
        Xte_f, yte_f = extract_features(st.session_state.feature_extractor, test_dl)

        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(Xtr_f, ytr_f)

        st.session_state.clf = clf
        st.session_state.class_names = class_names
        st.session_state.cam_model = build_resnet18_for_cam(num_classes=len(class_names))
        st.session_state.Xte_feats = Xte_f
        st.session_state.Xte_paths = Xte
        st.session_state.yte = yte_f
        return True
    except Exception as e:
        st.error("Auto-training failed.")
        st.exception(e)
        return False

# ---- Grad-CAM -> Mask -> BBox ----
def cam_to_bbox(cam_01, threshold=0.6):
    """
    cam_01: (H,W) in [0,1]
    returns bbox (x1,y1,x2,y2) in cam coordinates, or None
    """
    H, W = cam_01.shape
    mask = (cam_01 >= threshold).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x+w, y+h), mask

def draw_bbox_on_pil(pil_img, bbox, color=(255, 0, 0), width_px=4, label_text=None):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    for i in range(width_px):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
    if label_text:
        # simple label box
        text_bg = (color[0], color[1], color[2], 128)
        draw.text((x1 + 5, y1 + 5), label_text, fill=color)
    return img

def hotspot_quadrant(bbox, W, H):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    vert = "upper" if cy < H/2 else "lower"
    horiz = "left" if cx < W/2 else "right"
    return f"{vert}-{horiz}"

def nlp_report(pred_name, proba_vec, class_names, bbox, img_w, img_h):
    conf = float(np.max(proba_vec))
    # second best
    order = np.argsort(proba_vec)[::-1]
    best, second = order[0], order[1] if len(order) > 1 else None
    second_name = class_names[second] if second is not None else None
    second_conf = float(proba_vec[second]) if second is not None else 0.0

    if bbox is not None:
        loc = hotspot_quadrant(bbox, img_w, img_h)
        box_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        area_pct = 100.0 * box_area / (img_w * img_h)
        area_txt = f"hotspot localized in the **{loc}** region (~{area_pct:.1f}% area)"
    else:
        area_txt = "no clear hotspot localized at the current threshold"

    text = (
        f"**Prediction:** {pred_name}  \n"
        f"**Confidence:** {conf:.3f}  \n"
        + (f"**Second guess:** {second_name} ({second_conf:.3f})  \n" if second_name else "")
        + f"**Grad-CAM:** {area_txt}.  \n"
        f"**Caution:** this is a screening aid; confirm with radiologist and clinical context."
    )
    return text

# ---------- TRAIN ----------
if do_train:
    with st.spinner("Training on your dataset..."):
        ok = ensure_trained(mini=False)
    if ok:
        clf = st.session_state.clf
        class_names = st.session_state.class_names
        Xte_f = st.session_state.Xte_feats
        yte = st.session_state.yte
        y_pred = clf.predict(Xte_f)
        y_proba = clf.predict_proba(Xte_f)

        acc = accuracy_score(yte, y_pred)
        f1m = f1_score(yte, y_pred, average="macro")
        st.subheader("📊 Performance")
        st.write(f"Accuracy: **{acc:.3f}** | Macro-F1: **{f1m:.3f}**")

        rep = classification_report(yte, y_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.style.format(precision=3), use_container_width=True)  # table supports use_container_width; ignore warning

        cm = confusion_matrix(yte, y_pred)
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        im = ax_cm.imshow(cm, interpolation='nearest', aspect='auto')
        ax_cm.figure.colorbar(im, ax=ax_cm)
        ax_cm.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                  xticklabels=class_names, yticklabels=class_names,
                  ylabel='True', xlabel='Predicted', title='Confusion Matrix')
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig_cm)

        # ROC (one-vs-rest)
        st.subheader("ROC Curves (One-vs-Rest)")
        try:
            y_bin = label_binarize(yte, classes=np.arange(len(class_names)))
            fig_roc, ax_roc = plt.subplots()
            for i, cname in enumerate(class_names):
                RocCurveDisplay.from_predictions(y_bin[:, i], y_proba[:, i], name=cname, ax=ax_roc)
            st.pyplot(fig_roc)
            auc_macro = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
            st.write(f"Macro ROC-AUC: **{auc_macro:.3f}**")
        except Exception as e:
            st.info(f"ROC-AUC not available: {e}")

        # Grad-CAM gallery with bbox + NLP
        st.subheader("🩺 Grad-CAM Heatmaps with Bounding Boxes + NLP")
        cam_model = st.session_state.cam_model
        grad_cam = GradCAM(cam_model)

        cols = st.columns(3)
        for i, path in enumerate(st.session_state.Xte_paths[:top_k_gallery]):
            try:
                pil = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                tens = TFM(pil).unsqueeze(0).to(device)
                # Predict via feature head
                f = st.session_state.feature_extractor(tens).view(1, -1).cpu().numpy()
                proba = clf.predict_proba(f)[0]
                pred_idx = int(np.argmax(proba))
                pred_name = class_names[pred_idx]
                # CAM
                _ = cam_model(tens)
                cam = grad_cam.generate(tens, class_idx=pred_idx)
                # BBox
                bbox, mask = cam_to_bbox(cam, threshold=cam_thresh)
                draw = pil
                if bbox is not None:
                    draw = draw_bbox_on_pil(pil, bbox, color=(255, 64, 64), width_px=3, label_text=pred_name)
                # Layout
                c = cols[i % 3]
                c.image(draw, caption=os.path.basename(path), width="stretch")
                c.markdown(nlp_report(pred_name, proba, class_names, bbox,
                                      img_w=IMG_SIZE, img_h=IMG_SIZE))
            except Exception as e:
                st.warning(f"Failed on {os.path.basename(path)}: {e}")

# ---------- SINGLE IMAGE UPLOAD ----------
st.subheader("🔎 Try your own MRI image")
up = st.file_uploader("Upload a single MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

if up is not None:
    if st.session_state.clf is None:
        with st.spinner("No trained model found. Training a quick mini model..."):
            ok = ensure_trained(mini=True)
            if not ok:
                st.stop()

    clf = st.session_state.clf
    class_names = st.session_state.class_names
    feature_extractor = st.session_state.feature_extractor
    cam_model = st.session_state.cam_model or build_resnet18_for_cam(num_classes=len(class_names))
    st.session_state.cam_model = cam_model
    grad_cam = GradCAM(cam_model)

    pil = Image.open(up).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    tens = TFM(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        f = feature_extractor(tens).view(1, -1).cpu().numpy()
    proba = clf.predict_proba(f)[0]
    pred_idx = int(np.argmax(proba))
    pred_name = class_names[pred_idx]

    # Grad-CAM + bbox
    with torch.no_grad():
        _ = cam_model(tens)
    cam = grad_cam.generate(tens, class_idx=pred_idx)
    bbox, mask = cam_to_bbox(cam, threshold=cam_thresh)
    draw = pil
    if bbox is not None:
        draw = draw_bbox_on_pil(pil, bbox, color=(255, 64, 64), width_px=3, label_text=pred_name)

    # Show results
    st.image(draw, caption=f"Prediction: {pred_name}", width="stretch")
    st.bar_chart(pd.Series(proba, index=class_names))
    st.markdown(nlp_report(pred_name, proba, class_names, bbox,
                           img_w=IMG_SIZE, img_h=IMG_SIZE))
