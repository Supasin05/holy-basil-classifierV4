"""
แอปตรวจใบกะเพรา: ใบดี / ใบเสีย (MobileNetV2 + Grad-CAM)
รัน:  streamlit run app.py
ต้องมีไฟล์โมเดล best_basil_mobilenetv2_bgaug.pt อยู่โฟลเดอร์เดียวกับไฟล์นี้
(ไฟล์ที่ notebook v2 เซฟไว้ใน Drive โฟลเดอร์ basil/)
"""
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

MODEL_CANDIDATES = ["best_basil_mobilenetv2_bgaug.pt", "best_basil_mobilenetv2.pt"]
DEFAULT_CLASSES = ["bad", "good"]          # ต้องตรงกับลำดับใน notebook (bad = 0, good = 1)
PLAIN_BG = (70, 70, 70)                    # พื้นเทาเข้ม ใกล้ภาพที่ใช้ฝึก (ใช้กับภาพ PNG โปร่งใส)
MAX_SIDE = 1280                            # ย่อภาพใหญ่ก่อนเข้าโมเดลเพื่อความเร็ว
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    classes = ckpt.get("classes", DEFAULT_CLASSES) if isinstance(ckpt, dict) else DEFAULT_CLASSES
    img_size = int(ckpt.get("img_size", 224)) if isinstance(ckpt, dict) else 224
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, len(classes)))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, list(classes), img_size, path


def load_image(file) -> Image.Image:
    """เปิดภาพ แก้การหมุนจากมือถือ และวางภาพโปร่งใสบนพื้นเทาเข้ม"""
    im = Image.open(file)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, PLAIN_BG + (255,))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def make_transform(img_size: int):
    # ต้องเหมือน eval_tf ใน notebook
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(model, x) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0].cpu().numpy()


def gradcam_overlay(model, x, resized: Image.Image) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    try:
        heat = cam(input_tensor=x)[0]
    finally:  # ปล่อย hook ทุกครั้ง ไม่ให้สะสมเมื่อโมเดลถูกแชร์ข้ามผู้ใช้
        release = getattr(getattr(cam, "activations_and_grads", None), "release", None)
        if release:
            release()
    rgb = np.asarray(resized, dtype=np.float32) / 255
    return show_cam_on_image(rgb, heat, use_rgb=True)


def main():
    st.set_page_config(page_title="ตรวจใบกะเพรา ดี/เสีย", page_icon="🌿", layout="wide")

    loaded = load_model()
    if loaded is None:
        st.error("ไม่พบไฟล์โมเดล `best_basil_mobilenetv2_bgaug.pt` ในโฟลเดอร์เดียวกับ app.py")
        st.stop()
    model, classes, img_size, model_path = loaded

    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        threshold = st.slider(
            "เกณฑ์ความมั่นใจขั้นต่ำ", 0.50, 0.95, 0.70, 0.05,
            help="ถ้าความมั่นใจสูงสุดต่ำกว่าค่านี้ จะตอบว่า 'ไม่แน่ใจ' แทนการเดา")
        show_cam = st.checkbox("แสดง Grad-CAM", value=True)
        st.divider()
        st.subheader("ข้อจำกัดของโมเดล")
        st.markdown(
            "- ฝึกกับใบกะเพรา **ใบเดียว** พื้นหลังเรียบ\n"
            "- ใบพืชชนิดอื่น หรือภาพที่มีหลายใบ อาจให้ผลไม่แม่น\n"
            "- เป็นการคัดกรองเบื้องต้นจากภาพ **ไม่ใช่การวินิจฉัยโรค**\n"
            "- ตำหนิเล็กมากอาจตรวจไม่เจอ")
        st.caption(f"โมเดล: {os.path.basename(model_path)} | ขนาดภาพเข้า {img_size}px | {device.type}")

    st.title("🌿 ตรวจใบกะเพรา: ใบดี / ใบเสีย")
    st.caption("ถ่ายหรืออัปโหลดรูปใบกะเพรา 1 ใบ ให้เห็นทั้งใบ แสงพอ วางบนพื้นเรียบจะแม่นที่สุด")

    tab_up, tab_cam = st.tabs(["📁 อัปโหลดรูป", "📷 ถ่ายด้วยกล้อง"])
    with tab_up:
        uploaded = st.file_uploader("เลือกรูปใบกะเพรา", type=["jpg", "jpeg", "png", "webp"])
    with tab_cam:
        captured = st.camera_input("ถ่ายใบกะเพรา")
    source = uploaded if uploaded is not None else captured

    if source is None:
        st.info("อัปโหลดหรือถ่ายรูปใบกะเพราเพื่อเริ่มตรวจ")
        st.stop()

    try:
        pil = load_image(source)
    except Exception:
        st.error("เปิดไฟล์ภาพนี้ไม่ได้ ลองใช้ไฟล์ .jpg หรือ .png")
        st.stop()

    resized = pil.resize((img_size, img_size), Image.BILINEAR)
    x = make_transform(img_size)(pil).unsqueeze(0).to(device)
    probs = predict(model, x)

    pred_idx = int(probs.argmax())
    pred, conf = classes[pred_idx], float(probs[pred_idx])
    p_good = float(probs[classes.index("good")])
    p_bad = float(probs[classes.index("bad")])

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📷 ภาพที่อัปโหลด")
        st.image(pil, use_container_width=True)
    with col_b:
        if show_cam:
            st.subheader("🔍 พื้นที่วิเคราะห์ (Grad-CAM)")
            st.image(gradcam_overlay(model, x, resized), use_container_width=True)
            st.caption("สีแดง = บริเวณที่โมเดลใช้ตัดสินใจมากที่สุด "
                       "ถ้าสีแดงอยู่นอกใบหรือที่พื้นหลัง ให้ระวังผลลัพธ์")

    st.divider()
    st.subheader("📊 สรุปผลการตรวจ")

    if conf < threshold:
        st.warning(
            f"⚠️ **ผลไม่แน่ชัด** (ความมั่นใจสูงสุด {conf:.1%} ต่ำกว่าเกณฑ์ {threshold:.0%})\n\n"
            "โมเดลตัดสินไม่ได้ อาจเพราะพื้นหลัง แสง หรือมุมถ่ายต่างจากข้อมูลที่ใช้ฝึก "
            "ลองถ่ายใหม่ให้เห็นใบเดียวชัดๆ บนพื้นเรียบ")
    elif pred == "good":
        st.success(f"✅ **ใบดี** (ความมั่นใจ {conf:.1%})")
        st.write("ไม่พบลักษณะใบเสียที่โมเดลรู้จัก แต่ควรสังเกตใบด้วยตาอีกครั้งโดยเฉพาะด้านหลังใบ")
    else:
        st.error(f"❌ **ใบเสีย** (ความมั่นใจ {conf:.1%})")
        st.write("พบลักษณะคล้ายใบที่มีตำหนิ (เช่น จุดดำ หรือสีเหลืองซีดเป็นหย่อม) "
                 "ควรตรวจใบด้วยตาและแยกใบนี้ออกก่อน")

    st.progress(p_good, text=f"ใบดี {p_good:.1%}")
    st.progress(p_bad, text=f"ใบเสีย {p_bad:.1%}")

    if min(pil.size) < 150:
        st.caption("⚠️ ภาพมีขนาดเล็กมาก รอยตำหนิอาจมองไม่เห็น ลองใช้ภาพที่ชัดกว่านี้")

    with st.expander("รายละเอียดทางเทคนิค"):
        st.write({classes[i]: round(float(p), 4) for i, p in enumerate(probs)})
        st.write(f"ขนาดภาพเดิม {pil.size[0]}x{pil.size[1]} -> ย่อเป็น {img_size}x{img_size} ก่อนเข้าโมเดล")


if __name__ == "__main__":
    main()
