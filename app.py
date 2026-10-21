import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Hair & Scalp Analyzer", layout="centered")

ingredient_map = {
    "Dandruff": ["Salicylic Acid", "Ketoconazole", "Zinc Pyrithione"],
    "Dermatitis": ["Ketoconazole", "Coal Tar", "Aloe Vera"],
    "Psoriasis": ["Salicylic Acid", "Coal Tar", "Soothing agents"],
    "Folliculitis": ["Tea Tree Oil", "Benzoyl Peroxide", "Salicylic Acid"],
    "Oily Hair": ["Salicylic Acid", "Tea Tree Oil", "Niacinamide"],
    "Dry Hair": ["Ceramides", "Panthenol", "Argan Oil"],
    "Alopecia": ["Minoxidil", "Biotin", "Caffeine"],
    "Hair Loss": ["Minoxidil", "Biotin", "Caffeine"],
    "Hair Loss Stage 1": ["Biotin", "Caffeine", "Saw Palmetto"],
    "Hair Loss Stage 2": ["Minoxidil", "Biotin", "Saw Palmetto"],
    "Hair Loss Stage 3": ["Minoxidil", "Redensyl", "Saw Palmetto"],
    "Hair Loss Stage 4": ["Minoxidil", "Redensyl", "Consult a dermatologist"],
    "Head Lice": ["Permethrin (consult pharmacist)", "Tea Tree Oil"],
    "Healthy Scalp": ["Maintain routine — no treatment needed"],
    "Grey Hair": ["No ingredient recommendation — cosmetic only"],
    "Red Hair": ["No ingredient recommendation — cosmetic only"],
}

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🧴 AI Hair & Scalp Analyzer")
st.write("Upload a scalp/hair image to detect conditions and get ingredient suggestions.")

st.caption("Ingredient suggestions are general guidance, not medical advice. Please consult a dermatologist for diagnosis.")
conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.35)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        results = model.predict(img, conf=conf_thresh, iou=0.5)

    detections = results[0].boxes
    if len(detections) == 0:
        st.warning("No conditions detected with confidence above threshold.")
    else:
        st.subheader("Detected conditions")
        seen = set()
        for box in detections:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"**{cls_name}** — {conf:.0%} confidence")
            if cls_name not in seen and cls_name in ingredient_map:
                st.write("Recommended ingredients:", ", ".join(ingredient_map[cls_name]))
            seen.add(cls_name)

        # show annotated image
        annotated = results[0].plot()
        st.image(annotated, caption="Detections", use_container_width=True, channels="BGR")