import json
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw


API_URL = "https://predict.ultralytics.com"
API_KEY = os.getenv("ULTRALYTICS_API_KEY", "ba9e29dcecf5b19a768e56a4f616df5d03baeef036")
MODEL_URL = "https://hub.ultralytics.com/models/eAjS72HEB8er9T7UWut0"


def call_ultralytics_api(image_bytes: bytes):
    if not API_KEY:
        raise RuntimeError("Ultralytics API key is missing. Set ULTRALYTICS_API_KEY env var or edit app.py.")

    headers = {"x-api-key": API_KEY}
    data = {"model": MODEL_URL, "imgsz": 640, "conf": 0.25, "iou": 0.45}

    files = {"file": ("upload.jpg", BytesIO(image_bytes), "image/jpeg")}

    response = requests.post(API_URL, headers=headers, data=data, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def interpret_results(result_json):
    images = result_json.get("images", [])
    if not images:
        return "No prediction available", [], False

    results = images[0].get("results", [])
    if not results:
        return "Normal Kidney (no stones detected)", results, False

    # Treat detections with names different from normal kidney as positive for stones.
    normal_labels = {"normal kidney", "normal_kidney", "normal"}
    has_non_normal = any((det.get("name") or "").lower() not in normal_labels for det in results)

    if has_non_normal:
        return "Kidney Stone Detected", results, True

    return "Normal Kidney (no stones detected)", results, False


def draw_bounding_boxes(image_bytes: bytes, detections):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    for det in detections or []:
        box = det.get("box") or {}
        x1 = box.get("x1")
        x2 = box.get("x2")
        y1 = box.get("y1")
        y2 = box.get("y2")

        if None in (x1, x2, y1, y2):
            continue

        label = (det.get("name") or "object").strip()
        confidence = det.get("confidence")

        color = "red"
        if label.lower() in {"normal kidney", "normal_kidney", "normal"}:
            color = "green"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return image


def main():
    st.set_page_config(page_title="RadAI: Kidney Stone Detection", page_icon="🩺", layout="centered")

    st.markdown("## RadAI: Kidney Stone Detection 🩺")
    st.markdown(
        """RadAI is an AI-powered binary classification tool designed to assist in detecting kidney stones in
        ultrasound images. The model classifies images into two categories: **Kidney Stone Detected** or
        **Normal Kidney**. Simply upload or capture an image, and RadAI will process it and provide real-time
        results based on the analysis."""
    )

    with st.expander("View Privacy Policy Details", expanded=True):
        st.markdown(
            """### Data Storage Disclaimer

**Data Storage**  
We only store the survey responses you provide. Your uploaded image is processed temporarily for
analysis during this survey and is not saved or retained after processing.

**Purpose of Data Collection**  
The information collected through this survey will be used solely for academic research
purposes. Our goal is to analyze the accuracy and effectiveness of our kidney stone detection
model as part of our study. Your data will not be used for commercial purposes or shared with
third parties outside of the research team.

**Confidentiality**  
Your personal information will be kept confidential and used only to validate the survey
responses. In any reports or publications resulting from this research, data will be presented in
an aggregated manner, ensuring that individual responses cannot be identified.

**Voluntary Participation**  
Participation in this survey is voluntary, and you may withdraw at any time. Should you choose to
withdraw, any data collected up to that point will be deleted from our records.

**User Acceptance Testing Survey**  
As part of our research requirements, we encourage participants to take the User Acceptance
Testing (UAT) survey. Your feedback is invaluable in assessing and improving the usability and
performance of our kidney stone detection model. Participants are required to email their
usability testing results to `winsonbaring10@gmail.com` or `winson.baring@cit.edu` so that we
can further interpret the data.

**Contact Information**  
If you have any concerns or inquiries regarding this survey or the research, you may contact John
Marc G. Balbada from the emails provided above.
"""
        )

    agree = st.checkbox("I agree to the Privacy Policy")

    if not agree:
        st.info("Please read and agree to the Privacy Policy above to proceed with image upload and analysis.")
        return

    uploaded_file = st.file_uploader(
        "Upload an ultrasound image for kidney stone detection",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()

            with st.spinner("Analyzing image, please wait..."):
                result_json = call_ultralytics_api(image_bytes)
                message, detections, positive = interpret_results(result_json)
                annotated_image = draw_bounding_boxes(image_bytes, detections)

            st.image(annotated_image, caption="Analyzed Image", use_container_width=True)

            if positive:
                st.error(message)
            else:
                st.success(message)

            with st.expander("View Raw Model Output"):
                st.json(result_json)
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")


if __name__ == "__main__":
    main()
