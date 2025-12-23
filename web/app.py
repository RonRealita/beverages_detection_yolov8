import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="YOLOv8 Beverages Detection",
    layout="centered"
)

st.title("Deteksi Minuman")
st.write("Deteksi minuman pada gambar, video, dan webcam secara realtime.")

# =========================
# LOAD MODEL
# =========================
model = YOLO("best.pt")

# =========================
# MENU MODE
# =========================
menu = st.radio(
    "Pilih Mode Deteksi:",
    ["Gambar", "Video", "Webcam"]
)

# =========================
# MODE 1: DETEKSI GAMBAR
# =========================
if menu == "Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli", use_column_width=True)

        img_array = np.array(image)

        with st.spinner("Sedang mendeteksi objek..."):
            results = model(img_array)

        result_img = results[0].plot()

        st.image(
            result_img,
            caption="Hasil Deteksi",
            use_column_width=True
        )

        st.success("Deteksi gambar selesai")

# =========================
# MODE 2: DETEKSI VIDEO
# =========================
elif menu == "Video":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("Deteksi video selesai")

# =========================
# MODE 3: REALTIME WEBCAM
# =========================
elif menu == "Webcam":
    st.warning("Webcam hanya berjalan di lokal, bukan di server cloud.")

    run = st.checkbox("Aktifkan Webcam")
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("Webcam dimatikan")
