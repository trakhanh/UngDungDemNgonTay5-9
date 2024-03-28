import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import utils as utl
import matplotlib.pyplot as plt
import cv2
# CSS tùy chỉnh cho ứng dụng
custom_css = """
<style>
    /* Background color and white text for the entire app */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
        
    }
    /* Distinctive color for the title */
       h1 {
        color: #FF6347; /* Màu chữ tiêu đề */
        font-size: 3em; /* Kích thước chữ tiêu đề */
    }
    /* Styling for buttons */
    .stButton>button {
        background-color: #000000 /* Màu nền nút */
        color: #FFFFFF; /* Màu chữ nút */
        border-radius: 10px; /* Bo tròn viền nút */
        padding: 10px 20px; /* Khoảng cách bên trong nút */
        font-size: 1em; /* Kích thước chữ nút */
    }
    /* Styling for radio buttons */
    .stRadio>div>div>label>span:first-child {
        background-color: #FF6347;
        border-color: #FF6347;
    }
       /* Styling for file uploader */
    .stFileUploader>div>div>div>button {
        background-color: #FF6347 !important; /* Màu nền nút uploader */
        color: #FFFFFF !important; /* Màu chữ nút uploader */
        border-radius: 5px; /* Bo tròn viền nút uploader */
        padding: 10px 20px; /* Khoảng cách bên trong nút uploader */
        font-size: 1em; /* Kích thước chữ nút uploader */
    }
    /* Styling for progress bar */
    .stProgress>div>div>div>div {
        background-color: #FF6347;
    }
    /* Styling for tooltips */
    .stTooltip>div {
        background-color: #FF6347;
        
    }
</style>
"""
st.set_page_config(page_title="Ứng dụng phân lớp và phân đoạn ảnh", page_icon=":camera:")

# Sử dụng CSS tùy chỉnh
st.markdown(custom_css, unsafe_allow_html=True)
# Hàm để tải mô hình phân lớp và phân đoạn
def load_models():
    classification_model = tf.keras.models.load_model('best1_model_vgg16.h5')
    segmentation_model = tf.keras.models.load_model('best_model_unet_hihi.h5')
    return classification_model, segmentation_model
# Định nghĩa bản đồ màu cho phân đoạn
color_map = np.array([
    [0, 0, 0],    # Màu đen
    [255, 255, 255], # Màu trắng
    [0, 0, 255],  # Màu xanh dương
    [255, 255, 0],# Màu vàng
    [255, 165, 0] # Màu cam
])
# Hàm tiền xử lý ảnh cho mô hình phân lớp
def preprocess_image_classification(image, target_size=(150, 150)):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Hàm tiền xử lý ảnh cho mô hình phân đoạn
def preprocess_image_segmentation(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Hàm để hiển thị kết quả phân đoạn
def prediction_to_image(prediction, color_map):
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_class = predicted_class[0]  # Loại bỏ batch dimension
    segmented_image = np.zeros((predicted_class.shape[0], predicted_class.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(color_map):
        segmented_image[predicted_class == i] = color
    return segmented_image
# Load mô hình
classification_model, segmentation_model = load_models()

st.title("Ứng Dụng Phân Lớp và Phân Đoạn Ảnh")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Cho phép người dùng chọn chức năng
task = st.radio("Chọn chức năng:", ("Phân lớp ảnh", "Phân đoạn ảnh"))

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)

    if st.button('Xử lý ảnh'):
        if task == "Phân lớp ảnh":
          with st.spinner('Đang dự đoán phân lớp...'):
            processed_image = preprocess_image_classification(image)
            prediction = classification_model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)
            CLASSES = ['5 Ngon', '6 Ngon', '7 Ngon', '8 Ngon', '9 Ngon']
            class_name = CLASSES[predicted_class[0]].replace('Ngon', 'Ngón')
            st.success('Kết quả phân lớp ảnh')
            st.write(f'Kết quả: {class_name}')


        elif task == "Phân đoạn ảnh":
          with st.spinner('Đang dự đoán phân đoạn...'):
                processed_image = preprocess_image_segmentation(image, target_size=(256, 256))
                prediction = segmentation_model.predict(processed_image)
                segmented_image = prediction_to_image(prediction, color_map)
                st.success('Kết quả phân đoạn ảnh')
                st.image(segmented_image, caption='Kết quả phân đoạn ảnh', use_column_width=True)
