
from pathlib import Path
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Sayfa ayarları
st.set_page_config(
    page_title="Fruit & Vegetable AI",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruit & Vegetable Classification")
st.write("CNN ve VGG16 modelleri ile meyve-sebze sınıflandırma uygulaması.")

# Sınıf isimleri
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn',
    'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
    'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
    'onion', 'orange', 'paprika', 'pear', 'peas',
    'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon', 'pineapple'
]

# Modelleri yükle
@st.cache_resource
def load_models():
    base_path = Path(__file__).parent

    cnn_model = tf.keras.models.load_model(
        base_path / "Fruits.model.keras"
    )

    vgg_model = tf.keras.models.load_model(
        base_path / "Fruits_modelTL.keras"
    )

    return cnn_model, vgg_model

try:
    cnn_model, vgg_model = load_models()
    st.success("Modeller başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# Model seçimi
model_choice = st.selectbox(
    "Model seçiniz:",
    ["CNN", "VGG16"]
)

# Görsel yükleme
uploaded_file = st.file_uploader(
    "Bir meyve veya sebze görseli yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Görsel ön işleme
    img = image.resize((128, 128))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    if model_choice == "CNN":
        prediction = cnn_model.predict(img_array, verbose=0)
    else:
        prediction = vgg_model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_index]) * 100

    st.subheader("Tahmin Sonucu")
    st.success(f"Tahmin: {class_names[predicted_index]}")
    st.write(f"Güven oranı: %{confidence:.2f}")

    st.subheader("En yüksek 5 tahmin")

    top_5 = np.argsort(prediction[0])[-5:][::-1]

    for index in top_5:
        st.write(
            f"{class_names[index]}: %{prediction[0][index] * 100:.2f}"
        )
