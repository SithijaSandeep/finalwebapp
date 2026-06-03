import streamlit as st
import tensorflow as tf
import numpy as np
import json

# ============================================================
# FIX: Lambda layer support for old model
# ============================================================

def dummy_cast(x):
    return tf.cast(x, tf.float32)

custom_objects = {
    "tf": tf,
    "cast": dummy_cast
}

# ============================================================
# LOAD MODEL (FIXED)
# ============================================================

model = tf.keras.models.load_model(
    "waste_classifier.keras",
    custom_objects=custom_objects,
    compile=False
)

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ============================================================
# UI
# ============================================================

st.title("♻️ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = tf.keras.utils.load_img(uploaded_file, target_size=(160,160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(uploaded_file, caption="Uploaded Image")
    st.write("Prediction:", predicted_class)

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import json

# # Load model
# model = tf.keras.models.load_model("waste_classifier.keras")

# # Load class names
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)

# st.title("♻️ Waste Classification App")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:

#     img = tf.keras.utils.load_img(uploaded_file, target_size=(160,160))
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]

#     st.image(uploaded_file, caption="Uploaded Image")
#     st.write("Prediction:", predicted_class)
