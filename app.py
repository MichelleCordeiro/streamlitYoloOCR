from pathlib import Path
import PIL
import streamlit as st

import settings
import helper

st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Object Detection using YOLOv8")
st.markdown('Updload a photo or a video, paste a url from Youtube and open webcam with this hand signals: :+1:, :hand:, :spock-hand: and :i_love_you_hand_sign:.')
st.markdown('Then click the Detect Objects button and check the result.')

st.sidebar.header("ML Model Config")

model_type = st.sidebar.radio(
    "TASK", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "MODEL CONFIDENCE", 25, 100, 50)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True,  width=40)



        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")
