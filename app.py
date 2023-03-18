import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import time

angry = 'https://www.youtube.com/embed/j1hrZIA-2nM'
happy = 'https://www.youtube.com/embed/tNca0jr850M'
neutral = 'https://www.youtube.com/embed/gZGLnxVrzD0'
sad = 'https://www.youtube.com/embed/e4N9al7vhVQ'
surprise = 'https://www.youtube.com/embed/dOKQeqGNJwY'

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}

# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

def play_sound(link):
    html_string="""
               <iframe width="1" height="1"  src="{}?autoplay=1" ></iframe>
            """.format(link)
    sound = st.empty()
    sound.markdown(html_string, unsafe_allow_html=True)
    time.sleep(400)
    sound.empty()

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def classify_image(face):
    resu = []
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(face, 1.3, 5)
    for (x, y, w, h) in faces:
        face = face[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([face]) != 0:
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0) #48,48,1
            #face = preprocess_input(face)
            pred = classifier.predict(face)[0]
            pred = int(np.argmax(pred)) # 0.98, 0.01, 0.01, 0.01, 0.01 = 1
            final_pred = emotion_dict[pred]  # happy
            output = str(final_pred) # happy 
            resu.append(output)

    return resu

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        choice = st.selectbox("Select Mode", ["Live", "Photo"])
        if choice == "Live":
            st.write("Click on start to use webcam and detect your face emotion")
            st.error("You don't get audio experience in this mode")
            st.success("Use Photo mode for audio experience")
            webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_transformer_factory=Faceemotion)
        elif choice == "Photo":
            img = st.camera_input("Webcam", key="webcam")
            if img is not None:
                image = Image.open(img)
                image = np.array(image)
                res = classify_image(image)
                #st.write(res)
                if len(res) != 0:
                    if res[0] == 'angry':
                        st.error("You are angry")
                        play_sound(angry)

                    elif res[0] == 'happy':
                        st.success("You are happy")
                        play_sound(happy)

                    elif res[0] == 'neutral':
                        st.write("You are neutral")
                        play_sound(neutral)

                    elif res[0] == 'sad':
                        st.write("You are sad")
                        play_sound(sad)

                    elif res[0] == 'surprise':
                        st.success("You are surprise")
                        play_sound(surprise)
                else:
                    st.error("No emotion detected")

    else:
        pass


if __name__ == "__main__":
    main()
