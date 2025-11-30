import streamlit as st
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Flatten, 
                                     Bidirectional, Permute, multiply)
import numpy as np
import mediapipe as mp
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="AI Personal Fitness Trainer")

# ------------------ Model Definition ------------------ #
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul') 
    return output_attention_mul

@st.cache_resource
def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)
    model.load_weights("./models/LSTM_Attention.h5")
    return model

HIDDEN_UNITS = 256
model = build_model(HIDDEN_UNITS)

# ------------------ Streamlit UI ------------------ #
st.write("# AI Personal Fitness Trainer Web App")

st.markdown("❗❗ **Development Note** ❗❗")
st.markdown("Using x, y, z pose keypoints normalized w.r.t. frame size.")
st.markdown("- Next: joint angles or bounding box-relative features.")
st.write("Stay Tuned!")

st.write("## Settings")
threshold1 = st.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)
st.write("## Activate the AI 🤖🏋️‍♂️")

# ------------------ Pose Setup ------------------ #
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2)

# ------------------ Video Processor ------------------ #
class VideoProcessor:
    def __init__(self):
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = threshold3
        self.sequence = []
        self.current_action = ''
        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None

    def draw_landmarks(self, image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
    def extract_keypoints(self, results):
        return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    def calculate_angle(self, a,b,c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        return 360-angle if angle > 180.0 else angle

    def get_coordinates(self, landmarks, mp_pose, side, joint):
        coord = getattr(mp_pose.PoseLandmark, side.upper()+"_"+joint.upper())
        return [landmarks[coord.value].x, landmarks[coord.value].y]

    def viz_joint_angle(self, image, angle, joint):
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def count_reps(self, image, landmarks, mp_pose):
        if self.current_action == 'curl':
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            angle = self.calculate_angle(shoulder, elbow, wrist)
            if angle < 30: self.curl_stage = "up"
            if angle > 140 and self.curl_stage == 'up':
                self.curl_stage = "down"
                self.curl_counter += 1
            self.viz_joint_angle(image, angle, elbow)
        elif self.current_action == 'press':
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            if (elbow_angle > 130): self.press_stage = "up"
            if (elbow_angle < 50) and (self.press_stage == 'up'):
                self.press_stage = 'down'
                self.press_counter += 1
            self.viz_joint_angle(image, elbow_angle, elbow)
        elif self.current_action == 'squat':
            ls = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            lh = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
            lk = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
            la = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            rs = self.get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
            rh = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
            rk = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
            ra = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            lk_angle = self.calculate_angle(lh, lk, la)
            rh_angle = self.calculate_angle(rs, rh, rk)
            if lk_angle < 165 and rh_angle < 165:
                self.squat_stage = "down"
            if lk_angle > 165 and rh_angle > 165 and self.squat_stage == "down":
                self.squat_stage = "up"
                self.squat_counter += 1
            self.viz_joint_angle(image, lk_angle, lk)

    def prob_viz(self, res, image):
        for num, prob in enumerate(res):
            cv2.rectangle(image, (0,60+num*40), (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(image, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return image

    def process(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.draw_landmarks(image, results)
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32'))
        self.sequence = self.sequence[-self.sequence_length:]

        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            self.current_action = self.actions[np.argmax(res)] if np.max(res) >= self.threshold else ''
            image = self.prob_viz(res, image)
            try:
                landmarks = results.pose_landmarks.landmark
                self.count_reps(image, landmarks, mp_pose)
            except: pass
            cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
            cv2.putText(image, 'curl ' + str(self.curl_counter), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, 'press ' + str(self.press_counter), (240,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, 'squat ' + str(self.squat_counter), (490,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ Stream Video ------------------ #
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(
    key="AI trainer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
