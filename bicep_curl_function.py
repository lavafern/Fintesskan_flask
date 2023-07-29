import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
import os

def generate_frames(vidpath):
    IMPORTANT_LMS = [
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "RIGHT_ELBOW",
        "LEFT_ELBOW",
        "RIGHT_WRIST",
        "LEFT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
    ]

    # Generate all columns of the data frame

    HEADERS = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    base_directory = os.path.dirname(os.path.abspath(__file__))
    model_folder=os.path.join(base_directory, 'model')
    input_scaler = os.path.join(model_folder,'input_scaler.pkl')
    dlmodel=os.path.join(model_folder,'savedmodel')

    # Load input scaler
    with open(input_scaler, "rb") as f:
        input_scaler = pickle.load(f)

    # load machine learning model
    DL_model = load_model(dlmodel)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    fil=vidpath
    cap=cv2.VideoCapture(fil)
    visibility_threshold=0.65
    POSTURE_ERROR_THRESHOLD = 0.70
    posture = 0
    left_counter = 0 
    right_counter = 0
    left_stage = None
    right_stage= None
    posture = None
    predicted_class = None
    prediction_probability = None
    upper_hand_pose=None
    
    def rescale_frame(frame, percent=50):
        '''
        Rescale a frame to a certain percentage compare to its original frame
        '''
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    
    
    def extract_important_keypoints(results, important_landmarks: list) -> list:
        '''
        Extract important keypoints from mediapipe pose detection
        '''
        landmarks = results.pose_landmarks.landmark

        data = []
        for lm in important_landmarks:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

        return np.array(data).flatten().tolist()

    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 
    
    def counter_logic(angle,stage,counter):
        if angle>120:
            stage='down'
        if angle  < 90 and stage =='down':
            stage='up'
            counter +=1
            print(counter)
        return stage, counter
    
    def loose_hand_logic(angle,pose):
        if angle>40:
            if pose == None:
                pose='error'
            
            
            return pose
    
    def draw_rectangle(frame_height, frame_width):
        # Calculate the coordinates of the rectangle based on the frame size
        rect_start_point = (0, 0)  # Adjust the percentages as needed
        rect_end_point = (frame_width, int(0.1*frame_height))   # Adjust the percentages as needed
        # Draw the rectangle on the image
        color = (245, 117, 16)  # BGR color (orange in this case)
        thickness = -1  # Negative thickness means the rectangle will be filled

        return rect_start_point, rect_end_point, color, thickness
        

    
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            print(frame.shape[0])
            
            if frame.shape[0]>1000:
            
                frame=rescale_frame(frame, percent=50)
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            )
            
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # Calculate angle
                if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > visibility_threshold and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > visibility_threshold and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > visibility_threshold:
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    left_upper_hand_angle = calculate_angle(left_hip,left_shoulder,left_elbow)
                else:
                    left_angle = None

                if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > visibility_threshold and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility > visibility_threshold and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > visibility_threshold:
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    right_upper_hand_angle = calculate_angle(right_hip,right_shoulder,right_elbow)
                else:
                    right_angle = None




                if left_angle is not None:
                    left_stage, left_counter = counter_logic(left_angle, left_stage, left_counter)
                    upper_hand_pose = loose_hand_logic(left_upper_hand_angle,upper_hand_pose)

                if right_angle is not None:   
                    right_stage, right_counter = counter_logic(right_angle, right_stage, right_counter)
                    upper_hand_pose = loose_hand_logic(right_upper_hand_angle,upper_hand_pose)
                    
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row, ], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))
                #Make prediction and its probability
                prediction = DL_model.predict(X)
                predicted_class = np.argmax(prediction, axis=1)[0]
                prediction_probability = round(max(prediction.tolist()[0]), 2)
                if prediction_probability >= POSTURE_ERROR_THRESHOLD:
                    posture = predicted_class
            
            
            except:
                
                pass
            
            
            
            rect_start_point, rect_end_point, color, thickness=draw_rectangle(frame.shape[0],frame.shape[1])
            
            cv2.rectangle(image, rect_start_point,rect_end_point, color, thickness)

            # Rep data
            cv2.putText(image, 'L.REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_counter), 
                        (20,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


            # Rep data
            cv2.putText(image, 'R.REPS', (90,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_counter), 
                        (90,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(image, "POSTURE ", (200,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if predicted_class==0:
                cv2.putText(image, str("CORRECT"), (200,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str("WRONG"), (200,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "L.A.ERROR", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if upper_hand_pose==None:

                cv2.putText(image, str(upper_hand_pose), (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            else:
               cv2.putText(image, str(upper_hand_pose), (380, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) 
               
               
            ret,buffer=cv2.imencode('.jpg',image)
            image=buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')