import numpy as np
import joblib
from __init__  import  features_from_base_point, find_points_dividing_line
import matplotlib.path as mplPath
import cv2
from ultralytics import YOLO
from collections import defaultdict

"""
This is the data to test the trained model
"""
# Load the model
try:
    mode_predicr = joblib.load('LogisticRegression.pkl')
except FileNotFoundError:
    print("Error: The file 'randomforestclassifier.pkl' was not found.")
    raise
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    raise

# Prepare new data for prediction
width = 1280
height = 720
fix_left_line = ((int(width * 0.3), int(height * 0.6)), (int(width * 0.2), int(height * 0.85)))
fix_right_line = ((int(width * 0.7), int(height * 0.6)), (int(width * 0.8), int(height * 0.85)))

right_mid = find_points_dividing_line(fix_right_line[0], fix_right_line[1], 1 / 2)
left_mid = find_points_dividing_line(fix_left_line[0], fix_left_line[1], 1 / 2)

one_zone = mplPath.Path([fix_left_line[0], fix_right_line[0], fix_right_line[1], fix_left_line[1]])

model = YOLO("yolov8n.pt")
fps=30
frame_interval_human = 0.3 * fps
VIDEO_Path='test/VID_350.MOV'
track_history = defaultdict(lambda: [])
feature_mapping=defaultdict(lambda:[])
frame_count = 0
frame_count_human = 0

cap = cv2.VideoCapture(VIDEO_Path)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (width, height))
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')  # adding tracker cann lower delay =]
        boxes = results[0].boxes.xywh.cpu().numpy()  # this is for the center of the box
        boxes2 = results[0].boxes.xyxy.cpu().numpy()  # this is the actual box
        class_ids = results[0].boxes.cls.cpu().numpy()
        annotated_frame = frame

        # plot the detected lane
        cv2.line(annotated_frame, fix_left_line[0], fix_left_line[1], [255, 255, 255], 5)  # white
        cv2.line(annotated_frame, fix_right_line[0], fix_right_line[1], [0, 255, 0], 5)  # green

        if (results[0].boxes.id == None):
            track_ids = []
        else:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        filtered = [(x, k, y, z) for x, k, y, z in zip(boxes, boxes2, track_ids, class_ids) if z == 0]
        for box, box2, track_id, class_id in filtered:
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, box2[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box
            track = track_history[track_id]
            track.append((x, y))  # x, y center point
            if len(track) > (frame_interval_human):  # retain 90 tracks for 90 frames
                track.pop(0)

            start_point = track[0]
            end_point = track[len(track) - 1]
            if (len(track) > 1):
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                if frame_count_human % frame_interval_human == 0:

                    # (angle, color_code, color,use_dir,start_point,end_point)
                    moving_feature = features_from_base_point(one_zone, start_point, end_point)
                    feature = feature_mapping[track_id]
                    feature.append((frame_count_human, moving_feature))

                    new_data = np.array([[moving_feature[0], moving_feature[1], moving_feature[3],  moving_feature[8]]])
                    prediction = mode_predicr.predict(new_data)
                    if prediction ==1:
                       cv2.putText(annotated_frame,'caution',(int(start_point[0]),int(start_point[1])),cv2.FONT_HERSHEY_SIMPLEX ,1,(0, 255, 255),2,cv2.LINE_4)
                    elif prediction==15:
                        cv2.putText(annotated_frame, 'NO!!!!!!', (int(start_point[0]),int(start_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_4)
                    else:
                        cv2.putText(annotated_frame,'safe',(int(start_point[0]),int(start_point[1])),cv2.FONT_HERSHEY_SIMPLEX ,1,(0, 255, 0),2,cv2.LINE_4)


                cv2.line(annotated_frame, (int(start_point[0]), int(start_point[1])),
                         (int(end_point[0]), int(end_point[1])),
                         color=(255, 255, 255), thickness=3)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
cap.release()
cv2.destroyAllWindows()


total=0
match=0
for f in feature_mapping:
    kkk =feature_mapping[f]
    prediff = 0
    for k in kkk:
        if (prediff + 2 < len(kkk)):
            post = prediff + 2
            kh = prediff
            maxrisk=k[1][1]
            if maxrisk !=15:
                while kh<=post:
                    if kkk[kh][1][1]>maxrisk:
                        maxrisk=kkk[kh][1][1]
                    kh+=1
            new_data = np.array([[k[1][0], k[1][1], k[1][3], k[1][8]]])
            prediction = mode_predicr.predict(new_data)
            if prediction ==maxrisk:
                match+=1
            total+=1
        prediff+=1
print(f'Accuracy: {match/total:.4f}')
