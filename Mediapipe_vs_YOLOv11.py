import cv2
import numpy as np
import time
import mediapipe as mp

avg_fps = []
video_writer = None

def draw_text_with_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    padding = 20  # Equal padding on all sides
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rect_width = text_width + 2 * padding
    rect_height = text_height + baseline + 2 * padding
    x, y = pos  # Calculate rectangle position
    rect_x1 = x
    rect_y1 = y - text_height
    rect_x2 = rect_x1 + rect_width
    rect_y2 = rect_y1 + rect_height
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)  # Draw background rectangle
    text_x = rect_x1 + padding  # Center text in rectangle with equal padding
    text_y = rect_y1 + padding + text_height  # Properly centered vertically
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)



def mediapipe_pose(source=0, size=(1440, 810)):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Initialize MediaPipe
    cap = cv2.VideoCapture(source)

    _, _, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("mp-pose.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)

        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)  # Process frame
        process_time = time.time() - start_time

        # Draw pose
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=3),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=3)
        )

        fps = 1.0 / process_time if process_time > 0 else 0  # Calculate FPS
        avg_fps.append(fps)
        fps = sum(avg_fps) / len(avg_fps)
        # Add text with background
        draw_text_with_bg(frame, f'MediaPipe FPS: {int(fps)}', (50, 60), bg_color=(0, 100, 0), font_scale=1.2)
        draw_text_with_bg(frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(0, 100, 0),
                          font_scale=1.2)

        cv2.imshow('MediaPipe Pose', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def yolo_pose(source=0, size=(1440, 810)):
    from ultralytics import YOLO
    yolo_model = YOLO("yolo11n-pose.pt")  # Initialize YOLO

    # YOLO connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    cap = cv2.VideoCapture(source)

    _, _, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("yolo-pose.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)
        start_time = time.time()
        results = yolo_model.predict(frame, verbose=False, device="cpu")  # Process frame
        process_time = time.time() - start_time

        # Draw pose
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()

            for person_kpts in keypoints:
                for start, end in connections:  # Draw connections
                    if start < len(person_kpts) and end < len(person_kpts):
                        pt1, pt2 = person_kpts[start], person_kpts[end]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])), (245, 66, 230), 3)

                for pt in person_kpts:  # Draw keypoints
                    if pt[0] > 0 and pt[1] > 0:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (245, 117, 66), -1)
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)

        fps = 1.0 / process_time if process_time > 0 else 0  # Calculate FPS
        avg_fps.append(fps)
        fps = sum(avg_fps) / len(avg_fps)

        # Add text with background
        draw_text_with_bg(frame, f'YOLO-Pose FPS: {int(fps)}', (50, 60), bg_color=(255, 27, 108), font_scale=1.2)
        draw_text_with_bg(frame, f'Time: {process_time * 1000:.1f}ms', (50, 145), bg_color=(255, 27, 108),
                          font_scale=1.2)

        cv2.imshow('YOLO Pose', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Usage examples:

    # mediapipe_pose("bodypose.mp4")  # MediaPipe with video

    #yolo_pose("bodypose.mp4")      # YOLO with video

     mediapipe_pose(1)              # MediaPipe with webcam

    # yolo_pose(0)                   # YOLO with webcam