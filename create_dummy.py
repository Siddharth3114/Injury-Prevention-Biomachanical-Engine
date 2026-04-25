import cv2
import numpy as np

# Create a 1-second 30fps black video with a moving white square
out = cv2.VideoWriter('dummy_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (i*10, 200), (i*10+50, 250), (255, 255, 255), -1)
    out.write(frame)
out.release()
print("Created dummy_test.mp4")
