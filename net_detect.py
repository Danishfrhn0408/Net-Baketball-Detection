import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load trained YOLOv8 model
model = YOLO("runs/detect/train8/weights/best.pt")

# Load MiDaS lightweight model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
midas.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Get the center of the frame
frame_width = 640
frame_height = 480
frame_center = (frame_width // 2, frame_height // 2)

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Skipping empty frame.")
        continue

    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLOv8 detection
    results = model(frame_resized)

    # Resize for MiDaS and estimate depth
    try:
        input_midas = cv2.resize(frame_resized, (256, 256))  # smaller input speeds up MiDaS
        input_rgb = cv2.cvtColor(input_midas, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transforms(input_rgb).to(device)

        with torch.no_grad():
            depth = midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(frame_height, frame_width),  # resize back to match original frame size
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            depth_np = depth.cpu().numpy()
    except Exception as e:
        print("MiDaS depth estimation failed:", e)
        continue

    # Draw detection and depth
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box[:4])
            label = names[int(cls)]

            # Draw bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get depth at center of bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cy < depth_np.shape[0] and 0 <= cx < depth_np.shape[1]:
                depth_val = depth_np[cy, cx]
                cv2.putText(frame_resized, f'{label} | {depth_val:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(frame_resized, (cx, cy), 6, (0, 0, 255), -1)

                # Calculate angle to frame center
                delta_x = cx - frame_center[0]
                delta_y = cy - frame_center[1]
                angle = np.arctan2(delta_y, delta_x)  # angle in radians
                angle_degrees = np.degrees(angle)  # convert to degrees

                # Display angle
                cv2.putText(frame_resized, f'Angle: {angle_degrees:.2f}°', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw bounding circle if label is "board"
                if label == "board":
                    radius = max((x2 - x1), (y2 - y1)) // 4
                    cv2.circle(frame_resized, (cx, cy), radius, (255, 0, 0), 2)
            else:
                cv2.putText(frame_resized, f'{label} | Depth N/A', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("YOLOv8 + Depth + Circle", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# python c:\Users\User\farhan\roboccon\net_detect.py