import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request


def draw_grid_with_labels(frame, rows=4, cols=4, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    cell_height = h // rows
    cell_width = w // cols

    for row in range(rows):
        for col in range(cols):
            top_left = (col * cell_width, row * cell_height)
            bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)

            # Draw rectangle
            cv2.rectangle(frame, top_left, bottom_right, color, thickness)

            # Put label in top-left corner
            label = f"({row},{col})"
            cv2.putText(
                frame, label,
                (top_left[0] + 5, top_left[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )

    return cell_width, cell_height


def get_zone(x, y, cell_width, cell_height):
    col = x // cell_width
    row = y // cell_height
    return int(row), int(col)


def main():
    # Download sample image if needed
    image_url = "https://ultralytics.com/images/bus.jpg"
    image_path = "sample.jpg"
    urllib.request.urlretrieve(image_url, image_path)

    # Load image
    image_np = cv2.imread(image_path)

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Inference
    results = model(image_np, verbose=False)
    annotated_frame = results[0].plot()

    # Draw grid and labels
    cell_w, cell_h = draw_grid_with_labels(annotated_frame, rows=4, cols=4)

    # Optional: Draw center point of detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        zone_row, zone_col = get_zone(cx, cy, cell_w, cell_h)

        # Visual feedback of zone mapping (optional)
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

    # Display result
    cv2.imshow("YOLOv8 + Grid Zones", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
