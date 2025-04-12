import gxipy as gx
import cv2
import numpy as np
from ultralytics import YOLO


def initialize_camera():
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        raise RuntimeError("No camera found.")

    cam = device_manager.open_device_by_index(1)

    if not cam.PixelColorFilter.is_implemented():
        raise RuntimeError("Mono camera not supported.")

    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.ExposureTime.set(10000.0)
    cam.Gain.set(10.0)

    gamma_lut = gx.Utility.get_gamma_lut(cam.GammaParam.get()) if cam.GammaParam.is_readable() else None
    contrast_lut = gx.Utility.get_contrast_lut(cam.ContrastParam.get()) if cam.ContrastParam.is_readable() else None
    color_correction = cam.ColorCorrectionParam.get() if cam.ColorCorrectionParam.is_readable() else 0

    return cam, gamma_lut, contrast_lut, color_correction


def get_frame(cam, gamma_lut, contrast_lut, color_correction):
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        return None

    rgb_image = raw_image.convert("RGB")
    rgb_image.image_improvement(color_correction, contrast_lut, gamma_lut)
    return rgb_image.get_numpy_array()


def draw_grid(frame, rows=4, cols=4, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    cell_height = h // rows
    cell_width = w // cols

    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_height), (w, i * cell_height), color, thickness)
    for j in range(1, cols):
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, h), color, thickness)

    return cell_width, cell_height


def get_zone(x, y, cell_width, cell_height):
    col = x // cell_width
    row = y // cell_height
    return int(row), int(col)


def main():
    print("\n--- GX Camera + YOLO Inference with Grid + Video Saving ---\n")

    output_path = "output_video.avi"
    output_fps = 20

    try:
        cam, gamma_lut, contrast_lut, color_correction = initialize_camera()
        cam.stream_on()

        model = YOLO("yolov8n.pt")

        # Init video writer
        frame = get_frame(cam, gamma_lut, contrast_lut, color_correction)
        if frame is None:
            raise RuntimeError("Failed to get initial frame for video writer.")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), output_fps, (width, height))

        print("Camera streaming started. Press 'q' to quit.")

        while True:
            image_np = get_frame(cam, gamma_lut, contrast_lut, color_correction)
            if image_np is None:
                print("Failed to retrieve image.")
                continue

            # Inference
            results = model(image_np, conf=0.15, iou=0.45, verbose=False)
            annotated_frame = results[0].plot(line_width=3, font_size=0.8)

            # Draw grid
            cell_w, cell_h = draw_grid(annotated_frame, rows=4, cols=4, thickness=2)

            # Add zone info
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                zone_row, zone_col = get_zone(cx, cy, cell_w, cell_h)

                label = f"Zone ({zone_row}, {zone_col})"
                text_pos = (x1 + 5, y1 - 10 if y1 - 10 > 10 else y1 + 15)

                cv2.putText(annotated_frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 6, (255, 0, 0), -1)

            writer.write(annotated_frame)
            cv2.imshow("YOLOv8 Inference + Grid", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cam' in locals():
            cam.stream_off()
            cam.close_device()
        if 'writer' in locals():
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
