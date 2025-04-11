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


def main():
    print("\n--- GX Camera + YOLOv8 Inference ---\n")

    try:
        cam, gamma_lut, contrast_lut, color_correction = initialize_camera()
        cam.stream_on()

        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")  # You can replace with your custom weights

        print("Camera streaming started. Press 'q' to quit.")

        while True:
            image_np = get_frame(cam, gamma_lut, contrast_lut, color_correction)
            if image_np is None:
                print("Failed to retrieve image.")
                continue

            # Run YOLOv8 inference on the frame
            results = model(image_np, verbose=False)

            # Get the annotated frame
            annotated_frame = results[0].plot()

            # Display
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cam' in locals():
            cam.stream_off()
            cam.close_device()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
