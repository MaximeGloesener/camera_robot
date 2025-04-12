import gxipy as gx
import cv2
import numpy as np
from ultralytics import YOLO


def get_frame(cam, gamma_lut, contrast_lut, color_correction):
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        return None

    raw_image = raw_image.convert("RGB")
    raw_image.image_improvement(color_correction, contrast_lut, gamma_lut)
    return raw_image.get_numpy_array()  # This is RGB


def main():
    print("\n--- GX Camera + YOLO Inference ---\n")
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

    cam.data_stream[0].set_acquisition_buffer_number(1)
    cam.stream_on()

    model = YOLO("seg_planche_small.pt")  # Replace with your own model if needed

    print("Camera streaming started. Press 'q' to quit.")

    while True:
        image_rgb = get_frame(cam, gamma_lut, contrast_lut, color_correction)
        if image_rgb is None:
            print("Failed to retrieve image.")
            continue
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = model(image_rgb, verbose=False)  # RGB input

        annotated_rgb = results[0].plot()
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLOv8 Inference", annotated_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
