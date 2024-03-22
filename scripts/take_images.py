import nanocamera as nano
import cv2
from pathlib import Path
import threading
import time

def start_camera(flip=0, width=1280, height=720):
    # Connect to another CSI camera on the board with ID 1
    camera = nano.Camera(device_id=0, flip=flip, width=width, height=height, debug=False, enforce_fps=True)
    status = camera.hasError()
    codes, has_error = status
    if has_error:
        return False, codes, None
    else:
        return True, None, camera

counting = False
thread = None
def start_counter():
    # Counts up in a thread
    global counting, thread
    counting = True
    cur_count = 0
    def count():
        nonlocal cur_count
        while counting:
            print(cur_count)
            cur_count += 1
            time.sleep(0.1)

    thread = threading.Thread(target=count)
    thread.start()

def stop_counter():
    global counting, thread
    counting = False
    thread.join()
    thread = None

def main(image_path: Path, period=None, flip=0, width=1280, height=720):
    assert image_path.exists(), f"Path {image_path} does not exist"

    cam_success, cam_codes, camera = start_camera(flip=flip, width=width, height=height)
    save_folder = image_path / f"f{flip}_w{width}_h{height}"
    save_folder.mkdir(exist_ok=True)
    if not cam_success:
        print("Failed to initialize camera. Information on camera codes here: https://github.com/thehapyone/NanoCamera?tab=readme-ov-file#errors-and-exceptions-handling")
        print(cam_codes)
        return False

    num_existing_images = len(list(save_folder.glob("*.jpg")))
    print(f"Number of existing images: {num_existing_images}")

    inp = "y"
    input("\n\n\n\nPress enter to start capturing images.")
    if period is not None:
        time.sleep(period)
    # start_counter()
    try:
        while inp.lower() == "y" and camera.isReady():
            print("Capturing image...")
            frame = camera.read()  # Returns a BGR image
            cv2.imwrite(str(save_folder / f"cal_image_{num_existing_images}.jpg"), frame)
            print(f"Image saved. Num taken {num_existing_images+1}")
            num_existing_images += 1
            if period is None:
                inp = input("Capture another image now? (y/n): ")
            else:
                time.sleep(period)
        if not camera.isReady():
            print("Camera is not ready. Exiting...")
            camera.release()
            return False
    except KeyboardInterrupt:
        pass

    camera.release()
    # stop_counter()
    print(f"Number of images captured: {num_existing_images}")
    return True
    

if __name__ == "__main__":
    image_path = Path("/home/rob498/catkin_ws/src/cap/data/test_images")
    main(image_path, period=None)