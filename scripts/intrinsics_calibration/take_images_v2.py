import cv2
#from nanocamera.NanoCam import Camera
import nanocamera as nano
from pathlib import Path

if __name__ == '__main__':
    # Create the Camera instance
    image_path = Path("/home/rob498/catkin_ws/src/cap/data/cal_2")
    flip = 0
    width = 1280
    height = 720

    camera = nano.Camera(flip=flip, width=width, height=height, fps=30)
    print('CSI Camera ready? - ', camera.isReady())

    assert image_path.exists(), f"Path {image_path} does not exist"

    save_folder = image_path / f"f{flip}_w{width}_h{height}"
    save_folder.mkdir(exist_ok=True)

    num_existing_images = len(list(save_folder.glob("*.jpg")))
    print(f"Number of existing images: {num_existing_images}")

    while camera.isReady():
        try:
            # read the camera image
            frame = camera.read()
            # display the frame
            cv2.imshow("Video Frame", frame)

            # check for the 'q' key to quit the program
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # check for the 's' key to save a snapshot
            elif key == ord('s'):
                # save the current frame as an image file
                out_file = str(save_folder / f"cal_image_{num_existing_images}.jpg")
                cv2.imwrite(out_file, frame)
                print(f"Snapshot taken and saved as {out_file}")
                num_existing_images += 1
        except KeyboardInterrupt:
            break

    # close the camera instance
    camera.release()

    # remove camera object
    del camera