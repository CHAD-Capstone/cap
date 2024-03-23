import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

def calibrate(checker_size_mm: int, cal_img_dir: Path, pattern_shape: Tuple[int, int], out_file: Path, cal_img_file_type: str = 'jpg', fisheye: bool = False, max_num_imgs: int = None, visualize: bool = False):
    cal_img_files = sorted(cal_img_dir.glob(f'*.{cal_img_file_type}'))

    # FROM: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_shape[0]*pattern_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_shape[0],0:pattern_shape[1]].T.reshape(-1,2)
    objp *= checker_size_mm # 20mm square size
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    if max_num_imgs is not None:
        cal_img_files = cal_img_files[:max_num_imgs]

    if fisheye:
        print("Using fisheye calibration.")

    for i, fname in enumerate(cal_img_files):
        print(f"Processing image {i+1}/{len(cal_img_files)}: {fname.absolute().as_posix()}")
        img = cv2.imread(fname.absolute().as_posix())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (pattern_shape[0],pattern_shape[1]), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (pattern_shape[0],pattern_shape[1]), corners2, ret)
            if visualize:
                cv2.imshow('img', img)
                cv2.waitKey(500)
        else:
            print(f"Could not find corners in image {fname.absolute().as_posix()}")
    cv2.destroyAllWindows()

    if fisheye:
        print("Calibrating fisheye camera.")
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        calibration_flags = cv2.fisheye.CALIB_FIX_SKEW
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        objpoints = np.expand_dims(np.asarray(objpoints), -2)  # From https://github.com/opencv/opencv/issues/9150
        imgpoints = np.array(imgpoints)
        print(objpoints.shape, imgpoints.shape)
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    else:
        print("Calibrating camera.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration complete. RMS error: {ret}")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    # Undistort the cal images
    undistorted_dir = cal_img_dir / "undistorted"
    undistorted_dir.mkdir(exist_ok=True)

    # Write the calibration data to disk
    np.savez(out_file.absolute().as_posix(), mtx=mtx, dist=dist)

    for i, fname in enumerate(cal_img_files):
        if fisheye:
            print(f"Undistorting fisheye image {i+1}/{len(cal_img_files)}: {fname.absolute().as_posix()}")
        else:
            print(f"Undistorting image {i+1}/{len(cal_img_files)}: {fname.absolute().as_posix()}")
        img = cv2.imread(fname.absolute().as_posix())
        h,  w = img.shape[:2]
        if fisheye:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.fisheye.undistortImage(img, mtx, dist, None, newcameramtx)
        else:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(str(undistorted_dir / fname.name), dst)

    # # Project the expected position of the calibration back into the images to visualize the calibration
    # reprojected_dir = cal_img_dir / "reprojected"
    # reprojected_dir.mkdir(exist_ok=True)

    # for i, fname in enumerate(cal_img_files):
    #     if fisheye:
    #         print(f"Reprojecting fisheye image {i+1}/{len(cal_img_files)}: {fname.absolute().as_posix()}")
    #     else:
    #         print(f"Reprojecting image {i+1}/{len(cal_img_files)}: {fname.absolute().as_posix()}")
        
    #     rvec = rvecs[i]
    #     tvec = tvecs[i]

    #     img = cv2.imread(fname.absolute().as_posix())
    #     h,  w = img.shape[:2]

    #     # Project the 4 corners of the board back into the image
    #     axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0]]).reshape(-1,3)
    #     imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    #     imgpts = np.int32(imgpts).reshape(-1,2)
    #     img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), 3)

    #     cv2.imwrite(str(reprojected_dir / fname.name), img)

    return mtx, dist

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-img-dir', type=str, required=True, help='Directory containing calibration images.')
    parser.add_argument('--pattern-shape', type=int, nargs=2, required=True, help='Number of inner corners in the calibration pattern.')
    parser.add_argument('--out-file', type=str, required=True, help='Path to save the calibration data.')
    parser.add_argument('--cal-img-file-type', type=str, default='jpg', help='File type of calibration images.')
    parser.add_argument('--checker-size-mm', type=int, default=20, help='Size of the checkerboard squares in mm.')
    parser.add_argument('--fish-eye', action='store_true', help='Use fish-eye calibration.')
    parser.add_argument('--max-num-imgs', type=int, help='Maximum number of images to use for calibration.')
    parser.add_argument('--visualize', action='store_true', help='Display the detected corners.')
    args = parser.parse_args()

    cal_img_dir = Path(args.cal_img_dir)
    assert cal_img_dir.exists(), f"Calibration image directory does not exist: {cal_img_dir.absolute().as_posix()}"
    pattern_shape = tuple(args.pattern_shape)
    out_file = Path(args.out_file)
    if out_file.exists():
        choice = input(f"Output file already exists: {out_file.absolute().as_posix()}. Overwrite? (y/n) ")
        if choice.lower() != 'y':
            exit(0)
    cal_img_file_type = args.cal_img_file_type
    fisheye = args.fish_eye
    max_num_imgs = args.max_num_imgs
    visualize = args.visualize

    calibrate(args.checker_size_mm, cal_img_dir, pattern_shape, out_file, cal_img_file_type, fisheye, max_num_imgs, visualize)