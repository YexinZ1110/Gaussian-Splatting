import cv2
import numpy as np
import glob
from tqdm import tqdm

def calibrate(imgs_path): 
    chessboard_size = (9, 6)  # internal corner number of chessboard
    square_size = 1.0  # real size of square
    
    # world coordinate(z=0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # world coordinate
    imgpoints = []  # image coordinate
    
    images = glob.glob(imgs_path+'/*.jpg')
    # print("images: ", images)
    for image_path in tqdm(images, desc="Processing Images"):
        img = cv2.imread(image_path)
        # cv2.imshow('img', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # corners detection
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # print("ret: ",ret)
        # print("corners: ",corners)
    
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            # cv2.namedWindow('Chessboard Corners', 0)
            # cv2.resizeWindow('Chessboard Corners', 600, 500)
            # cv2.imshow('Chessboard Corners', img)
            # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()
    print("objpoints: ", objpoints)
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    np.savez('../camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

def para_to_str(path):
    file = np.load(path)
    mtx = file['mtx']
    dist = file['dist']
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2, k3 = dist[0]
    k4, k5, k6 = 0, 0, 0

    params_str = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2},{k3},{k4},{k5},{k6}"
    return params_str
 

test = False
if test:
    for image_path in images:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
    
        # calculate new camera matrix and undistort image
        # newcameramtx: new camera matrix
        # roi: effective pixel area
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
        # use new camera matrix undistort method to correct distortion image
        # mtx: original camera matrix
        # dist: distortion coefficients
        # newcameramtx: new camera matrix
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
        # get the valid pixel area of the undistorted image
        x, y, w, h = roi
        # cut the valid pixel area
        dst = dst[y:y + h, x:x + w]
        cv2.namedWindow('Undistorted Image', 0)
        cv2.resizeWindow('Undistorted Image', 600, 500)
        cv2.imshow('Undistorted Image', dst)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    data = np.load('camera_calibration.npz')
    print('test')
    print(data['mtx'])
    print(data['dist'])