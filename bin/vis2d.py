import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

CALIBRATION_FILE = "../kittidata/{0}_sync/calib_velo_to_cam.txt"
RECTIFYING_FILE = "../kittidata/{0}_sync/calib_cam_to_cam.txt"
PCD_FILE = "../kittidata/{0}_sync/velodyne_points/pcd_data/{1}.pcd"
IMG_FILE = "../kittidata/{0}_sync/image_00/data/{1}.png"

def load_file(pcd_file, calibration_file, rectifying_file):
    with open(pcd_file, "r") as f:
        s = f.read()
    s = s.splitlines()[11:]
    s = [[float(i) for i in l.split()] for l in s]
    data = np.array(s)
    with open(calibration_file, "r") as f:
        c = f.read()
    c = c.splitlines()

    R_camvelo, t_camvelo = c[1], c[2]
    R_camvelo = np.array([float(i) for i in R_camvelo[3:].split()])\
                  .reshape((3, 3))
    t_camvelo = np.array([float(i) for i in t_camvelo[3:].split()])\
                  .reshape((1, 3))

    T_camvelo = np.eye(4)
    T_camvelo[:3, :3] = R_camvelo
    T_camvelo[:3, 3] = t_camvelo

    with open(rectifying_file, "r") as f:
        s = f.read()
    s = s.splitlines()
    R_rect0, P_rect0 = s[8], s[9]
    R_rect0 = np.array([float(i) for i in R_rect0[10:].split()])\
              .reshape((3, 3))
    t = np.eye(4)
    t[:3, :3] = R_rect0

    R_rect0 = t
    P_rect0 = np.array([float(i) for i in P_rect0[10:].split()])\
              .reshape((3, 4))
    return data, T_camvelo, R_rect0, P_rect0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Visualize")
    parser.add_argument('dataset', type=int,
                        help="Dataset choice")
    parser.add_argument('pcd', type=int,
                        help="pcd choice")

    args = parser.parse_args()
    dataset, pcd = args.dataset, args.pcd

    pcd_file         = PCD_FILE.format(str(dataset).zfill(4), str(pcd).zfill(10))
    calibration_file = CALIBRATION_FILE.format(str(dataset).zfill(4))
    rectifying_file  = RECTIFYING_FILE.format(str(dataset).zfill(4))
    img_file         = IMG_FILE.format(str(dataset).zfill(4), str(pcd).zfill(10))

    if not os.path.exists(pcd_file):
        print("PCD {} not found".format(pcd))
        exit()

    if not os.path.exists(calibration_file):
        print("Calibration files not found for dataset {}".format(dataset))
        exit()
    if not os.path.exists(img_file):
        print(img_file)
        print("Image file not found for {}".format(pcd))
        exit()

    data, T, R, P = load_file(pcd_file, calibration_file, rectifying_file)
    data[:][4] = 1
    img = mpimg.imread(img_file)
    print(P)
    print(R)
    print(T)
    cam_data = data.T
    print(cam_data[:10])

    cam_data = np.dot(T, cam_data)
    cam_data = np.dot(R, cam_data)
    #cam_data = np.dot(P.T, cam_data)

    cam_data = np.array([r for r in cam_data.T if r[2] > 0]).T
    plt.scatter(cam_data[0], -cam_data[1], c=cam_data[2])
    plt.imshow(img, extent=[-30, 30, -10, 10], cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
