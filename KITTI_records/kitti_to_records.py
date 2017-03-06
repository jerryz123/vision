#encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.python.platform import gfile
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter

from data import image_shape
from data import get_drive_dir, Calib, get_inds, image_shape

from PIL import Image

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, '../kittidata')

def get_velodyne_points(velodyne_dir, frame):
    points_path = os.path.join(velodyne_dir, "data/%010d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def load_disparity_points(velodyne_dir, frame, color=False, **kwargs):

    calib = Calib(velodyne_dir, color=color, **kwargs)

    # read velodyne points
    points = get_velodyne_points(velodyne_dir, frame)

    # remove all points behind image plane (approximation)
    points = points[points[:, 0] >= 1, :]

    # convert points to each camera
    xyd = calib.velo2disp(points)

    # take only points that fall in the first image
    xyd = calib.filter_disps(xyd)

    return xyd

def test(vel, verbose=False):
    velodyne_dir = os.path.join(data_dir, vel)

    velodyne_bin_dir = os.path.join(velodyne_dir, "data")
    bin_list = os.listdir(velodyne_bin_dir)

    for i in range(len(bin_list)):
        points = get_velodyne_points(velodyne_dir, i)
        # print(("points data type: %s" % type(points)))
        # print((points.shape))
        #
        # # x
        # print(("x max: %f, min: %f" % (np.max(points[:,0]), np.min(points[:,0]))))
        # # y
        # print(("y max: %f, min: %f" % (np.max(points[:,1]), np.min(points[:,1]))))
        # # z
        # print(("z max: %f, min: %f" % (np.max(points[:,2]), np.min(points[:,2]))))

        # reflectance
        #print("r max: %f, min: %f" % (np.max(points[:,3]), np.min(points[:,3])))

        image_array = np.asarray([1224, 368])
        xyd = load_disparity_points(velodyne_dir, i, color=True)
        disp = np.zeros(image_shape, dtype=np.float)
        for x, y, d in np.round(xyd):
            disp[int(y), int(x)] = d
        ones = np.ones(image_shape, dtype=np.float)
        # print("Min depth:", np.min([x for x in np.ravel(disp) if x > 0]), "Max depth:", np.max(disp))
        image = Image.fromarray(np.uint8(((disp/np.max(disp)))*255.0))
        save_dir = os.path.join(velodyne_dir, "depth")
        if not gfile.Exists(save_dir):
            gfile.MakeDirs(save_dir)
        save_path = os.path.join(velodyne_dir, "depth/%010d.png" % i)
        width, height = image.size
        #image.thumbnail((width // 2, height // 2), Image.BILINEAR)
        image.save(save_path)
        # plt.imshow(image)
        # plt.show()

        if verbose:
            newimg = np.zeros(width*height)
            
            x, y, d = xyd.T
            points = np.array([x, y]).T
            grid_x, grid_y = np.mgrid[0:width, 0:height]
            newpoints = []
            for x in range(0, width, 5):
                for y in range(0, height, 5):
                    if np.linalg.norm(points - np.array((x, y)), axis=1).min() > 8:
                        newpoints += [[x, y]]

            newpoints = np.array(newpoints)
            points = np.concatenate((points, newpoints))
            d = np.concatenate((d, np.zeros(len(newpoints))))
            grid = griddata(points, d, (grid_x, grid_y), method="nearest", fill_value=0)
            grid = np.clip(grid, 0, 99999)
            grid = gaussian_filter(grid,2)
            plt.clf()
            plt.imshow(grid.T, cmap="gray")
            plt.colorbar()
            plt.show()
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="createimg")
    parser.add_argument('dataset', type=int,
                        help="Dataset choice")
    parser.add_argument("-v", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()


    args = parser.parse_args()
    dataset = args.dataset
    verbose = args.v

    velodyne_dir = os.path.join(data_dir, "%04d_sync" % dataset, "velodyne_points")
    test(velodyne_dir, verbose)
