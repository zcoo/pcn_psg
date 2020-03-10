
import numpy as np
import open3d
import skimage
import skimage.io
import skimage.transform
import numpy as np


def read_pcd(filename):
    pcd = open3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.io.Vector3dVector(points)
    open3d.io.write_point_cloud(filename, pcd)



# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img[:, :, :3]
    img = img / 255.0
    img = img.reshape((1, 224, 224, 3))
    return img

