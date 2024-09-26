import cv2
import numpy as np
import open3d as o3d
from poseestimator import landmark_pointcloud_from_image

def add_image_to_visualization(path: str, 
                               intrinsics:o3d.camera.PinholeCameraIntrinsic, 
                               pitch_angle:float, 
                               yaw_angle:float)-> o3d.geometry.PointCloud:
    """This function takes in the path to the image and the intrinsics of the camera and returns a point cloud."""
    raw_data = np.load(path)
    color_image = raw_data[:, :, :3]
    color_image = color_image[:,:,::-1].astype(np.uint8)
    depth_image = (raw_data[:, :, 3]).astype(np.float32)

    color_image = np.ascontiguousarray(color_image)
    depth_image = np.ascontiguousarray(depth_image)

    depth_o3d = o3d.geometry.Image(depth_image)
    color_od3 = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_od3, 
                                                                    depth_o3d, 
                                                                    depth_scale = 10000,
                                                                    convert_rgb_to_intensity = False,
                                                                    depth_trunc = 3.0)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    ## First we rotate the point cloud to desired orientation.
    ## Rotate 90 degrees about the x-axis.
    point_cloud.transform([[1, 0, 0, 0],
                        [0, np.cos(-np.pi/2), -np.sin(-np.pi/2), 0],
                        [0, np.sin(-np.pi/2), np.cos(-np.pi/2), 0],
                        [0, 0, 0, 1]])

    ## For Pitch.
    pitch_angle = -np.radians(pitch_angle)
    point_cloud.transform([[1, 0, 0, 0],
                            [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
                            [0, np.sin(pitch_angle), np.cos(pitch_angle), 0],
                            [0, 0, 0, 1]])
    
    ## Rotate about z as yaw angle.
    yaw_angle = np.radians(yaw_angle)
    point_cloud.transform([[np.cos(yaw_angle), -np.sin(yaw_angle), 0, 0],
                            [np.sin(yaw_angle), np.cos(yaw_angle), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    return point_cloud

def create_coordinate_frame(size: float, origin: list)-> o3d.geometry.TriangleMesh:
    """This function creates a coordinate frame."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def create_refrence_cube(size: float, origin:list)-> o3d.geometry.TriangleMesh:
    """This function creates a cube."""
    cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    cube.compute_vertex_normals()
    cube.translate(origin)
    cube.paint_uniform_color([0.9, 0.1, 0.1])
    return cube

intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=386.837, fy=386.867, cx=314.029, cy=237.872
)

### Generate items and add them to the visualization
coordinate_frame = create_coordinate_frame(0.5, [0,0,0])
cube = create_refrence_cube(1, [2, 2, 2])

intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=386.837, fy=386.867, cx=314.029, cy=237.872)
image_0 = add_image_to_visualization('./media/lab_data/mohit_image_9.npy', intrinsics, 20, 0)
image_1 = add_image_to_visualization('./media/lab_data/mohit_image_1.npy', intrinsics, 20, 60)
image_2 = add_image_to_visualization('./media/lab_data/mohit_image_2.npy', intrinsics, 20, 120)
image_3 = add_image_to_visualization('./media/lab_data/mohit_image_3.npy', intrinsics, 20, 180)
image_4 = add_image_to_visualization('./media/lab_data/mohit_image_4.npy', intrinsics, 20, 240)
image_5 = add_image_to_visualization('./media/lab_data/mohit_image_5.npy', intrinsics, 20, 300)

## Get Landmarks on Human.
landmarks = landmark_pointcloud_from_image('./media/lab_data/mohit_image_9.npy')

## For showing the landmarks in big size.
landmark_points = []
for landmark in landmarks:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(landmark)
    landmark_points.append(sphere)

geometries = [image_0, image_1, image_2, image_3, image_4, image_5, cube, coordinate_frame]
geometries.extend(landmark_points)

o3d.visualization.draw_geometries(geometries, window_name='Open3D')

