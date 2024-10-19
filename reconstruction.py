import cv2
import numpy as np
import open3d as o3d
import poseestimator

def image_path_to_pointcloud(path: str, 
                               intrinsics:o3d.camera.PinholeCameraIntrinsic, 
                               pitch_angle:float, 
                               yaw_angle:float)-> o3d.geometry.PointCloud:
    """This function takes in the path to the image and the intrinsics of the camera and returns a point cloud."""
    raw_data = np.load(path)
    color_image = raw_data[:, :, :3]
    color_image = color_image[:,:,::-1].astype(np.uint8)
    depth_image = (raw_data[:, :, 3]).astype(np.float32)
    if path == 'new_image.npy':
        depth_image = depth_image * 1000.0
    depth_image[depth_image == 0] = 0.1 ## Added to get a pointcloud point for each pixel.
    depth_image[depth_image > 40000.0] = 0.1 ## Added to get a pointcloud point for each pixel.

    color_image = np.ascontiguousarray(color_image)
    depth_image = np.ascontiguousarray(depth_image)

    depth_o3d = o3d.geometry.Image(depth_image)
    color_od3 = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_od3, 
                                                                    depth_o3d, 
                                                                    depth_scale = 10000,
                                                                    convert_rgb_to_intensity = False,
                                                                    depth_trunc = 10000.0)

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

def image_to_pointcloud(image: np.array, 
                               intrinsics:o3d.camera.PinholeCameraIntrinsic, 
                               pitch_angle:float, 
                               yaw_angle:float)-> o3d.geometry.PointCloud:
    """This function takes in the path to the image and the intrinsics of the camera and returns a point cloud."""
    color_image = image[:, :, :3]
    color_image = color_image[:,:,::-1].astype(np.uint8)
    depth_image = (image[:, :, 3]).astype(np.float32)

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

def pixel_to_pointcloud_index(pixel: tuple, image_shape:tuple)-> int:
    """This function takes in the pixel and returns the index of the point in the point cloud."""
    return pixel[0]*image_shape[1] + pixel[1]

def points_to_arrow(start: np.array, end: np.array)-> o3d.geometry.TriangleMesh:
    """This function takes in the start and end points and returns an arrow."""
    arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, 
                                                        cone_radius=0.05, 
                                                        cylinder_height=1.5, 
                                                        cone_height=0.5)
    arrow_mesh.paint_uniform_color([1, 0, 0])
    
    # arrow_mesh.rotate(rotation_matrix, center=(0, 0, 0))

    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length

    axis_of_rotation = np.cross([0, 0, 1], direction)
    rotation_angle = np.arccos(np.clip(np.dot([0,0,1], direction), -1.0, 1.0))
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_of_rotation*rotation_angle)
    
    arrow_mesh.rotate(rotation_matrix, center=(0, 0, 0))
    arrow_mesh.translate(start)

    return arrow_mesh

def get_distance_between_point_and_line(point: np.array, line_start: np.array, line_end: np.array)-> float:
    """This function takes in a point and a line and returns the distance between the point and the line."""
    return np.linalg.norm(np.cross(line_end - line_start, line_start - point))/np.linalg.norm(line_end - line_start)

def points_away_from_points(point:np.array, line_start:np.array, line_end:np.array)-> bool:
    """This function takes in a point and a line and returns if the point is away from the line."""
    return np.dot(line_end - line_start, point - line_end) > 0

intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=386.837, fy=386.867, cx=314.029, cy=237.872
)

### Generate items and add them to the visualization
coordinate_frame = create_coordinate_frame(0.5, [0,0,0])
cube = create_refrence_cube(1, [2, 2, 2])

intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=386.837, fy=386.867, cx=314.029, cy=237.872)
image_0 = image_path_to_pointcloud('./media/lab_data/mohit_image_9.npy', intrinsics, 20, 0)
# image_0 = image_path_to_pointcloud('new_image.npy', intrinsics, 20, 0)
image_1 = image_path_to_pointcloud('./media/lab_data/mohit_image_1.npy', intrinsics, 20, 60)
image_2 = image_path_to_pointcloud('./media/lab_data/mohit_image_2.npy', intrinsics, 20, 120)
image_3 = image_path_to_pointcloud('./media/lab_data/mohit_image_3.npy', intrinsics, 20, 180)
image_4 = image_path_to_pointcloud('./media/lab_data/mohit_image_4.npy', intrinsics, 20, 240)
image_5 = image_path_to_pointcloud('./media/lab_data/mohit_image_5.npy', intrinsics, 20, 300)

# Get Landmarks on Human.
landmarks, hip_pixels = poseestimator.landmark_pointcloud_from_image('./media/lab_data/mohit_image_9.npy')
landmarks = poseestimator.rotate_landmarks(landmarks, 'x', 20)
landmarks = poseestimator.rotate_landmarks(landmarks, 'x', 90)

## This is naive impelementation of getting the hip position.
print("This is naive implementation of getting the hip position. Implement a better one.")
hip_position = image_0.points[pixel_to_pointcloud_index(hip_pixels, (480, 640))]
landmarks = poseestimator.translate_landmarks(landmarks, hip_position)

## Show the arrow from the human being.
arrow = points_to_arrow(landmarks[12], landmarks[16])

# ## Filter out points that are close to the line.
# filtered_points = []
# for image in [image_0, image_1, image_2, image_3, image_4, image_5]:
#     for point in image.points:
#         if get_distance_between_point_and_line(point, landmarks[12], landmarks[16]) < 0.2:
#             filtered_points.append(point)
# print("Number of points filtered: ", len(filtered_points))

# ## Apply the filter for points to be outwards of endpoitns.
# filtered_points = [point for point in filtered_points if points_away_from_points(point, landmarks[12], landmarks[16])]

# close_points = o3d.geometry.PointCloud()
# close_points.points = o3d.utility.Vector3dVector(filtered_points)
# close_points.paint_uniform_color([0, 1, 0])

## For showing the landmarks in big size.
landmark_points = []
for landmark in landmarks:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(landmark)
    landmark_points.append(sphere)

geometries = [image_0, image_1, image_2, image_3, image_4, image_5, cube, coordinate_frame, arrow]#, close_points]
geometries.extend(landmark_points)

o3d.visualization.draw_geometries(geometries, window_name='Open3D')

