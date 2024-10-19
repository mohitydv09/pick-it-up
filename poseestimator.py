import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def landmark_pointcloud_from_image(path:str):
    """This function takes in the path to the image and returns a point cloud of the landmarks."""
    color_image = np.load(path)[:,:,:3].astype(np.uint8)
    color_image = color_image[:,:,::-1]

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        mp_result = pose.process(color_image)

        hip_pixel = get_hip_pixel(mp_result, color_image.shape)

        world_landmarks = mp_result.pose_world_landmarks.landmark

        landmark_points = []
        for landmark in world_landmarks:
            landmark_points.append([landmark.x, landmark.y, landmark.z])

        return np.array(landmark_points), hip_pixel

def get_hip_pixel(landmarks_2d: mp.solutions.pose.PoseLandmark, image_shape: tuple):
    left_hip_2d_landmark = landmarks_2d.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip_2d_landmark = landmarks_2d.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    x_pixel = int((left_hip_2d_landmark.x + right_hip_2d_landmark.x)/2 * image_shape[1])
    y_pixel = int((left_hip_2d_landmark.y + right_hip_2d_landmark.y)/2 * image_shape[0])
    return y_pixel, x_pixel
    

def rotate_landmarks(landmarks: np.array, axis:str, angle:float):
    """Rotate the landmarks about the axis by the angle."""
    angle = np.radians(angle)
    rotation_matrix = np.eye(3)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                     [0, np.cos(angle), -np.sin(angle)],
                                     [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                     [0, 1, 0],
                                     [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0],
                                     [0, 0, 1]])
        
    return landmarks @ rotation_matrix

def translate_landmarks(landmarks: np.array, translation: np.array):
    """Translate the landmarks by the translation vector."""
    return landmarks + translation