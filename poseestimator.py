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

        world_landmarks = mp_result.pose_world_landmarks.landmark

        landmark_points = []
        for landmark in world_landmarks:
            landmark_points.append([landmark.x, landmark.y, landmark.z])

        return np.array(landmark_points)