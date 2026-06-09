import cv2
import mediapipe as mp
import numpy as np

# MediaPipe face mesh connection definitions
mp_face_mesh = mp.solutions.face_mesh

# Define indices for left eye, right eye, and mouth as specified in the guide
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 81, 13, 311, 308, 402, 14, 178]

# 6 landmarks used for head pose estimation (solvePnP)
# 1: Nose tip
# 152: Chin (sometimes 199 is used, we'll configure it to match the guide's 199 or 152. Guide specifies: 1, 33, 61, 199, 263, 291)
POSE_INDICES = [1, 33, 61, 199, 263, 291]

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def find_face_landmarks(self, frame):
        """
        Process the frame and extract face landmarks in normalized coordinate format.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    @staticmethod
    def get_pixel_landmarks(face_landmarks, frame_shape):
        """
        Converts normalized landmarks to pixel coordinates (x, y) based on frame dimensions.
        """
        h, w, _ = frame_shape
        landmarks = []
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))
        return landmarks

    def close(self):
        self.face_mesh.close()
