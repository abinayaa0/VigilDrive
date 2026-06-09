import cv2
import numpy as np

# Generic 3D model points of a human face (in mm or generic unit coordinate system)
# Origin is at the nose tip
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (idx 1)
    (0.0, -330.0, -65.0),        # Chin (idx 199)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (idx 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (idx 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (idx 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (idx 291)
], dtype=np.float32)

def estimate_head_pose(landmarks, frame_shape):
    """
    Estimate head pose angles (pitch, yaw, roll) using cv2.solvePnP.
    landmarks: list of (x, y) coordinates of all 468/478 FaceMesh points.
    frame_shape: tuple of (height, width, channels) of the frame.
    """
    h, w, _ = frame_shape
    
    # 2D image points of the selected facial features
    # Indices map to Pose landmark points: Nose(1), Left Eye Corner(33), Left Mouth Corner(61), Chin(199), Right Eye Corner(263), Right Mouth Corner(291)
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[199],  # Chin
        landmarks[33],   # Left eye outer corner
        landmarks[263],  # Right eye outer corner
        landmarks[61],   # Left mouth corner
        landmarks[291]   # Right mouth corner
    ], dtype=np.float32)
    
    # Camera intrinsic matrix approximation
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Assuming zero lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None, None, None
        
    # Get rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Decompose projection matrix to extract Euler angles
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    
    # Extract pitch, yaw, roll in degrees
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    # Adjust orientation depending on camera mirror/sensor alignments
    # Standard output: pitch (positive is down, negative is up), yaw (positive is right, negative is left), roll (positive is tilt right, negative tilt left)
    # Let's align to: Pitch (look up/down), Yaw (look left/right), Roll (tilt)
    # We can normalize them slightly:
    pitch = np.clip(pitch, -180.0, 180.0)
    yaw = np.clip(yaw, -180.0, 180.0)
    roll = np.clip(roll, -180.0, 180.0)
    
    # Projection matrix points to draw 3D axis at nose tip
    axis_length = 150.0
    axis_points_3d = np.array([
        (axis_length, 0, 0),  # X axis (Red) - Left/Right
        (0, axis_length, 0),  # Y axis (Green) - Down/Up
        (0, 0, axis_length)   # Z axis (Blue) - In/Out
    ], dtype=np.float32)
    
    # Project 3D points to 2D image plane
    nose_tip_2d = image_points[0]
    projected_points_2d, _ = cv2.projectPoints(
        axis_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    
    # Flatten the projected points array
    projected_points_2d = projected_points_2d.reshape(-1, 2).astype(int)
    
    return pitch, yaw, roll, nose_tip_2d.astype(int), projected_points_2d
