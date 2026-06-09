import cv2
import numpy as np
from landmarks import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES

def draw_hud(frame, metrics, states, fps, head_pose):
    """
    Renders telemetry data and alert flags on the video frame.
    """
    h, w, _ = frame.shape
    pitch, yaw, roll = head_pose
    
    # 1. Draw semi-transparent background card for telemetry text (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 240), (20, 20, 20), -1)
    # Apply alpha blend to overlay card
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Colors
    c_white = (255, 255, 255)
    c_green = (50, 255, 50)
    c_red = (50, 50, 255)
    c_yellow = (0, 255, 255)
    c_blue = (255, 100, 50)
    
    # Telemetry HUD text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"VigilDrive DMS v1.0", (20, 35), font, 0.6, c_yellow, 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 65), font, 0.5, c_white, 1)
    
    # EAR
    ear_color = c_red if states["eyes_closed"] else c_green
    cv2.putText(frame, f"EAR: {metrics['ear']:.2f}", (20, 95), font, 0.5, ear_color, 1)
    cv2.putText(frame, f"Blinks: {states['blink_count']}", (180, 95), font, 0.5, c_white, 1)
    
    # MAR
    mar_color = c_red if states["yawning"] else c_green
    cv2.putText(frame, f"MAR: {metrics['mar']:.2f}", (20, 125), font, 0.5, mar_color, 1)
    cv2.putText(frame, f"Yawns: {states['yawn_count']}", (180, 125), font, 0.5, c_white, 1)
    
    # Head Pose
    pose_color = c_red if states["distracted"] else c_green
    cv2.putText(frame, f"Head Pitch: {pitch:.1f} deg", (20, 155), font, 0.5, pose_color, 1)
    cv2.putText(frame, f"Head Yaw:   {yaw:.1f} deg", (20, 185), font, 0.5, pose_color, 1)
    cv2.putText(frame, f"Head Roll:  {roll:.1f} deg", (20, 215), font, 0.5, pose_color, 1)
    
    # 2. Draw warning banners (top-center/right)
    alert_y = 40
    if states["drowsy_alert"]:
        # Drowsy banner
        overlay_alert = frame.copy()
        cv2.rectangle(overlay_alert, (w - 300, alert_y), (w - 10, alert_y + 45), (0, 0, 255), -1)
        cv2.addWeighted(overlay_alert, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "!!! DROWSINESS WARNING !!!", (w - 280, alert_y + 28), font, 0.6, c_white, 2)
        alert_y += 55
        
    if states["distracted"]:
        # Distraction banner
        overlay_alert = frame.copy()
        cv2.rectangle(overlay_alert, (w - 300, alert_y), (w - 10, alert_y + 45), (0, 165, 255), -1)
        cv2.addWeighted(overlay_alert, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "!!! DISTRACTED DRIVER !!!", (w - 280, alert_y + 28), font, 0.6, c_white, 2)
        alert_y += 55
        
    if states["yawning"] and not states["drowsy_alert"]:
        # Yawning status banner
        overlay_alert = frame.copy()
        cv2.rectangle(overlay_alert, (w - 300, alert_y), (w - 10, alert_y + 45), (255, 100, 0), -1)
        cv2.addWeighted(overlay_alert, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "DRIVER YAWNING DETECTED", (w - 285, alert_y + 28), font, 0.5, c_white, 2)

def draw_facial_landmarks(frame, landmarks, states):
    """
    Draws facial contours (eyes and mouth) and key points on the frame.
    """
    c_white = (220, 220, 220)
    c_green = (50, 255, 50)
    c_red = (50, 50, 255)
    c_yellow = (0, 255, 255)
    
    # 1. Draw all landmarks as tiny white points (subtle)
    for lm in landmarks:
        cv2.circle(frame, lm, 1, c_white, -1)
        
    # 2. Highlight eyes
    eye_color = c_red if states["eyes_closed"] else c_green
    
    # Left eye contour
    left_eye_pts = np.array([landmarks[i] for i in LEFT_EYE_INDICES], dtype=np.int32)
    cv2.polylines(frame, [left_eye_pts], isClosed=True, color=eye_color, thickness=2)
    
    # Right eye contour
    right_eye_pts = np.array([landmarks[i] for i in RIGHT_EYE_INDICES], dtype=np.int32)
    cv2.polylines(frame, [right_eye_pts], isClosed=True, color=eye_color, thickness=2)
    
    # 3. Highlight mouth
    mouth_color = c_red if states["yawning"] else c_yellow
    mouth_pts = np.array([landmarks[i] for i in MOUTH_INDICES], dtype=np.int32)
    cv2.polylines(frame, [mouth_pts], isClosed=True, color=mouth_color, thickness=2)

def draw_head_pose_axes(frame, nose_tip, projected_points):
    """
    Draws a 3D coordinate system axis centered at the nose tip to show head pose direction.
    """
    if nose_tip is None or projected_points is None:
        return
        
    p_nose = tuple(nose_tip)
    p_x = tuple(projected_points[0])
    p_y = tuple(projected_points[1])
    p_z = tuple(projected_points[2])
    
    # Draw axes lines: Red (X), Green (Y), Blue (Z)
    cv2.line(frame, p_nose, p_x, (0, 0, 255), 3)  # Pitch axis
    cv2.line(frame, p_nose, p_y, (0, 255, 0), 3)  # Yaw axis
    cv2.line(frame, p_nose, p_z, (255, 0, 0), 3)  # Roll axis (pointing out of face)
    
    # Draw center dot
    cv2.circle(frame, p_nose, 4, (0, 255, 255), -1)
