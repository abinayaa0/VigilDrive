import time
from collections import deque
import numpy as np

class MovingAverageSmoother:
    """
    Queue-based moving average filter to smooth coordinates or values,
    reducing noise and detection jitter.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, val):
        self.values.append(val)
        return np.mean(self.values, axis=0)

    def get_latest(self):
        if not self.values:
            return 0.0
        return np.mean(self.values, axis=0)


class DriverStateTracker:
    """
    Tracks state transitions over time to calculate blink frequency,
    eye closure durations (drowsiness), yawning, and gaze distraction.
    """
    def __init__(self, ear_threshold=0.25, mar_threshold=0.55,
                 drowsy_time_threshold=1.5, yawn_time_threshold=1.5,
                 distracted_time_threshold=1.5,
                 pitch_threshold=20.0, yaw_threshold=25.0, roll_threshold=25.0):
        
        # Configuration Thresholds
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_time_threshold = drowsy_time_threshold  # seconds
        self.yawn_time_threshold = yawn_time_threshold      # seconds
        self.distracted_time_threshold = distracted_time_threshold  # seconds
        
        # Head Pose Thresholds (in degrees)
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.roll_threshold = roll_threshold

        # Real-time state counts
        self.blink_count = 0
        self.yawn_count = 0
        
        # Temporal states
        self.eyes_closed_start_time = None
        self.yawn_start_time = None
        self.distracted_start_time = None
        
        # Flags
        self.eyes_closed = False
        self.yawning = False
        self.distracted = False
        self.drowsy_alert = False

    def update_states(self, ear, mar, pitch, yaw, roll, fps):
        """
        Updates trackers based on current frame values. Returns current status dict.
        """
        curr_time = time.time()
        
        # --- 1. EYE CLOSURE / BLINK / DROWSINESS DETECTION ---
        if ear < self.ear_threshold:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.eyes_closed_start_time = curr_time
            else:
                # Check how long the eyes have been closed
                closed_duration = curr_time - self.eyes_closed_start_time
                if closed_duration >= self.drowsy_time_threshold:
                    self.drowsy_alert = True
        else:
            if self.eyes_closed:
                # If they were closed, check duration
                closed_duration = curr_time - self.eyes_closed_start_time
                # A normal blink is typically 0.1 to 0.4 seconds
                if 0.05 <= closed_duration < self.drowsy_time_threshold:
                    self.blink_count += 1
                
                # Reset eyes closed state
                self.eyes_closed = False
                self.eyes_closed_start_time = None
                self.drowsy_alert = False

        # --- 2. YAWNING DETECTION ---
        if mar > self.mar_threshold:
            if not self.yawning:
                self.yawning = True
                self.yawn_start_time = curr_time
            else:
                yawn_duration = curr_time - self.yawn_start_time
                # If they keep mouth wide open for a threshold, register a yawn
                if yawn_duration >= self.yawn_time_threshold and self.yawn_start_time is not None:
                    self.yawn_count += 1
                    # Prevent multiple counts for the same yawn by clearing start time until reset
                    self.yawn_start_time = None 
        else:
            if self.yawning:
                self.yawning = False
                self.yawn_start_time = None

        # --- 3. DISTRACTION / GAZE DETECTION ---
        # Detect if head angles exceed thresholds
        is_looking_away = (
            abs(pitch) > self.pitch_threshold or 
            abs(yaw) > self.yaw_threshold or 
            abs(roll) > self.roll_threshold
        )
        
        if is_looking_away:
            if self.distracted_start_time is None:
                self.distracted_start_time = curr_time
            else:
                distracted_duration = curr_time - self.distracted_start_time
                if distracted_duration >= self.distracted_time_threshold:
                    self.distracted = True
        else:
            self.distracted = False
            self.distracted_start_time = None

        # Prepare summary dictionary
        return {
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "eyes_closed": self.eyes_closed,
            "drowsy_alert": self.drowsy_alert,
            "yawning": self.yawning,
            "distracted": self.distracted,
            "eyes_closed_sec": (curr_time - self.eyes_closed_start_time) if self.eyes_closed_start_time else 0.0,
            "yawn_sec": (curr_time - self.yawn_start_time) if (self.yawning and self.yawn_start_time) else 0.0,
            "distracted_sec": (curr_time - self.distracted_start_time) if self.distracted_start_time else 0.0
        }
