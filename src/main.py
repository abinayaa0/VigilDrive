import os
import re
import cv2
import time
import argparse
import numpy as np

from landmarks import FaceMeshDetector, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
from ear_mar import eye_aspect_ratio, mouth_aspect_ratio
from head_pose import estimate_head_pose
from utils import MovingAverageSmoother, DriverStateTracker
from visualization import draw_hud, draw_facial_landmarks, draw_head_pose_axes

def get_sequence_images(directory_path):
    """
    Finds and chronologically sorts image sequences in a directory.
    Uses regex to extract sequential frame indices for correct chronological ordering.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        return []
        
    filenames = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_exts)]
    
    # Custom sort key to sort by the sequential frame number in filename
    def extract_frame_number(fname):
        nums = re.findall(r'\d+', fname)
        # Typically the filename structure is like '001_glasses_yawning_1000_drowsy.jpg'
        # The first number is subject ID (001), the second is the frame number (1000).
        if len(nums) >= 2:
            return int(nums[1])
        elif len(nums) == 1:
            return int(nums[0])
        return 0

    filenames.sort(key=extract_frame_number)
    return [os.path.join(directory_path, f) for f in filenames]

def process_frame(frame, detector, tracker, ear_smoother, mar_smoother, pitch_smoother, yaw_smoother, roll_smoother, fps):
    """
    Applies the full driver monitoring pipeline to a single image frame.
    """
    # Resize frame for uniform processing size (640, 480)
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    
    # Face Mesh processing
    results = detector.find_face_landmarks(frame)
    
    metrics = {"ear": 0.0, "mar": 0.0}
    states = {
        "blink_count": tracker.blink_count,
        "yawn_count": tracker.yawn_count,
        "eyes_closed": False,
        "drowsy_alert": False,
        "yawning": False,
        "distracted": False
    }
    head_pose = (0.0, 0.0, 0.0)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. Convert normalized coordinates to pixels
            pixel_landmarks = detector.get_pixel_landmarks(face_landmarks, frame.shape)
            
            # 2. Extract Eye and Mouth Regions
            left_eye = [pixel_landmarks[i] for i in LEFT_EYE_INDICES]
            right_eye = [pixel_landmarks[i] for i in RIGHT_EYE_INDICES]
            mouth = [pixel_landmarks[i] for i in MOUTH_INDICES]
            
            # 3. Calculate EAR and MAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            raw_ear = (left_ear + right_ear) / 2.0
            raw_mar = mouth_aspect_ratio(mouth)
            
            # 4. Smooth out EAR/MAR values to reduce jitter
            ear = ear_smoother.update(raw_ear)
            mar = mar_smoother.update(raw_mar)
            
            metrics["ear"] = ear
            metrics["mar"] = mar
            
            # 5. Estimate Head Pose
            raw_pitch, raw_yaw, raw_roll, nose_tip, projected_points = estimate_head_pose(pixel_landmarks, frame.shape)
            
            if raw_pitch is not None:
                pitch = pitch_smoother.update(raw_pitch)
                yaw = yaw_smoother.update(raw_yaw)
                roll = roll_smoother.update(raw_roll)
                head_pose = (pitch, yaw, roll)
                
                # Draw Pose axis at nose tip
                draw_head_pose_axes(frame, nose_tip, projected_points)
            
            # 6. Update Driver State machine and check thresholds
            states = tracker.update_states(ear, mar, head_pose[0], head_pose[1], head_pose[2], fps)
            
            # 7. Draw face contours and keypoints
            draw_facial_landmarks(frame, pixel_landmarks, states)
            break # Process only the first face detected
            
    # 8. Render HUD overlaid card and warning banners
    draw_hud(frame, metrics, states, fps, head_pose)
    
    return frame, states

def main():
    parser = argparse.ArgumentParser(description="VigilDrive Phase 1 — Driver Monitoring System Pipeline")
    parser.add_argument("--mode", type=str, default="dataset", choices=["webcam", "dataset", "video"],
                        help="Input source: 'webcam', 'dataset' (image directory sequences), or 'video'")
    parser.add_argument("--cam-idx", type=int, default=0, help="Camera device index for webcam mode")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\PROJECTS\VigilDrive\dataset_NTHU\Multi class\train\drowsy\yawning",
                        help="Directory path to NTHU dataset image sequences")
    parser.add_argument("--video-path", type=str, default="", help="Path to video file")
    parser.add_argument("--output-path", type=str, default="outputs/output.avi", help="Path to save processed video")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames to process (especially in dataset mode)")
    parser.add_argument("--no-show", action="store_true", help="Disable cv2.imshow for headless verification environment")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else "outputs", exist_ok=True)

    # Initialize modules
    detector = FaceMeshDetector(static_image_mode=(args.mode == "dataset"))
    tracker = DriverStateTracker()
    
    # Smoothers
    ear_smoother = MovingAverageSmoother(window_size=5)
    mar_smoother = MovingAverageSmoother(window_size=5)
    pitch_smoother = MovingAverageSmoother(window_size=5)
    yaw_smoother = MovingAverageSmoother(window_size=5)
    roll_smoother = MovingAverageSmoother(window_size=5)

    # Setup Video Source
    image_paths = []
    cap = None
    
    if args.mode == "dataset":
        print(f"Loading sequence from NTHU directory: {args.dataset_dir}")
        image_paths = get_sequence_images(args.dataset_dir)
        if not image_paths:
            print("Error: No images found in sequence directory.")
            return
        print(f"Loaded {len(image_paths)} images from NTHU dataset.")
        total_frames = min(len(image_paths), args.max_frames)
    elif args.mode == "video":
        if not os.path.exists(args.video_path):
            print(f"Error: Video file {args.video_path} does not exist.")
            return
        cap = cv2.VideoCapture(args.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)
    else: # webcam
        print(f"Initializing webcam capture device {args.cam_idx}...")
        cap = cv2.VideoCapture(args.cam_idx)
        if not cap.isOpened():
            print(f"Error: Webcam index {args.cam_idx} failed to open.")
            return
        total_frames = args.max_frames

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, 20.0, (640, 480))
    print(f"Video Writer initialized to save output to: {args.output_path}")

    frame_count = 0
    prev_time = time.time()
    
    print("\nStarting VigilDrive processing pipeline...")
    try:
        while frame_count < total_frames:
            # 1. Fetch frame
            if args.mode == "dataset":
                img_path = image_paths[frame_count]
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Error: Failed to read image {img_path}")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or failed to grab frame.")
                    break
            
            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1.0 / time_diff if time_diff > 0 else 20.0
            prev_time = curr_time
            
            # 2. Process frame through VigilDrive Pipeline
            processed_frame, states = process_frame(
                frame, detector, tracker, 
                ear_smoother, mar_smoother, 
                pitch_smoother, yaw_smoother, roll_smoother, 
                fps
            )
            
            # 3. Write output frame
            out.write(processed_frame)
            
            # Log periodic stats
            if frame_count % 30 == 0 or frame_count == total_frames - 1:
                print(f"Processed frame [{frame_count+1}/{total_frames}] | FPS: {fps:.1f} | "
                      f"Blinks: {states['blink_count']} | Yawns: {states['yawn_count']} | "
                      f"Drowsy alert: {states['drowsy_alert']} | Distracted: {states['distracted']}")

            # 4. Interactive Display (unless no-show is set)
            if not args.no_show:
                try:
                    cv2.imshow("VigilDrive DMS HUD", processed_frame)
                    # Press 'q' to abort early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User initiated abort.")
                        break
                except cv2.error:
                    # In a headless system, imshow will fail. Set args.no_show=True automatically.
                    print("GUI Display error detected (running in headless environment). Disabling display preview.")
                    args.no_show = True

            frame_count += 1
            
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        out.release()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\nPipeline finished! Processed {frame_count} frames.")
        print(f"Output saved to absolute path: {os.path.abspath(args.output_path)}")

if __name__ == "__main__":
    main()
