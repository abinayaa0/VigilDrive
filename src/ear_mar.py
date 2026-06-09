from scipy.spatial import distance

def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) to determine eye openness/blink status.
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2.0 * ||p1-p4||)
    """
    # Vertical distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = distance.euclidean(eye[0], eye[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """
    Calculate the Mouth Aspect Ratio (MAR) to determine yawning status.
    Formula: MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2.0 * ||p1-p5||)
    """
    # Vertical distances
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    # Horizontal distance
    D = distance.euclidean(mouth[0], mouth[4])
    
    # Calculate MAR
    mar = (A + B + C) / (2.0 * D)
    return mar
