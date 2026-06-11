import cv2
import numpy as np

# Indices for a more robust face silhouette
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
    54, 103, 67, 109
]

def build_face_depth_map(landmarks, W: int, H: int, rvec, tvec, K) -> np.ndarray:
    """
    Creates a 3D depth buffer of the face to allow hair to pass behind/in front.
    Uses a smooth 'dome' approximation for better performance and realism.
    """
    # 1. Initialize with "infinity" (nothing is occluded by default)
    depth_map = np.full((H, W), 999999.0, dtype=np.float32)

    # 2. Project base depth
    tvec_flat = np.array(tvec, dtype=np.float32).flatten()
    face_z_center = float(tvec_flat[2])
    
    # MediaPipe Z is normalized; scale it to real-world mm relative to face distance
    # 0.25 is a heuristic for human head depth (~20cm)
    depth_scale = face_z_center * 0.25 

    # 3. Get 2D points and calculate Z for each landmark
    pts2d = np.array([[lm.x * W, lm.y * H] for lm in landmarks], dtype=np.float32)
    pts_z = np.array([face_z_center + (lm.z * depth_scale) for lm in landmarks], dtype=np.float32)

    # 4. Create the Face Mask (The "Oval")
    hull_pts = pts2d[FACE_OVAL_IDX].astype(np.int32)
    face_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(face_mask, [hull_pts], 255)

    # 5. Smooth Depth Approximation (The "Dome" Effect)
    # Instead of flat planes, we use a distance transform to curve the edges back
    dist_transform = cv2.distanceTransform(face_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Extract key Z-values
    nose_z = pts_z[1]      # Tip of nose (closest to camera)
    forehead_z = pts_z[10] # Top of forehead
    chin_z = pts_z[152]    # Bottom of chin
    
    avg_edge_z = (forehead_z + chin_z) / 2.0 + 50.0 # Push edges back 50mm for skull curve

    # 6. Vectorized Depth Mapping
    # Find all pixels inside the face
    y_idx, x_idx = np.where(face_mask > 0)
    if len(y_idx) == 0:
        return depth_map

    # Map the distance transform to Z values
    # Pixels in the center (nose) get nose_z, pixels at the edge get avg_edge_z
    pixel_dists = dist_transform[y_idx, x_idx]
    
    # Formula: Z = CenterZ + (1.0 - distance_from_center) * Curvature
    # This creates a rounded face shape
    depth_values = nose_z + (1.0 - pixel_dists) * (avg_edge_z - nose_z)
    
    depth_map[y_idx, x_idx] = depth_values

    # 7. Smoothing (Removes aharsh edges that cause hair flickering)
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

    return depth_map