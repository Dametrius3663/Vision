import os
import cv2
import cv2.aruco as aruco
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from core.config import Config

if __name__ == "__main__":
    config = Config.get_instance()
    
    # --- ArUco Setup ---
    aruco_dict = config.aruco_dict
    aruco_params = config.aruco_params

    cam_matrix = config.cam_matrix
    dist_coeffs = config.dist_coeffs
    pntr_id = config.pntr_id
    ref_id = config.ref_id
    marker_size = config.marker_size
    
    # Parameters
    estimate_pose = True
    show_rejected = False
    camera_id = config.camera_id
    video_file = ""  # Leave blank to use webcam

    # Setup dictionary and detector
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # Define object points for ArUco
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)

    # Load STL file
    stl_path = "C:/Users/DHall29/workspace/Vision/app/assets/STLs/1121.STL"
    mesh = trimesh.load(stl_path)
    scale_factor = 0.1
    mesh.apply_scale(scale_factor)

    # Convert Trimesh to Pyrender mesh
    tri_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()

    # Create a camera node
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = scene.add(camera)

    # Add light
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    scene.add(light)

    # Renderer
    r = pyrender.OffscreenRenderer(1280, 960)

    # --- Video Capture Loop ---
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 960))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        
        if ids is not None:
            rvecs = []  # Initialize empty lists outside the loop
            tvecs = []  # Initialize empty lists outside the loop
            
            for i, marker_id in enumerate(ids):
                ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
                
                rvecs.append(rvec)
                tvecs.append(tvec)
                # Draw detected marker
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Get rotation and translation vectors
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i]

                # Build transformation matrix
                T = np.identity(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                # Clear previous scene
                for n in list(scene.mesh_nodes):
                    scene.remove_node(n)

                # Add mesh to scene with transformation
                scene.add(tri_mesh, pose=T)

                # Render the scene to image
                color, depth = r.render(scene)

                # Overlay rendered 3D object onto original frame
                mask = (depth != 0)
                frame[mask] = color[mask]

        # Show result
        cv2.imshow("AR Display", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()