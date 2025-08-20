import cv2
import cv2.aruco as aruco
import numpy as np
from core.config import Config
import yaml

def get_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def invert_4x4_transform(T):
    """Invert a 4x4 rigid transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T  # R^T
    T_inv[:3, 3] = -R.T @ t  # -R^T * t
    return T_inv

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

    # Variables to store the pose of the reference and pointer markers
    rvec_ref, tvec_ref = None, None
    rvec_pntr, tvec_pntr = None, None

    # Setup dictionary and detector
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    
    # Setup video input
    cap = cv2.VideoCapture(video_file if video_file else camera_id)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    wait_time = 1 # milliseconds; 1 allows real-time behavior

    # Define object points for a square planar ArUco marker (z=0)
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    
    # Define font, scale, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (0, 255, 255) # Yellow
    font_thickness = 2
    list_ids = [ref_id,pntr_id]
    
    list_rvec = [None] * 2
    list_tvec = [None] * 2
    origin = np.array([0, 0, 0, 1])  # Shape: (4,)
    previous_position = None
    tracked_positions = []
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2frame)
        frame = frame
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)
        
        if ids is None or ids.size < 2:
            print("Could not find pointer and/or reference")
            continue
        
        is_valid :bool = True
        for id in ids:
            if id not in list_ids:      		
                print("Problem")
                is_valid = False
            break
        if not is_valid:
            continue
        
        # we know we have the pointer and the reference frame
        for i in range(len(ids)):
            ret, rvec, tvec = cv2.solvePnP(obj_points, corners[i], cam_matrix, dist_coeffs)
            
            if ids[i] == ref_id:
                list_rvec[0] = rvec
                list_tvec[0] = tvec
            else:
                list_rvec[1] = rvec
                list_tvec[1] = tvec
            # Display the marker ID
            center_x = int(np.mean(corners[i][0][:, 0]))
            center_y = int(np.mean(corners[i][0][:, 1])) - 10

            # Convert the marker_id to a string
            text = f"   ID: {ids[i]}"

            # Put the text on the frame
            cv2.putText(frame, text, (center_x, center_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            #End of marker ID display
            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_size)
            
        rot_ref, _ = cv2.Rodrigues(list_rvec[0])
        rot_pntr, _ = cv2.Rodrigues(list_rvec[1])
        tr_ref = list_tvec[0]
        tr_pntr = list_tvec[1]
        
        # Camera pose in pointer's coordinate system
        # R_cam_to_pntr = rot_ref.T  # Inverse rotation
        # t_cam_to_pntr = -R_cam_to_pntr @ tvec  # Inverse translation
        # print("Camera position in pointer's frame:", t_cam_to_pntr)
        # print("Camera orientation in pointer's frame:", R_cam_to_pntr)
        
        # Build 4x4 matrix
        transf_cam_ref = get_transformation_matrix(rot_ref, tr_ref)
        transf_cam_pntr = get_transformation_matrix(rot_pntr, tr_pntr)
        
        # Build 4x4 matrix
        # transf_cam_pntr = np.eye(4)
        # transf_cam_pntr[:3, :3] = rot_pntr
        # transf_cam_pntr[:3, 3] = tr_pntr.flatten()

        # Using @ for matrix multiplication instead of * to properly transform 4x4 matrices
        # This operation calculates the transformation from the camera reference frame to the pointer frame
        # by multiplying the reference transformation with the inverted pointer transformation
        
        trans_pntr_ref = transf_cam_ref @ invert_4x4_transform(transf_cam_pntr)
        origin_ = trans_pntr_ref @ origin
        
        print(origin_)
                    
        # Display the frame
        cv2.imshow("ArUco Pose", frame)

        # Exit on keypress 'q'
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
    