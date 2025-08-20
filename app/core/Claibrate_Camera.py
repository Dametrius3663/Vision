import cv2
import numpy as np
import yaml
import os 
import shutil

# Define chessboard parameters
chessboard_size = (9, 6) # Number of inner corners per a chessboard row and column
square_size = 2.5 # Size of each square in centimeters (adjust to your board)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Capture video from webcam
cap = cv2.VideoCapture(1)

# Loop to capture images for calibration
print("Press 'c' to capture a calibration image. Press 'q' to quit.")

# Define the folder to save images 
good_images = "app/assets/good_images"
    
# --- Clear existing folders at the start of each run ---
if os.path.exists(good_images):
    print(f"Clearing existing folder: {good_images}")
    shutil.rmtree(good_images) # Delete the output folder and its contents recursively
    print("Folder cleared.")
    
if not os.path.exists(good_images):
    os.makedirs(good_images)
    print(f"Created folder: {good_images}")

image_count = 0  # To create unique filenames for each captured image

# Blur detection threshold
BLUR_THRESHOLD = 150  

# Function to detect blur using Laplacian variance
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Calculate Laplacian variance
    return laplacian_variance < threshold, laplacian_variance

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    is_blurry_image, variance = is_blurry(frame)
    
    # If found, add object points, image points
    if ret_corners and not is_blurry_image:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)

    # Display the frame
    cv2.imshow('Calibration', frame)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('c') and ret_corners: # Capture if 'c' is pressed and corners are found
        is_blurry_image, variance = is_blurry(frame)
        if not is_blurry_image:  # Only save if not blurry
            file_name = f"captured_image{image_count:04d}.jpg"
            save_path = os.path.join(good_images, file_name)

            # Save the image to the specified path
            cv2.imwrite(save_path, frame)  # Pass the 'frame' directly
            print(f"Image captured and saved to {save_path} (Variance: {variance:.2f})")
            image_count += 1
        else:
            print(f"Image captured but discarded due to blur (Variance: {variance:.2f})")

    elif key & 0xFF == ord('q'): # Quit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Calibrate the camera if images are captured
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration parameters to a YAML file
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.tolist()
    }

    with open('calibration_params.yml', 'w') as f:
        yaml.dump(calibration_data, f)

    print("Calibration parameters saved to calibration_params.yml")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                         mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    
    print("\nCalibration results:")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients: {dist.ravel()}")
    print(f"Mean reprojection error: {mean_error:.3f} pixels")
    
    

else:
    print("No images captured for calibration. Please capture images with the chessboard pattern.")