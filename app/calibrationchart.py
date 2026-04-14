import cv2
import numpy as np

# === Configuration ===
square_size_cm = 2.5       # Size of each square in centimeters
rows = 7                   # Number of inner corners vertically
cols = 9                   # Number of inner corners horizontally
dpi = 300                  # Print resolution (dots per inch)

# === Calculations ===
cm_to_inch = 1 / 2.54
square_size_px = int(square_size_cm * cm_to_inch * dpi)

# The checkerboard has (rows+1) x (cols+1) squares
img_height = (rows + 1) * square_size_px
img_width = (cols + 1) * square_size_px

# Create white background
checkerboard = np.ones((img_height, img_width), dtype=np.uint8) * 255

# Draw black squares
for r in range(rows + 1):
    for c in range(cols + 1):
        if (r + c) % 2 == 0:
            y_start = r * square_size_px
            y_end = y_start + square_size_px
            x_start = c * square_size_px
            x_end = x_start + square_size_px
            checkerboard[y_start:y_end, x_start:x_end] = 0

# Save as high-resolution PNG
cv2.imwrite("calibration_checkerboard.png", checkerboard)

print(f"Checkerboard saved as 'calibration_checkerboard.png'")
print(f"Square size: {square_size_cm} cm, Resolution: {dpi} DPI")
