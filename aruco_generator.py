import cv2
import cv2.aruco
import numpy as np
import os
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Page size: 11 x 17 inches, landscape orientation
page_width_inches = 17
page_height_inches = 11
page_width_points = page_width_inches * inch
page_height_points = page_height_inches * inch

# Print resolution for marker generation
print_dpi = 300

# Marker size on the page: use nearly full sheet with a small printable margin
page_margin_inches = 0.25
marker_size_inches = min(page_width_inches, page_height_inches) - (2 * page_margin_inches)
marker_size_pixels = int(marker_size_inches * print_dpi)
marker_size_points = marker_size_inches * inch

# Additional border around the marker inside the page
border_width_inches = 0.25
border_width_points = border_width_inches * inch
border_line_thickness_points = 2

# Output locations
output_folder = "aruco_marker_images"
os.makedirs(output_folder, exist_ok=True)
output_filename = "aruco_markers_sheet.pdf"
c = canvas.Canvas(output_filename, pagesize=(page_width_points, page_height_points))

marker_ids = range(1, 50)  # IDs 1 through 49
page_counter = 0

for marker_id in marker_ids:
    marker_image = np.zeros((marker_size_pixels, marker_size_pixels), dtype="uint8")
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels, marker_image, 1)

    temp_png_filename = os.path.join(output_folder, f"marker_{marker_id}.png")
    cv2.imwrite(temp_png_filename, marker_image)

    # Center the marker on the page
    x_pos = (page_width_points - marker_size_points) / 2
    y_pos = (page_height_points - marker_size_points) / 2

    # Draw border first
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(border_line_thickness_points)
    c.rect(
        x_pos - border_width_points,
        y_pos - border_width_points,
        marker_size_points + 2 * border_width_points,
        marker_size_points + 2 * border_width_points,
        stroke=1,
        fill=0,
    )

    # Draw marker image
    c.drawImage(temp_png_filename, x_pos, y_pos, width=marker_size_points, height=marker_size_points)

    # Add marker ID text if desired
    c.setFont("Helvetica", 14)
    c.drawCentredString(page_width_points / 2, y_pos - 18, f"ID: {marker_id}")

    c.showPage()
    page_counter += 1

c.save()
print(f"ArUco markers sheet generated: {output_filename} with {page_counter} page(s).")