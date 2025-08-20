import cv2
import cv2.aruco
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm # Import mm for easier calculations

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# Marker parameters
desired_cm = 5
desired_inches = desired_cm / 2.54
print_dpi = 300 # Common printing DPI (adjust if your printer uses a different DPI)
marker_size_pixels = int(desired_inches * print_dpi)

# Define marker and page properties
marker_size = marker_size_pixels # Size of the marker in pixels (calculated for 5cm at print_dpi)
border_bits = 1 # The marker's internal black border

# Define the *additional* border properties
border_width_cm = 0.5
border_width_inches = border_width_cm / 2.54
border_width_pixels = int(border_width_inches * print_dpi) # Width of the additional border in pixels

# For drawing the border in ReportLab, we'll use a 2-pixel line thickness
border_line_thickness_points = 2 * (72 / print_dpi) # Convert pixels to points for ReportLab drawing

margin_mm = 5 # Margin between markers in millimeters
margin_pixels = int((margin_mm / 25.4) * print_dpi)

# Get page dimensions
c = canvas.Canvas("aruco_markers_sheet.pdf", pagesize=letter)
width_points, height_points = letter # Page dimensions in points (72 points = 1 inch)

# Convert dimensions to points for ReportLab for placement and drawing
marker_size_points = marker_size_pixels * (72 / print_dpi)
border_width_points = border_width_pixels * (72 / print_dpi)
margin_points = margin_pixels * (72 / print_dpi)

# The total space each marker *including its border* will occupy
marker_with_border_size_points = marker_size_points + (2 * border_width_points)

# Calculate how many markers fit horizontally and vertically
page_margin_points = 0.5 * inch # Page margin from the edge of the paper
printable_width = width_points - (2 * page_margin_points)
printable_height = height_points - (2 * page_margin_points)

markers_per_row = int((printable_width + margin_points) / (marker_with_border_size_points + margin_points))
markers_per_col = int((printable_height + margin_points) / (marker_with_border_size_points + margin_points))

if markers_per_row > 0 and markers_per_col > 0:
    print(f"Generating {markers_per_row}x{markers_per_col} grid of ArUco markers per page.")
    print(f"Calculated marker_size (pixels) for {desired_cm}cm at {print_dpi} DPI: {marker_size_pixels}")
    print(f"Marker size (including border) in points for ReportLab: {marker_with_border_size_points:.2f}")

    marker_id_counter = 1
    page_counter = 1

    while True:
        # Calculate starting X and Y for centering the grid on the page
        total_grid_width = (markers_per_row * marker_with_border_size_points) + ((markers_per_row - 1) * margin_points)
        total_grid_height = (markers_per_col * marker_with_border_size_points) + ((markers_per_col - 1) * margin_points)

        start_x = (width_points - total_grid_width) / 2
        start_y = (height_points - total_grid_height) / 2

        for row in range(markers_per_col):
            for col in range(markers_per_row):
                if marker_id_counter > 49:
                    break

                marker_image = np.zeros((marker_size, marker_size, 1), dtype="uint8")
                cv2.aruco.generateImageMarker(aruco_dict, marker_id_counter, marker_size, marker_image, border_bits)

                temp_png_filename = f"marker_{marker_id_counter}.png"
                cv2.imwrite(temp_png_filename, marker_image)

                # Calculate position for the entire marker-with-border block
                block_x_pos = start_x + col * (marker_with_border_size_points + margin_points)
                # Y-axis is inverted in ReportLab
                block_y_pos = start_y + (markers_per_col - 1 - row) * (marker_with_border_size_points + margin_points) 

                # Draw the additional border (rectangle) first
                c.setStrokeColorRGB(0, 0, 0) # Black color for the border
                c.setLineWidth(border_line_thickness_points)
                c.rect(block_x_pos, block_y_pos, marker_with_border_size_points, marker_with_border_size_points)

                # Draw the ArUco marker image *inside* the border
                # The image starts 'border_width_points' in from the block_x_pos and block_y_pos
                image_x_pos = block_x_pos + border_width_points
                image_y_pos = block_y_pos + border_width_points

                c.drawImage(temp_png_filename, image_x_pos, image_y_pos, width=marker_size_points, height=marker_size_points)

                # Optionally add text below the marker
                text_x = block_x_pos + marker_with_border_size_points / 2
                text_y = block_y_pos - 15 # Adjust for text placement
                c.setFont("Helvetica", 8)
                c.drawCentredString(text_x, text_y, f"ID: {marker_id_counter}")

                marker_id_counter += 1
            if marker_id_counter > 49:
                break

        if marker_id_counter <= 49:
            c.showPage()
            page_counter += 1
        else:
            break

else:
    print("Markers (including borders) are too large to fit on the page with the given margins.")

c.save()

print(f"ArUco markers sheet generated: aruco_markers_sheet.pdf with {page_counter} page(s).")