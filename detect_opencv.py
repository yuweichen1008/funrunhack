import cv2
import numpy as np
import mss
import pyautogui
import os
from datetime import datetime

# Load the pattern image (e.g., play button)
template = cv2.imread('play.png', cv2.IMREAD_GRAYSCALE)

# Get the width and height of the template
w, h = template.shape[::-1]

# Directory to save the final Canny result
save_dir = "canny_results"

# Function to save the Canny image to the specified folder
def save_canny_image(edges_img):
    # Create a unique filename based on the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"canny_result_{timestamp}.png"

    # Save the Canny edge-detected image to the folder
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, edges_img)
    print(f"Canny edge-detected image saved to: {filepath}")

# Apply Canny edge detection
def apply_canny_edge_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray_img, 100, 200)  # Apply Canny edge detection
    return edges

# Callback function for trackbars (does nothing, just needed for trackbar creation)
def nothing(x):
    pass

# Create a window with trackbars to adjust the Canny thresholds
def create_canny_adjuster(img):
    # Convert to grayscale for Canny edge detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow('Canny Adjuster')

    # Create trackbars for adjusting the Canny edge detection thresholds
    cv2.createTrackbar('Min Threshold', 'Canny Adjuster', 50, 255, nothing)
    cv2.createTrackbar('Max Threshold', 'Canny Adjuster', 150, 255, nothing)

    while True:
        # Get current positions of the two trackbars
        min_thresh = cv2.getTrackbarPos('Min Threshold', 'Canny Adjuster')
        max_thresh = cv2.getTrackbarPos('Max Threshold', 'Canny Adjuster')

        # Apply Canny edge detection with the current trackbar values
        edges = cv2.Canny(gray_img, min_thresh, max_thresh)

        # Display the original image and the edges image
        combined = np.hstack((cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('Canny Adjuster', combined)

        # Wait for key press and handle 's' key to save, 'ESC' key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # 's' key to save the final result
            final_edges = edges  # Store the current edge-detected image
            save_canny_image(final_edges)
            print("Final Canny result saved.")

    # Clean up
    cv2.destroyAllWindows()

# Take a screenshot of the current display using mss
with mss.mss() as sct:
    screenshot = sct.grab(sct.monitors[1])  # Grab the first monitor (can adjust index if multiple monitors)

    # Convert the screenshot to a numpy array (OpenCV format)
    img = np.array(screenshot)

    # Convert the screenshot to grayscale for template matching
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform template matching to find the pattern
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # Set a threshold for matching accuracy
    threshold = 0.9  # Adjust this value as necessary

    # Get the locations where the template matches the screenshot
    locations = np.where(result >= threshold)
    
    # Move the mouse cursor to the first matched region (if any)
    for point in zip(*locations[::-1]):  # Reverse the x, y coordinates
        # Calculate the center of the matched region
        center_x = point[0] // 2 + w // 2
        center_y = point[1] // 2 + h // 2
        break

    monitor = {"top": center_y - 264, "left": center_x - 560, "width": 650, "height": 285}
    screenshot = sct.grab(monitor)  # Grab the first monitor (can adjust index if multiple monitors)
    img = np.array(screenshot)
    # cv2.imshow('Detected Image', apply_canny_edge_detection(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    create_canny_adjuster(img)