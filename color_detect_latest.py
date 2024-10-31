import cv2
import numpy as np

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper_limit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lower_limit = np.array([0, 100, 100], dtype=np.uint8)
        upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit

# Open the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Set to detect colors and their names
deep_blue_overlay = np.full((480, 640, 3), (255, 150, 0), dtype=np.uint8)  # More intense blue in BGR

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture frame!")
        break  # If frame is not read correctly, exit the loop

    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color limits using get_limits function
    pink_lower, pink_upper = get_limits([203, 192, 255])  # BGR for pink
    orange_lower, orange_upper = get_limits([0, 165, 255])  # BGR for orange
    yellow_lower, yellow_upper = get_limits([0, 255, 255])  # BGR for yellow

    # Create masks for the colors
    pink_mask = cv2.inRange(hsv_frame, pink_lower, pink_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

    # Combine all color masks
    combined_mask = pink_mask | orange_mask | yellow_mask

    # Find contours from the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to draw rectangles and labels
    detected_colors = []

    # Draw contours and bounding rectangles on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # Determine which color was detected
            if cv2.countNonZero(pink_mask[y:y + h, x:x + w]) > 0:
                detected_colors.append("Pink")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (203, 192, 255), 2)  # Draw pink rectangle
                cv2.putText(frame, "Pink", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (203, 192, 255), 2)
            if cv2.countNonZero(orange_mask[y:y + h, x:x + w]) > 0:
                detected_colors.append("Orange")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Draw orange rectangle
                cv2.putText(frame, "Orange", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Blend the blue overlay with the original frame
    overlay_alpha = 0.5  # Adjust alpha for blending
    frame_with_blue_filter = cv2.addWeighted(deep_blue_overlay, overlay_alpha, frame, 1 - overlay_alpha, 0)

    # Display the original frame with detected rectangles and color names
    cv2.imshow("Original Frame with Deep Blue Filter", frame_with_blue_filter)
    # Display the frame with rectangles and names
    cv2.imshow("Show Color Detection with Deep Blue Filter", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
print("Detecting colors with deep blue filter applied")
cv2.destroyAllWindows()
