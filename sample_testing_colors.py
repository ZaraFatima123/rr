import cv2
import numpy as np
# from util import get_limits

# Load an underwater image directly from the current directory
image_path = "image_color.jpg"  # Use the image filename only if it's in the same directory
image_path2 = "pp.jpg"
image_path3 = "underwater.jpg"
image_path4 = "pink_fish.jpeg"
image_path5 = "bartletts-anthias.jpg"
image_path5 = "Ocean_Underwater.jpg"
frame = cv2.imread(image_path4)

# Check if the image was loaded successfully
if frame is None:
    print("Failed to load image!")
else:
    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect Pink/Magenta color
    def pink_detect():
        low_pink = np.array([145, 50, 70])
        high_pink = np.array([170, 255, 255])
        pink_mask = cv2.inRange(hsv_frame, low_pink, high_pink)
        return pink_mask

    # Detect Orange color
    def orange_detect():
        low_orange = np.array([10, 100, 100])  # Adjusted lower limit
        high_orange = np.array([20, 255, 255])  # Adjusted upper limit
        orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)

        # Apply morphological operation to reduce noise
        # kernel = np.ones((5, 5), np.uint8)
        # orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        # orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        
        return orange_mask

    # Combine all color masks
    combined_mask = pink_detect() | orange_detect()
    
    # Apply the combined mask on the original frame
    combined_result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Display the original frame and the combined color detection frame
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Show Color Detection", combined_result)

    # Wait until a key is pressed
    cv2.waitKey(0)

# Release all OpenCV windows
cv2.destroyAllWindows()
