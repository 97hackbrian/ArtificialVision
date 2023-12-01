import cv2
import numpy as np

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Read the first frame
ret, first_frame = cap.read()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing purposes
mask = np.zeros_like(first_frame)

# Define initial points to track
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params).astype(np.float32)

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, winSize=(15, 15), maxLevel=2)

    # Select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Calculate displacement with two decimal places
        dx, dy = a - c, b - d
        print(f"Desplazamiento {i + 1}: dx = {dx:.2f}, dy = {dy:.2f}")

    # Combine the frame and the mask
    result = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Optical Flow', result)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame and previous points
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
