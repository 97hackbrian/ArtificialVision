import cv2
import numpy as np

# Function to perform Lucas-Kanade optical flow
def calculate_optical_flow(prev_frame, current_frame, prev_points):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, current_frame, prev_points, None, winSize=(15, 15), maxLevel=2
    )
    return next_points, status

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Read the first frame
ret, first_frame = cap.read()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Select points to track (e.g., corners)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Conoce la distancia en la escena (en metros)
distancia_real_metros = 2.0  # Ejemplo, 2 metros

# Mide cuántos píxeles abarca esta distancia en las imágenes
distancia_pixeles = 100  # Ejemplo, 100 píxeles

# Calcula la relación de píxeles a metros
relacion_pixel_metro = distancia_real_metros / distancia_pixeles

while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    next_points, status = calculate_optical_flow(prev_gray, current_gray, prev_points)

    # Select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # Print the displacement in meters
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        dx, dy = new.ravel() - old.ravel()
        dx_metros = dx * relacion_pixel_metro
        dy_metros = dy * relacion_pixel_metro
        print(f"Desplazamiento {i + 1}: dx = {dx_metros:.2f} metros, dy = {dy_metros:.2f} metros")

    # Draw the tracks on the mask
    mask = np.zeros_like(frame)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    # Overlay the tracks on the original frame
    img = cv2.add(frame, mask)

    # Display the result
    cv2.imshow("Object Tracking", img)

    # Update previous points and frames
    prev_gray = current_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
