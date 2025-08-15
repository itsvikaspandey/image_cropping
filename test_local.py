import cv2
from core.orientation import detect_skew_angle,detect_skew_angle_houghP, rotate_image
from core.cropping import crop_document

# Load your local image
image_path = "Dataset-Forensics\image (2).jpg"  # change to your image path
image = cv2.imread(image_path)
image = cv2.resize(image, (800, 600))  

if image is None:
    raise FileNotFoundError(f"Could not load image from {image_path}")

# Step 1: Detect skew angle
image,angle = detect_skew_angle_houghP(image)
print(f"Detected rotation angle: {angle:.2f}°")
cv2.imshow("houghp Image", image)
cv2.waitKey(0)
# Step 2: Rotate image to correct skew
rotated = rotate_image(image, angle)
cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
# Step 3: Crop document boundaries
cropped = crop_document(rotated)

# Step 4: Show images (Press any key to close windows)
cv2.imshow("Original", image)
cv2.imshow("Rotated", rotated)
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save output
cv2.imwrite("output_cropped.jpg", cropped)
print("Cropped image saved as output_cropped.jpg")
