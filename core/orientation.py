import cv2
import numpy as np

def detect_skew_angle(image: np.ndarray) -> float:
    """Detect document skew angle using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Canny Image", edges)
    cv2.waitKey(0)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = [(theta - np.pi/2) * 180 / np.pi for rho, theta in lines[:, 0]]
        return float(np.median(angles))
    return 0.0
import cv2
import numpy as np

def dilation_image(img, edges, kernel_size=(20, 50), show=True):
    """
    Dilates Canny edges, finds the contour closest to the image center,
    and draws a bounding rectangle around it.

    Args:
        img (np.ndarray): Original image.
        edges (np.ndarray): Canny edge output.
        kernel_size (tuple): Size of dilation kernel (height, width).
        show (bool): If True, display intermediate images.

    Returns:
        tuple: (output_img, (x, y, w, h)) where (x, y, w, h) is the bounding box.
    """
    # Step 1: Dilation
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    if show:
        cv2.imshow("Dilated Image", dilated)

    # Step 2: Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Image center
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Step 4: Find contour closest to image center
    closest_contour = None
    min_distance = float("inf")

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_contour = cnt

    # Step 5: Draw rectangle for closest contour
    output = img.copy()
    bbox = None
    if closest_contour is not None:
        x, y, bw, bh = cv2.boundingRect(closest_contour)
        bbox = (x, y, bw, bh)
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)  # Mark center
        if show:
            cv2.imshow("Closest Contour", output)

    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output, bbox
def detect_skew_angle_houghP(image):
    """Detect skew angle using probabilistic Hough transform and return median."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    cv2.imshow("Canny Image", edges)
    cv2.waitKey(0)
    output,bbox = dilation_image(image, edges)
    x, y, bw, bh = bbox
    cropped_img = output[y:y + bh, x:x + bw]
    cv2.imshow("Cropp Image", cropped_img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # cv2.line(cropped_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
    median_angle = np.median(angles) if angles else 0.0
    return cropped_img, median_angle
def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle (degrees)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
