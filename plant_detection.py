'''

import cv2
import pytesseract

def detect_and_crop_plant(image_path, output_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to get text from the image
    text = pytesseract.image_to_string(gray_img)

    # Check if the detected text indicates the presence of a plant
    if "plant" in text.lower():
        print("Plant detected!")

        # Convert the grayscale image to binary using thresholding
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours and find the bounding box around the plant
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # You can add additional conditions to refine the bounding box if needed

            # Crop the image using the bounding box
            cropped_img = img[y:y + h, x:x + w]

            # Save the cropped image
            cv2.imwrite(output_path, cropped_img)
            print("Cropped image saved at:", output_path)
            break  # Break after processing the first bounding box (you can modify this based on your needs)
    else:
        print("No plant detected.")

# Example usage
detect_and_crop_plant('/Users/joycekexinqian/Downloads/OO_data/train/LEAF_0035.jpg', '/Users/joycekexinqian/Downloads/OO_data/cropped')
'''
'''
import cv2

def detect_and_crop_plant(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 30, 150)  # Adjust these thresholds if needed

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to retain only significant ones
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Sort contours based on area in descending order
    significant_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)

    # If there are significant contours, crop the region of interest (ROI)
    if significant_contours:
        x, y, w, h = cv2.boundingRect(significant_contours[0])

        # Draw a rectangle around the detected plant area
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 450, 0), 2)

        # Save the cropped ROI as a new image
        cropped_image_path = "/Users/joycekexinqian/Downloads/cropped_plant.jpg"
        cv2.imwrite(cropped_image_path, img)

        # Display the image with the bounding box
        cv2.imshow('Detected Plant', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cropped_image_path
    else:
        print("No significant contours found.")
        return None

# Example usage:
image_path = '/Users/joycekexinqian/Downloads/alx.jpg'
cropped_image = detect_and_crop_plant(image_path)

if cropped_image:
    print(f"Cropped image saved at: {cropped_image}")
else:
    print("Plant not detected.")
'''

'''
def detect_and_crop_plant(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 30, 150)  # Adjust these thresholds if needed

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to retain only significant ones
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Sort contours based on area in descending order
    significant_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)

    # If there are significant contours, crop the region of interest (ROI)
    if significant_contours:
        x, y, w, h = cv2.boundingRect(significant_contours[0])

        # Crop out the most significant contour (plant area)
        roi = img[y:y + h, x:x + w]

        # Save the cropped ROI as a new image
        cropped_image_path = "/Users/joycekexinqian/Downloads/cropped_plant.jpg"
        cv2.imwrite(cropped_image_path, roi)

        return cropped_image_path
    else:
        print("No significant contours found.")
        return None
'''
import cv2
import numpy as np

def detect_and_crop_plant(image_path, margin=50):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to retain only significant ones
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    # Sort contours based on area in descending order
    significant_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)

    # If there are significant contours, crop the region of interest (ROI)
    if significant_contours:
        x, y, w, h = cv2.boundingRect(significant_contours[0])
        
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        # Use NumPy slicing to crop the region of interest (ROI) while keeping RGB information
        roi = img[y:y+h, x:x+w].copy()

        # Save the cropped ROI as a new image
        cropped_image_path = "/Users/joycekexinqian/Downloads/cropped_plant.jpg"
        cv2.imwrite(cropped_image_path, roi)

        return cropped_image_path
    else:
        print("No significant contours found.")
        return None
    
# Example usage:
image_path = '/Users/joycekexinqian/Downloads/alx.jpg'
cropped_image = detect_and_crop_plant(image_path,margin=150)

if cropped_image:
    print(f"Cropped image saved at: {cropped_image}")
else:
    print("Plant not detected.")
