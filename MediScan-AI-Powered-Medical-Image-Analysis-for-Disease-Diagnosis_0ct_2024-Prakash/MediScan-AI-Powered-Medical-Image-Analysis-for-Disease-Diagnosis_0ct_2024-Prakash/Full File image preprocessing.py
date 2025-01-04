import cv2
import numpy as np
import os

# Function to preprocess the image
def preprocess_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur for noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalize the image pixel values
    norm_img = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Enhance contrast using histogram equalization
    enhanced_img = cv2.equalizeHist(norm_img)

    return enhanced_img

# Folder paths
input_folder = 'C:\\Users\\prash\\Desktop\\Medical\\mediscan\\data\\raw\\test\\NRG\\'
processed_folder = 'C:\\Users\\prash\\Desktop\\Medical\\mediscan\\data\\processed\\'

# Create the processed folder if it does not exist
os.makedirs(processed_folder, exist_ok=True)

# Iterate over all the image files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Construct full file path
        file_path = os.path.join(input_folder, filename)

        # Load the image
        img = cv2.imread(file_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Error: Unable to load {file_path}")
            continue
        
        # Preprocess the image
        processed_img = preprocess_image(img)

        # Save the processed image to the processed folder
        save_path = os.path.join(processed_folder, filename)
        cv2.imwrite(save_path, processed_img)

        # Optional: Display the original and processed images (for debugging)
        # cv2.imshow("Original Image", img)
        # cv2.imshow("Processed Image", processed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(f"Processed and saved: {save_path}")

print("All images processed and saved in the 'processed' folder.")
