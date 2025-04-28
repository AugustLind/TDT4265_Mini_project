import os
import cv2

def save_image(img, output_path, filename="output.jpg"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, img)

