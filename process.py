import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Load and split the image
def load_and_split_image(image_path):
    # Load the image in RGB
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Error: Image at '{image_path}' could not be loaded. Please check the file path.")
        return None, None, None

    height, width, _ = image.shape

    # Calculate widths for each channel
    width_B = width // 3
    width_G = width // 3
    width_R = width - width_B - width_G  # R takes the rest

    # Split into B, G, R channels
    B = image[:, :width_B, 0]  # (Blue channel)
    G = image[:, width_B:width_B + width_G, 1]  # (Green channel)
    R = image[:, width_B + width_G:, 2]  #  (Red channel)

    return B, G, R

# Align channels
def align_channels(base_channel, shift_channel): 
    result = cv2.matchTemplate(base_channel, shift_channel, cv2.TM_CCOEFF_NORMED)
    # Get the best match position
    _, _, _, max_loc = cv2.minMaxLoc(result)

    translation_matrix = np.float32([[1, 0, max_loc[0]], [0, 1, max_loc[1]]])

    aligned_channel = cv2.warpAffine(shift_channel, translation_matrix, (base_channel.shape[1], base_channel.shape[0]))
    
    return aligned_channel

# Step 3: Combine aligned channels into a color image
def combine_channels(B, G, R):
    color_image = np.dstack([R, G, B])
    return color_image
def process_image(image_path):
    B, G, R = load_and_split_image(image_path)
    if B is None or G is None or R is None:
        return None   
    # Align G and R to B
    G_aligned = align_channels(B, G)
    R_aligned = align_channels(B, R)
    # Combine into a color image
    color_image = combine_channels(B, G_aligned, R_aligned)
    
    return color_image

# to process multiple images 
def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            image_path = os.path.join(directory, filename)
            print(f"Processing: {image_path}")
            colorized_image = process_image(image_path)

            if colorized_image is not None:
                output_path = os.path.join(directory, f"colorized_{filename}")
                cv2.imwrite(output_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)) 
                plt.imshow(colorized_image)
                plt.axis('off')  
                plt.title(f"Colorized: {filename}")
                plt.show()
            else:
                print(f"Failed to process {filename}")

image_path = "images/3.png"  # For a single image
colorized_image = process_image(image_path)

directory_path = "images"  
process_images_in_directory(directory_path)
