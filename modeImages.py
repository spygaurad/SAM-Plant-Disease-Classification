from PIL import Image
import os

# Path to the folder containing images with no background
input_folder = "cropped_images/B_Removed/"

# Create a new folder to store images with the background
output_folder = "cropped_images/keeping_background/"
os.makedirs(output_folder, exist_ok=True)

# Path to the background image
bg_image_path = "bg.png"

# Resize the background image to 224x224
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((224, 224))

# Iterate through each image in the input folder
for filename in os.listdir(input_folder):
    input_image_path = os.path.join(input_folder, filename)

    # Open the input image
    input_image = Image.open(input_image_path)

    # Resize the input image to 224x224
    input_image = input_image.resize((224, 224))

    # Create a new image with the same size as the background image
    merged_image = Image.new("RGBA", bg_image.size)

    # Paste the background image on the new image
    merged_image.paste(bg_image, (0, 0))

    # Paste the input image (with transparent parts) on the new image
    merged_image.paste(input_image, (0, 0), input_image)

    # Save the merged image with the same filename in the output folder
    output_image_path = os.path.join(output_folder, filename)
    merged_image.save(output_image_path)

print("Images with background applied have been saved in the 'keeping_background' folder.")
