import os
from PIL import Image
import torch
from torchvision import transforms
import sys
import glob
sys.path.append('../tools/')
from img_transforms import rescale, rgb_to_opponentcone

# Define the path to the unsplash images and the output directory
path_to_unsplash = '../../../ceph/Datasets/select_color_textures_unsplash'
output_dir = '../../../ceph/Datasets/select_color_textures_unsplash/opc_rescaled_texture_images'




# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
])

# Function to process and save images
def process_and_save_images(input_dir, output_dir, transform):
    image_files = glob.glob(os.path.join(input_dir, '*.jpg'))  # Assuming images are in .jpg format
    image_files.extend(glob.glob(os.path.join(input_dir, '*.png')))  # Include .png format as well
    print(f"Processing directory: {input_dir}")
    for file in image_files:
        # Step 1: Check if the path to the images exists
        
        #if file.endswith(('png', 'jpg', 'jpeg')):
        print(f"Processing image: {file}")
        # Step 2: Load the image and apply transformations
  
        img = Image.open(image_files).convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Convert to opponent cone color space
        opc_image_tensor = rgb_to_opponentcone(img_tensor)

        # Rescale image
        rescaled_image, _, _ = rescale(opc_image_tensor)

        # Save the transformed image
        rescaled_image = rescaled_image.squeeze(0)  # Remove batch dimension
        save_path = os.path.join(output_dir, file)
        save_image(rescaled_image, save_path)

def save_image(tensor, path):
    """
    Save a tensor as an image file.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor of shape (C, H, W).
    path : str
        Path to save the image file.
    """
    # Convert tensor to PIL image and save
    tensor = tensor.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C), which is what PIL expects
    tensor = tensor.mul(255).clamp(0, 255).byte()  # Scale to [0, 255] and convert to byte
    img = Image.fromarray(tensor.cpu().numpy())
    img.save(path)
    print(f"Image saved at {path}")

# Process and save images
process_and_save_images(path_to_unsplash, output_dir, transform)
print("Images processed and saved successfully.")
