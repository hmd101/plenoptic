import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Union, Callable, Optional
import plenoptic as po
import pyrtools as pt
import numpy as np
import sys
sys.path.append('../tools/')
from plenoptic.tools import img_transforms 


# Function to load and preprocess images
def load_and_preprocess_images(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    images = []
    for filename in glob.glob(os.path.join(image_path, '*.jpg')):  
        img = Image.open(filename).convert('RGB')
        img = transform(img)
        img = img_transforms.rgb_to_opponentcone(img)
        images.append(img)
    print("Images loaded and preprocessed.")
    return torch.stack(images)

def rescale_and_preprocess_images(images: torch.Tensor) -> torch.Tensor:
    img_transforms.rgb_to_opponentcone(images)
    # Rescale images to [0, 1]
    images = images - images.min()
    images = images / images.max()
    return images

# Function to inverse rescale and transform back to RGB
def inverse_rescale_and_transform(images: torch.Tensor) -> List[Image.Image]:
    rgb_images = img_transforms.opponentcone_to_rgb(images)
    rgb_images = [transforms.ToPILImage()(img) for img in rgb_images]
    print("Inverse rescale and transform complete.")
    return rgb_images


# Main function to run synthesis and save images
def main(model_name: str,max_iter: int = 300,init_image = None, 
        ctf_iters_to_check: int = 3, loss_function = po.tools.optim.l2_channelwise, coarse_to_fine: str = 'together', image_path: str = '../../../../../ceph/Datasets/select_color_textures_unsplash',save_path: str = '../../../../../ceph/experiments/color_texture_synth',):
    
    # TODO: Add these arguments from portillasimoncelli constructor:  
    #n_scales: int = 4,
    #n_orientations: int = 4,
    #spatial_corr_width: int = 9
    # crosschannel covariance
    images = load_and_preprocess_images(image_path)
    if init_image is None:
        init_image = torch.rand_like(images[0].unsqueeze(dim=0)) * .01 + images[0].unsqueeze(dim=0).mean()


    model = model_name(images[0].shape)

    metamer = po.synth.MetamerCTF(
        model=model, 
        loss_function=loss_function, 
        init_image=init_image, 
        coarse_to_fine=coarse_to_fine
    )
    
    metamer.synthesize(images, max_iter=max_iter, ctf_iters_to_check=ctf_iters_to_check)
    print("Synthesis complete.")
    synthesized_images = metamer.synthesized_image
    
    rgb_images = inverse_rescale_and_transform(synthesized_images)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory {save_path}")
    
    for i, img in enumerate(rgb_images):
        print(f"Saving image {i}")
        save_filename = f"{ctf_iters_to_check}_{model_name}_{max_iter}.png"
        img.save(os.path.join(save_path, save_filename))

if __name__ == "__main__":
    import argparse
    # path to the data: /mnt/home/hdettki/ceph/Datasets/select_color_textures_unsplash
    # path to color script: /mnt/home/hdettki/code/plenoptic/src/plenoptic/synthesize
    parser = argparse.ArgumentParser(description="Run image synthesis with specified parameters.")
    parser.add_argument("--image_path", type=str, required=False, default= '../../../../../ceph/Datasets/select_color_textures_unsplash',help="Path to the input images.")
    parser.add_argument("--save_path", type=str, required=False,default='../../../../../ceph/experiments/color_texture_synth', help="Path to save the output images.")
    parser.add_argument("--model_name", type=str, required=True, help="Model to be used.")
    #parser.add_argument("--loss_function", type=str, required=False, help="Loss function to be used. Recommended: l2channelwise in po.tools.optim")
    #parser.add_argument("--init_image", type=None, required=False, help="Initial image for synthesis.")
    parser.add_argument("--coarse_to_fine", type=bool, default=False, help="Use coarse to fine synthesis.")
    parser.add_argument("--max_iter", type=int, required=False,default=300, help="Maximum number of iterations. If GPU available, use > 3000.")
    parser.add_argument("--ctf_iters_to_check", type=int, nargs='+', required=False,default=3, help="Iterations to check in coarse to fine synthesis.")
    

    args = parser.parse_args()
    
    main(
        image_path=args.image_path, 
        save_path=args.save_path, 
        model_name=getattr(po.simul, args.model_name), 
        #loss_function=getattr(po.tools.optim, args.loss_function), 
        #init_image=torch.load(args.init_image) if args.init_image else None, 
        coarse_to_fine=args.coarse_to_fine, 
        max_iter=args.max_iter, 
        ctf_iters_to_check=args.ctf_iters_to_check
    )
