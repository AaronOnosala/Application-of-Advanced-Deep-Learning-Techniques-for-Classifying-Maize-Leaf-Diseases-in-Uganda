# Importing necessary libraries
import tqdm  # Provides a progress bar for loops
import os  # Provides functions for interacting with the operating system
import cv2  # OpenCV library for image processing
import albumentations as alb  # Library for image augmentation

def data_augmentation(images_path, partition="classname"):
    """
    Perform data augmentation on images in a specified directory.

    Parameters:
    images_path (str): Path to the directory containing images to be augmented.
    partition (str): Subdirectory within images_path where images are located (default is "classname").

    This function reads images from the specified partition directory, applies a series of augmentations,
    and saves the augmented images with a suffix indicating the augmentation iteration.
    """
    
    # Construct the path to the images directory
    images = os.path.join(images_path, partition)

    # Iterate over each image file in the directory
    for image in tqdm.tqdm(os.listdir(images)):
        img_path = os.path.join(images, image)  # Full path to the image file
        img = cv2.imread(img_path)  # Read the image using OpenCV
        
        # Apply augmentations multiple times
        for x in range(2):
            try:
                # Define the augmentation pipeline
                transform = alb.Compose([
                    alb.RandomCrop(width=640, height=640, p=1),  # Random crop to 640x640 pixels
                    alb.HorizontalFlip(p=0.4),  # Horizontal flip with 40% probability
                    alb.RandomGamma(p=0.2),  # Random gamma adjustment with 20% probability
                    alb.RGBShift(p=0.2),  # RGB color shift with 20% probability
                    alb.VerticalFlip(p=0.2),  # Vertical flip with 20% probability
                    alb.ColorJitter(
                        contrast=0,  # No contrast adjustment
                        saturation=0.1,  # Saturation adjustment
                        hue=0.015,  # Hue adjustment
                        brightness=0.4)  # Brightness adjustment
                ])

                # Apply the transformations to the image
                transformed_instance = transform(image=img)
                transformed_image = transformed_instance['image']
                
                # Save the augmented image with a suffix indicating the augmentation iteration
                cv2.imwrite(f'{os.path.splitext(image)[0]}_{x}_.jpg', transformed_image)
                
            except Exception as e:
                # Print the exception message (if needed for debugging)
                print(f"Error processing {image}: {e}")

# Call the data_augmentation function with the specified path and partition
data_augmentation('/Users/aarononosala/Documents/Makerere/Classification_maize/train/', partition="MSV")
