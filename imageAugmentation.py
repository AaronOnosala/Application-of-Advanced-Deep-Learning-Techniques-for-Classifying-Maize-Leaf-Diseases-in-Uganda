#pip install -U albumentation
# importing libraries
import tqdm
import os
import cv2
import albumentations as alb

def data_augmentation(images_path,partition="classname"):

    images=os.path.join(images_path,partition)


    for image in tqdm.tqdm(os.listdir(images)):
            img_path = os.path.join(images, image)
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                   
            for x in range(2):
                        # Albumentation phase

                try:
                    transform = alb.Compose([
                        alb.RandomCrop(width=640, height=640, p=1),
                        alb.HorizontalFlip(p=0.4), 
                        alb.RandomGamma(p=0.2), 
                        alb.RGBShift(p=0.2), 
                        alb.VerticalFlip(p=0.2),
                        alb.ColorJitter(
                            contrast=0,
                            saturation=0.1,
                            hue=0.015,
                            brightness=0.4 )])

                    transformed_instance = transform(image=img)
                            
                    transformed_image = transformed_instance['image']
                          
                    cv2.imwrite( f'{os.path.splitext(image)[0]}_{x}_.jpg', transformed_image)

                except Exception as e:
                        pass
                
data_augmentation('/Users/aarononosala/Documents/Makerere/Classification_maize/train/', partition="MSV")
