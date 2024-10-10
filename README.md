# Maize Diseases Classification Project

* Collaboration: [Makerere Artificial Intelligence Lab](https://air.ug), [Lacuna Fund](https://lacunafund.org/)
  
<img width="373" alt="Screenshot 2024-08-12 at 11 29 06" src="https://github.com/user-attachments/assets/96adf8ff-e206-4d07-9b30-f9971a68d0d2"> <img width="399" alt="Screenshot 2024-08-13 at 19 20 29" src="https://github.com/user-attachments/assets/fcf16b19-2048-49f0-b4c9-721c632b2fb7"> <img width="392" alt="Screenshot 2024-08-12 at 11 28 05" src="https://github.com/user-attachments/assets/03e911e9-1bee-4a29-a2fc-3168df487c22"> 

## Overview
The maize disease classification project aimed to develop a robust model capable of accurately identifying and classifying diseases in maize leaves. Given the critical importance of maize as a staple crop, early detection and classification of leaf diseases are crucial for preventing significant yield losses and ensuring food security. The project focused on distinguishing between healthy leaves and those affected by four distinct diseases:

1. FAW - Images show leaves that have suffered [fallarmy worm damage.](https://agriculture.go.ug/wp-content/uploads/2019/05/FAW-Brochure_MAAIF_DCP_revised_April_2018.pdf)
2. MLB - Images show leaves affected by [Maize Leaf Blight](https://lfl.bayern.de/ips/blattfruechte/050760/index.php#:~:text=First%20symptoms%20on%20maize%20plants,green%20to%20light%20brown%20lesions.)
3. MLN - Images show leaves affected by [Maize Lethal Necrosis](https://www.cabidigitallibrary.org/doi/10.1079/cabicompendium.119663)
4. MSV - Images show leaves affected by [Maize Streak Virus](https://www.cabidigitallibrary.org/doi/10.1079/cabicompendium.32620)

![maize](https://github.com/user-attachments/assets/e315adb6-7fd0-4bb2-a6c3-6853cf8b06a5)

## Problem Solved
Maize is a staple food crop in many parts of the world, especially in Africa. The primary objective of this project was to create a machine learning model that could reliably classify images of maize leaves into one of five categories: healthy, FAW, MLB, MLN, and MSV. This model would assist farmers and agricultural experts in early disease detection, enabling timely intervention and treatment. However, various diseases significantly affect maize yield and quality, posing a threat to food security. Early and accurate disease diagnosis is crucial for effective management and control of these diseases. This project addressed the need for an automated, reliable, and accessible solution for identifying maize leaf diseases for Ugandan farmers.

##  Approach, AI Tools and Techniques Applied 

### Framework Used In This Project
<img width="741" alt="Screenshot 2024-08-12 at 11 46 01" src="https://github.com/user-attachments/assets/47820292-eb6f-44ce-b30d-65314b952538">

To achieve this objective, various machine learning frameworks were explored, including CNN and VGG19. However, YOLOv8 (You Only Look Once) emerged as the best-performing model due to its superior accuracy and efficiency. The project involved several key steps:

### Data Collection and Preparation:

1. __Initial Dataset:__ The initial dataset comprised 26,043 images of maize leaves and can be found [here.](https://storage.googleapis.com/air-lab-hackathon/Maize/classification/Classification_maize.zip) This dataset represents 4 maize leaf disease classes i.e. MSV, FAW, MLN MLB and 1 class for HEALTHY images. The dataset is split into three sets:
     + Train with 26063 images.
     +  Validation with 7445 images.
     +  Test with 3729 images.
2. __Data Augmentation:__ To enhance the model's performance and robustness, data augmentation techniques were applied, expanding the dataset to 50,000 images on the training images only. Techniques included random cropping, horizontal and vertical flipping, random gamma adjustments, RGB shifts, and color jittering.

### Model Training and Evaluation:

1. __Model Selection:__ After experimenting with various frameworks, YOLOv8 was selected for its superior performance.
    + __YOLOv8 (You Only Look Once) :__ Utilized the YOLOv8 model for its superior performance in image classification tasks. This model was particularly chosen for its speed and accuracy.
   + __Data Augmentation:__ Applied various augmentation techniques using the Albumentations library to enhance the dataset and improve model generalization.
   + __Multiple Model Training and Ensemble:__ Trained multiple models and combined their predictions to achieve more robust and accurate results.
2. __Training:__ The model was trained for 100 epochs with an image size of 640 pixels, using the augmented dataset.
3. __Evaluation:__ The model's performance was evaluated using metrics such as training loss, validation loss, and validation accuracy.

## YOLO v8 Network Structures
<img width="912" alt="Screenshot 2024-08-09 at 10 52 41" src="https://github.com/user-attachments/assets/a71dfa95-d5ce-4161-a749-97dc06a707bc">

### Model Ensemble and Probability Averaging:

Two groups of models were trained independently, and their probabilities were averaged to enhance accuracy and reliability. Final predictions were obtained by averaging the probabilities from both groups, ensuring consistent and accurate classification.

## Challenges Faced and Overcoming Them
The project encountered several challenges:

### Time and Resource Constraints:

* __Challenge:__ Training deep learning models, especially with large datasets, required significant computational resources and time.
* __Solution:__ Efficient resource management and leveraging high-performance computing environments helped mitigate these constraints.

### Model Selection and Fine-Tuning:

* __Challenge:__ Selecting the optimal model and fine-tuning its parameters for the best performance was a complex and iterative process.
* __Solution:__ Systematic experimentation with different frameworks and hyperparameter tuning led to the selection of YOLOv8, which provided the best results.

### Data Augmentation:

* __Challenge:__ Ensuring that the augmented dataset remained representative and did not introduce noise or bias.
* __Solution:__ Careful application of augmentation techniques preserved the dataset's integrity and improved the model's robustness.

## Image Result of 4 class of the 5 class samples
<img width="982" alt="Screenshot 2024-08-09 at 08 56 00" src="https://github.com/user-attachments/assets/fcc43bba-9afd-40c6-8f89-43621ece8182">

## Outcomes and Achievements
The project achieved significant success, with the final model attaining an impressive accuracy of 99.7%, up from an initial 94%. This remarkable improvement underscores the effectiveness of data augmentation. The model's high accuracy demonstrates its potential to significantly aid in early disease detection, ultimately benefiting farmers and agricultural stakeholders.

## Quantified Success

- __Accuracy Improvement:__ Enhanced model accuracy from 94% to 99.7% through data augmentation.
- __Dataset Expansion:__ Augmented the initial dataset from 26,043 to 50,000 images, increasing the model's training data and improving its performance.

## Conclusion
This project showcases the power of advanced AI techniques, particularly YOLOv8, in solving real-world agricultural problems. The challenges faced and overcome highlight the importance of perseverance, systematic experimentation, and efficient resource management. The model developed has the potential to make a tangible impact in the field of agriculture, aiding in the early detection and treatment of maize diseases.

By leveraging cutting-edge AI tools and techniques, we have demonstrated how technology can contribute to enhancing food security and supporting sustainable agricultural practices.





