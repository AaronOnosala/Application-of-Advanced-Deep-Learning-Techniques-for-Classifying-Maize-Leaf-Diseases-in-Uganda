import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_probabilities_and_paths(model, input_path):
    results = model(input_path)
    probabilities = [result.probs.cpu().data.numpy() for result in results]
    file_paths = [result.path for result in results]
    names = results[0].names  # Assuming the category names are consistent across results
    return probabilities, file_paths, names

def average_probabilities(probabilities_list):
    num_images = len(probabilities_list[0])
    combined_probs = np.zeros_like(probabilities_list[0][0])

    for i in range(num_images):
        probs_array = np.array([probs_list[i] for probs_list in probabilities_list])
        combined_probs = np.mean(probs_array, axis=0)

    return combined_probs

# Load the models
model_paths_group1 = [
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train38/weights/last.pt',
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train67/weights/best.pt',
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train40/weights/best.pt'
        ]

model_paths_group2 = [
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train63/weights/last.pt',
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train61/weights/best.pt',
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train68/weights/best.pt',
            '/Users/aarononosala/Documents/MaizeClassification/runs/classify/train71/weights/last.pt'
        ]

models_group1 = [YOLO(model_path) for model_path in model_paths_group1]
models_group2 = [YOLO(model_path) for model_path in model_paths_group2]

# Input path for test data
input_path = '/Users/aarononosala/Documents/Makerere/Classification_maize_test/test'

all_probs_lists_group1 = []
all_file_paths_lists_group1 = []
all_names_lists_group1 = []

all_probs_lists_group2 = []
all_file_paths_lists_group2 = []
all_names_lists_group2 = []

 # Get probabilities and file paths from each model in group 1
for model in models_group1:
    probs_list, file_paths, names = get_probabilities_and_paths(model, input_path)
    all_probs_lists_group1.append(probs_list)
    all_file_paths_lists_group1.append(file_paths)
    all_names_lists_group1.append(names)

 # Get probabilities and file paths from each model in group 2
for model in models_group2:
    probs_list, file_paths, names = get_probabilities_and_paths(model, input_path)
    all_probs_lists_group2.append(probs_list)
    all_file_paths_lists_group2.append(file_paths)
    all_names_lists_group2.append(names)

# Ensure all lists have the same length and consistency for group 1
num_images = len(all_probs_lists_group1[0])
for probs_list in all_probs_lists_group1:
    assert len(probs_list) == num_images, "Mismatch in number of images processed by the models in group 1."

file_paths1 = all_file_paths_lists_group1[0]
for file_paths in all_file_paths_lists_group1:
    assert file_paths == file_paths1, "Mismatch in file paths between model results in group 1."

names1 = all_names_lists_group1[0]
for names in all_names_lists_group1:
    assert names == names1, "Mismatch in category names between model results in group 1."

 # Ensure all lists have the same length and consistency for group 2
for probs_list in all_probs_lists_group2:
    assert len(probs_list) == num_images, "Mismatch in number of images processed by the models in group 2."

file_paths2 = all_file_paths_lists_group2[0]
for file_paths in all_file_paths_lists_group2:
    assert file_paths == file_paths2, "Mismatch in file paths between model results in group 2."

names2 = all_names_lists_group2[0]
for names in all_names_lists_group2:
    assert names == names2, "Mismatch in category names between model results in group 2."

 # Average probabilities for group 1
averaged_probs_group1 = []
for i in range(num_images):
    probs_array_group1 = np.array([probs_list[i] for probs_list in all_probs_lists_group1])
    averaged_probs_group1.append(np.mean(probs_array_group1, axis=0))

 # Average probabilities for group 2
averaged_probs_group2 = []
for i in range(num_images):
    probs_array_group2 = np.array([probs_list[i] for probs_list in all_probs_lists_group2])
    averaged_probs_group2.append(np.mean(probs_array_group2, axis=0))

# Final average of the two combined models
final_averaged_probs = []
for i in range(num_images):
    combined_probs_array = np.array([averaged_probs_group1[i], averaged_probs_group2[i]])
    final_averaged_probs.append(np.mean(combined_probs_array, axis=0))

# Initialize lists to store results
file_names = []
results_dicts = []

for i in range(num_images):
    file_name = file_paths1[i].split('/')[-1]
    file_names.append(file_name)

    file_results = {names1[j]: 0 for j in range(len(final_averaged_probs[i]))}
    top1_idx = np.argmax(final_averaged_probs[i])
    file_results[names1[top1_idx]] = 1

    results_dicts.append(file_results)

data = pd.DataFrame(results_dicts)
data['filename'] = file_names

print(data)

# Save the DataFrame to a CSV file
data.to_csv('Aaron Onosala.csv', index=False)
