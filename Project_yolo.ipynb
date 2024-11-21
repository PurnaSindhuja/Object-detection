# ********************** Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import cv2
import os

!pip install ultralytics

import shutil
import random
import yaml
import from PIL import Image
from IPython.display import display
import torch

# *********************** EDA:

train_data_path =  '.. path to train images /Train_Images'
test_data_path = '. path to test images /Test_Images'
train_annotations_data = pd.read_csv('/.. path to train Annotations/Train Annotations.csv')
test_annotations_data = pd.read_csv('/.. path to test Annotations/Test Annotation.csv')

train_annotations_data = train_annotations_data.rename(columns={
    'Bounding Box coordinates':'xmin',
    'Unnamed: 2': 'ymin',
    'Unnamed: 3': 'xmax',
    'Unnamed: 4': 'ymax'
})
test_annotations_data = test_annotations_data.rename(columns={
    'Bounding Box coordinates':'xmin',
    'Unnamed: 2': 'ymin',
    'Unnamed: 3': 'xmax',
    'Unnamed: 4': 'ymax'
})

# Checking for missing values in Image Class
print("Number of missing values in train Image class:",train_annotations_data['Image class'].isna().sum())
print("Number of missing values in test Image class:",test_annotations_data['Image class'].isna().sum())

# Checking for missing values in Bounding Box Coordinates:
print(train_annotations_data[['xmin', 'ymin', 'xmax', 'ymax']].isnull().sum())
print(test_annotations_data[['xmin', 'ymin', 'xmax', 'ymax']].isnull().sum())

def load_image_data(data_path):
    # Initialize lists to store image paths, class labels, and image names
    image_paths = []
    labels = []
    image_names = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                # Get the class label from the folder name (the last part of the root path)
                label = os.path.basename(root)

                # Full path to the image
                image_path = os.path.join(root, file)

                # Extract just the image name 
                image_name = file

                # Store the image path, label, and image name
                image_paths.append(image_path)
                labels.append(label)
                image_names.append(image_name)

    # Create a DataFrame to keep track of images, labels, and image names
    data = pd.DataFrame({'image_path': image_paths, 'label': labels, 'Image Name': image_names})

    return data

train_data_image = []
test_data_image = []

# Load train and test datasets
train_data_image = load_image_data(train_data_path)
test_data_image = load_image_data(test_data_path)

# Display the first few rows of train and test data to verify
print("Train Data_image:")
print(train_data_image.head(3))
print()
print("\nTest Data_image:")
print(test_data_image.head(3))

# Merge annotations with previously generated train and test data
train_data = pd.merge(train_data_image, train_annotations_data, how='left', on='Image Name')
test_data = pd.merge(test_data_image, test_annotations_data, how='left', on='Image Name')

# Display the first few rows of the merged train data
print("\nTrain Data_image:")
print(train_data.head(3))
print()
print("\nTest Data_image:")
print(test_data.head(3))

# ***************** Analysing Class Distribution
class_counts = train_data['label'].value_counts()
print(class_counts[0:19])

# Top 20-Classes Bar plot
plt.figure(figsize=(15, 6))
class_counts[0:19].plot(kind='bar')
plt.title('Class Distribution of Car Models')
plt.xlabel('Car Model')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)  # Rotate the x-axis labels if there are many classes
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 8))
train_data['label'].value_counts().head(10).plot.pie(autopct='%1.1f%%', startangle=140, cmap='viridis', legend=False)
plt.title('Top 10 Car Models in Terms of Number of Samples')
plt.show()

plt.figure(figsize=(10, 8))
train_data['label'].value_counts().tail(10).plot.pie(autopct='%1.1f%%', startangle=140, cmap='viridis', legend=False)
plt.title('Lowest 10 Car Models in Terms of Number of Samples')
plt.show()

# ********************* Analyzing Normalized Bounding Box Area:
def normalize_bounding_box_areas(data):
    normalized_areas = []

    for idx, row in data.iterrows():
        # Read the image to get its dimensions using the 'image_path' column
        image_path = row['image_path']
        image = cv2.imread(image_path)

        if image is not None:
            image_height, image_width = image.shape[:2]

            # Calculate bounding box area
            box_area = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])

            # Calculate the total image area
            image_area = image_width * image_height

            # Normalize the bounding box area
            normalized_area = box_area / image_area if image_area != 0 else 0
            normalized_areas.append(normalized_area)
        else:
            normalized_areas.append(None)  # Append None if image not found

    # Add normalized areas to the DataFrame
    data['normalized_area'] = normalized_areas
    return data

# Call the function to normalize bounding box areas
normalized_train_data = normalize_bounding_box_areas(train_data)

# Plotting Normalized Bounding Box Areas
plt.figure(figsize=(12, 6))
plt.hist(normalized_train_data['normalized_area'], bins=50)
plt.title('Distribution of Normalized Bounding Box Areas')
plt.xlabel('Normalized Bounding Box Area')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# ****************** Observing Bounding Box Outliers:
def filter_small_large_bounding_boxes(data, small_threshold=0.1, large_threshold=0.9):
    # Filter small bounding boxes
    small_boxes = data[data['normalized_area'] < small_threshold]
    # Filter large bounding boxes
    large_boxes = data[data['normalized_area'] > large_threshold]
    return small_boxes, large_boxes
small_boxes, large_boxes = filter_small_large_bounding_boxes(normalized_train_data)

# Displaying some rows to verify
print("Small bounding boxes:")
print(small_boxes[['Image Name', 'normalized_area']].head())

print("\nLarge bounding boxes:")
print(large_boxes[['Image Name', 'normalized_area']].head())


# ***************** Visualizing Bounding Box Outliers:
def visualize_filtered_bounding_boxes(data, image_folder, num_images=5, title=""):
    plt.figure(figsize=(15, 10))

    for i, (_, row) in enumerate(data.head(num_images).iterrows()):
        # Read the image to get its dimensions using the 'image_path' column
        image_path = row['image_path']
        image = cv2.imread(image_path)

        if image is not None:
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            # Draw the bounding box on the image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color for clarity

            # Convert the image from BGR to RGB for Matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Plot the image with bounding box
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image_rgb)
            plt.title(f"{title}: {row['Image Name']}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_filtered_bounding_boxes(small_boxes, 'path/to/image/folder', num_images=5, title="Small Bounding Boxes")
visualize_filtered_bounding_boxes(large_boxes, 'path/to/image/folder', num_images=5, title="Large Bounding Boxes")

# ***************** Visualizing an image with Bounding boxes
num_images_to_display = int(input("Enter the number of images to display: "))
display_images_with_bbox_inline(train_data, num_images_to_display)

# ***************** Checking for variation in input image sizes
for k in range (1,6,2):
    image = cv2.imread(train_data['image_path'][k])
    print(f"Image dimensions: {image.shape}") # Observation : Hence standardising the image shape is necessary

# *************************************** Yolo v11 for object detection ********************************************************

# ******** Code for creating labels in YOLO format:
for _, row in train_data.iterrows():
    image_path = row['image_path']
    label = int(row['Image class']) - 1  # Adjusted to zero-indexed
    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

    # Read image to get dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Calculate normalized bounding box values
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    # Set label file path in the same directory as the image
    txt_filename = f"{os.path.splitext(row['Image Name'])[0]}.txt"
    txt_path = os.path.join(os.path.dirname(image_path), txt_filename)  # Save in the image directory

    # Write the YOLO format label to the text file
    with open(txt_path, 'w') as f:  # Use 'w' to avoid appending duplicate entries
        f.write(f"{label} {x_center} {y_center} {width} {height}\n")

# ************ Creating Validation dataset from train dataset :
val_data_path = '/... path to create validation data set.../Validation'

# Iterate over each class folder in the train data
for class_folder in os.listdir(train_data_path):
    # Path to class images
    class_train_path = os.path.join(train_data_path, class_folder)

    # Create equivalent class folder in validation path
    class_val_path = os.path.join(val_data_path, class_folder)
    os.makedirs(class_val_path, exist_ok=True)

    if os.path.isdir(class_train_path):
        images = [f for f in os.listdir(class_train_path) if f.endswith(('.jpg', '.png'))]
        val_count = int(len(images) * 0.1)  # 10% for validation

        val_images = random.sample(images, val_count)

        # Copy each selected image and corresponding label to validation directories
        for image in val_images:
            src_image_path = os.path.join(class_train_path, image)
            dst_image_path = os.path.join(class_val_path, image)
            shutil.copy(src_image_path, dst_image_path)

            # Copy corresponding label file
            label_file = os.path.splitext(image)[0] + '.txt'
            src_label_path = os.path.join(class_train_path, label_file)
            dst_label_path = os.path.join(class_val_path, label_file)

            # Check if label file exists before copying
            if os.path.exists(src_label_path):
              shutil.copy(src_label_path, dst_label_path)
            else:
                print(f"Label file {label_file} does not exist for image {image}.")

# class names :
unique_pairs = train_data[['label', 'Image class']].drop_duplicates().sort_values(by='Image class').head(196)
image_class_dict = {index: row['label'] for index, (_, row) in enumerate(unique_pairs.iterrows())}
yaml_names = "names:\n"
for class_id, name in sorted(image_class_dict.items()):
    yaml_names += f"  {class_id}: {name}\n"
print(yaml_names)

# Code to create yaml file:
dataset_path = "/.... Root Directory /Car Images" # root dir

# Creating YAML content
yaml_content = f"""
path: {dataset_path}
train: {train_data_path}
val: {val_data_path}
test: {test_data_path}

{yaml_names}
"""

yaml_file_path = "/.... path to create yaml file /car_dataset.yaml"
with open(yaml_file_path, "w") as file:
    file.write(yaml_content)

# *********************** Yolo training
# Case 1: Using pre-trained model with image size of 224

model_yolo_1 = YOLO("yolo11n.pt")
results_1 = model_yolo_1.train(data='/.. path to yaml file /car_dataset.yaml', epochs=5, imgsz=224, batch=32, save = True, save_period = 1, project='/.. path to save project', name='yolo_train1', workers=8, device=0, patience =	100)

# ************* Fine tuning Yolo :
# Case 2: (Yolo Model 2) Unfreezed 380 layers for training with larger image size of 640x640 :

model_yolo_2 = YOLO("yolo11n.pt")
results_2 = model_yolo_2.train(data='/.. path to yaml file-1 /car_dataset.yaml', epochs=10, imgsz=640, batch=32, freeze=10, workers=8,project='/.. path to save project', name='yolo_train2',save = True, save_period = 1,device = 0)

# saving model 2 prediction results with iou = 0.1 & conf = 0.1
Test_results_2a = model_yolo_2.predict(source='/Path to test images/**', imgsz=640, project = '/... path to save yolo perdictions/Yolo_predictions/model_2/', iou = 0.1, conf = 0.1)
# saving model 2 prediction results with iou = 0.5 & conf = 0.25
Test_results_2b = model_yolo_2p.predict(source='/Path to test images/**', imgsz=640, project = '/... path to save yolo perdictions/Yolo_predictions/model_2/', iou = 0.5, conf = 0.25, stream = True, workers = 4, device = 0 )

# Case 3: ********* (Model 3) With cosine lr
# Model training with yaml - file-1
model_yolo_3 = YOLO("yolo11n.pt")
results_3 = model_yolo_3.train(data='/Path to yaml- file 1', lr0 = 0.001, lrf = 0.1,cos_lr = True, weight_decay = 0.0004, dropout=0.05 ,epochs=50, imgsz=640, batch=64, freeze=10, workers=8,project='.. path to save project',name='yolo_train3',save = True, save_period = 1,device = 0)

 # ---   # Model 3 testing:
# Code to create yaml file2: with test data passed as validation in-order to check validation summary after model.val()
dataset_path = "/.. path to root dir/Car Images" 

# Creating YAML2 with sampled  test images
yaml_content = f"""
path: {dataset_path}
train: {train_data_path}
val: {sampled_images_dir}


{yaml_names}
"""

yaml_file_path = "/.. path to save yaml 2/car_dataset2.yaml"
with open(yaml_file_path, "w") as file:
    file.write(yaml_content)


Test_results_3 = model_yolo_3.val(data = '/.. path to yaml file/car_dataset2.yaml')  # Validating model performance on test dataset

print(Test_results_3.box.maps) # for each category map50-95
print("Precision:", Test_results_3.box.p)
print("Recall:", Test_results_3.box.r)
print("F1 Score:", Test_results_3.box.f1)
print("Mean Precision:", Test_results_3.box.mp)
print("Mean Recall:", Test_results_3.box.mr)

# Metrics
png_files = [
    'runs/detect/val/confusion_matrix_normalized.png',
    'runs/detect/val/R_curve.png',
    'runs/detect/val/PR_curve.png',
    'runs/detect/val/P_curve.png',
    'runs/detect/val/F1_curve.png',
    'runs/detect/val/confusion_matrix.png'
]

# Plot
for file in png_files:
    if os.path.exists(file):
        img = Image.open(file)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.title(os.path.basename(file))
        plt.show()
    else:
        print(f"File not found: {file}")

# Define paths
output_dir = 'runs/detect/val'  # YOLO output directory
drive_output_dir = '/.. path to yolo predictions.. '  # Google Drive directory

# Supported image extensions
image_extensions = ('.png', '.jpg', '.jpeg')

# Check if the output directory exists
if os.path.exists(output_dir):
    # Create the destination directory in Google Drive if it doesn't exist
    if not os.path.exists(drive_output_dir):
        os.makedirs(drive_output_dir)

    # Copy all image files from the YOLO output directory to Google Drive
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(image_extensions):  # Filter by image extensions
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, output_dir)
                dest_path = os.path.join(drive_output_dir, relative_path)

                # Create subdirectories in Google Drive if needed
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)
                print(f"Copied: {file_path} -> {dest_path}")

    print(f"All image outputs have been copied to {drive_output_dir}")
else:
    print(f"Output directory {output_dir} does not exist.")

