import os
import shutil

# Define paths
image_valid_dir = "C:/thanu/tancam/dataset/images/valid"
label_train_dir = "C:/thanu/tancam/dataset/labels/train"
label_valid_dir = "C:/thanu/tancam/dataset/labels/valid"

# Ensure label directory exists
os.makedirs(label_valid_dir, exist_ok=True)

# Move valid labels to the correct folder
for image_file in os.listdir(image_valid_dir):
    label_file = os.path.splitext(image_file)[0] + ".txt"
    src_path = os.path.join(label_train_dir, label_file)
    dest_path = os.path.join(label_valid_dir, label_file)

    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved {label_file} to valid folder")
