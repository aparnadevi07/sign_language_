import os
import shutil
import random

# Define paths
dataset_path = "C:/thanu/tancam/dataset/"
train_img_path = os.path.join(dataset_path, "train/images")
train_lbl_path = os.path.join(dataset_path, "train/labels")
valid_img_path = os.path.join(dataset_path, "valid/images")
valid_lbl_path = os.path.join(dataset_path, "valid/labels")

# Create valid folders if they don't exist
os.makedirs(valid_img_path, exist_ok=True)
os.makedirs(valid_lbl_path, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(train_img_path) if f.endswith(".jpg") or f.endswith(".png")]

# Shuffle images
random.shuffle(image_files)

# Define 80-20 split
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
valid_files = image_files[split_index:]

# Function to move files
def move_files(file_list, src_img_path, src_lbl_path, dest_img_path, dest_lbl_path):
    for file in file_list:
        img_src = os.path.join(src_img_path, file)
        lbl_src = os.path.join(src_lbl_path, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img_dest = os.path.join(dest_img_path, file)
        lbl_dest = os.path.join(dest_lbl_path, file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Move image
        shutil.move(img_src, img_dest)

        # Move corresponding label file if it exists
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dest)

# Move validation images & labels
move_files(valid_files, train_img_path, train_lbl_path, valid_img_path, valid_lbl_path)

print(f"✅ Split complete: {len(train_files)} training images, {len(valid_files)} validation images.")
