import os
import shutil
import random

# ✅ Your original EuroSAT folder path (update if needed)
original_dataset_dir = "/Users/arghaauddy/Downloads/archive-3/EuroSAT"

# ✅ Your target dataset folder on Desktop inside ISL Project
target_dir = "/Users/arghaauddy/Desktop/ISL Project/dataset"

# ✅ Your class names
classes = ["Forest", "Residential", "River", "Industrial", "Pasture"]

# ✅ For each class
for cls in classes:
    src_dir = os.path.join(original_dataset_dir, cls)
    images = os.listdir(src_dir)
    random.shuffle(images)

    split_idx = int(0.8 * len(images))  # 80% train, 20% val
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # ✅ Create class folders inside train and val
    train_class_dir = os.path.join(target_dir, "train", cls)
    val_class_dir = os.path.join(target_dir, "val", cls)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # ✅ Copy train images
    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_class_dir, img))

    # ✅ Copy val images
    for img in val_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_class_dir, img))

print("✅ Done splitting images into train and val folders!")
