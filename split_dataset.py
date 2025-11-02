import os, random, shutil

# ✅ Correct path for your dataset
source_dir = "dataset/TRAIN"   # this matches your actual folder
target_dir = "data"
split_ratio = (0.7, 0.2, 0.1)  # 70% train, 20% val, 10% test

# Create empty folders for train/val/test
for split in ["train", "val", "test"]:
    for cls in os.listdir(source_dir):
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# Split images for each class
for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue  # skip if not folder
    images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    n_train = int(len(images) * split_ratio[0])
    n_val = int(len(images) * split_ratio[1])

    for i, img in enumerate(images):
        src = os.path.join(cls_path, img)
        if i < n_train:
            dest = os.path.join(target_dir, "train", cls, img)
        elif i < n_train + n_val:
            dest = os.path.join(target_dir, "val", cls, img)
        else:
            dest = os.path.join(target_dir, "test", cls, img)
        shutil.copy(src, dest)

print("✅ Dataset split complete! Check the 'data' folder.")
