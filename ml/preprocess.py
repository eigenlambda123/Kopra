import json
import numpy as np
from PIL import Image
from pathlib import Path

# 1. SETTINGS
# resize images to 64x64 pixels to reduce the number of features and speed up training.
# 64 * 64 * 3 (RGB colors) = 12,288 numbers per image.
IMG_SIZE = 64

# Resolve paths from the project root so running from ml\ works on Windows.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "new_data_copra" / "train"
ANNOTATIONS_FILE = "_annotations.coco.json"

def load_data(data_dir):
    images = []
    labels = []

    data_dir = Path(data_dir)
    annotations_path = data_dir / ANNOTATIONS_FILE

    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])
    image_entries = coco.get("images", [])

    # Ignore the dataset root category if present.
    valid_categories = [c for c in categories if c.get("name") != "Copra-w4Vb"]
    valid_categories.sort(key=lambda c: c.get("id", 0))

    class_names = [c["name"] for c in valid_categories]
    category_id_to_class_num = {
        c["id"]: idx for idx, c in enumerate(valid_categories)
    }

    # Group category ids by image id; if multiple boxes exist, use first valid class.
    image_id_to_category_ids = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id is None or category_id is None:
            continue
        image_id_to_category_ids.setdefault(image_id, []).append(category_id)

    for entry in image_entries:
        image_id = entry.get("id")
        img_name = entry.get("file_name")
        if image_id is None or not img_name:
            continue

        category_ids = image_id_to_category_ids.get(image_id, [])
        class_num = None
        for category_id in category_ids:
            if category_id in category_id_to_class_num:
                class_num = category_id_to_class_num[category_id]
                break

        if class_num is None:
            continue

        image_path = data_dir / img_name
        if not image_path.exists():
            continue

        try:
            # Open image and convert to RGB
            img = Image.open(image_path).convert("RGB")

            # Resize so all images are uniform
            img = img.resize((IMG_SIZE, IMG_SIZE))

            # Convert image to a list of numbers and normalize
            img_array = np.array(img) / 255.0

            # Flatten 64x64x3 into one long vector
            flattened_img = img_array.flatten()

            images.append(flattened_img)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")

    print(f"Detected classes: {class_names}")

    return np.array(images), np.array(labels)

# 2. RUN THE LOADER
print("Loading images... this might take a second.")
X, y = load_data(DATA_PATH)

# 3. SAVE THE DATA
# We save these as .npy files so you don't have to process the images again.
np.save('X_data.npy', X)
np.save('y_data.npy', y)

print(f"Success! Loaded {X.shape[0]} images.")
if X.shape[0] > 0:
    print(f"Each image is now a vector of {X.shape[1]} numbers.")