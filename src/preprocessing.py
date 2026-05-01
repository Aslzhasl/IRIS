from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def preprocess_image(image_path, image_size=(128, 128)):
    """Read and preprocess a single iris image.

    Returns a normalized grayscale image, or None if the image cannot be read.
    """
    image_path = Path(image_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    try:
        resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    except cv2.error:
        return None

    return blurred.astype(np.float32) / 255.0


def load_dataset(data_dir):
    """Load iris images from class folders.

    Args:
        data_dir: Directory containing one subfolder per class/person.

    Returns:
        X_images: np.ndarray with shape (N, 128, 128), dtype float32.
        y_labels: np.ndarray with integer class labels.
        class_names: List of class/person folder names.
    """
    data_dir = Path(data_dir)
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]

    images = []
    labels = []

    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            image = preprocess_image(image_path)
            if image is None:
                continue

            images.append(image)
            labels.append(label)

    if images:
        X_images = np.stack(images).astype(np.float32)
    else:
        X_images = np.empty((0, 128, 128), dtype=np.float32)

    y_labels = np.array(labels, dtype=np.int64)

    return X_images, y_labels, class_names
