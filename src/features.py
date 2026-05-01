import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2

ORB_N_FEATURES = 64
ORB_DESCRIPTOR_SIZE = 32
ORB_FEATURE_DIM = ORB_N_FEATURES * ORB_DESCRIPTOR_SIZE


def _as_image_batch(images):
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError("images must have shape (N, height, width)")
    return images


def _to_uint8_image(image):
    clipped = np.clip(image, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def _hog_feature_dim(image_shape):
    height, width = image_shape
    cell_height, cell_width = (16, 16)
    block_height, block_width = (2, 2)
    orientations = 9

    cells_y = height // cell_height
    cells_x = width // cell_width
    blocks_y = cells_y - block_height + 1
    blocks_x = cells_x - block_width + 1

    if blocks_y <= 0 or blocks_x <= 0:
        return 0

    return blocks_y * blocks_x * block_height * block_width * orientations


def extract_lbp_features(images):
    """Extract normalized Local Binary Pattern histograms from grayscale images."""
    images = _as_image_batch(images)
    features = []

    for image in images:
        uint8_image = _to_uint8_image(image)
        lbp_image = local_binary_pattern(
            uint8_image,
            P=LBP_POINTS,
            R=LBP_RADIUS,
            method=LBP_METHOD,
        )
        histogram, _ = np.histogram(
            lbp_image.ravel(),
            bins=LBP_BINS,
            range=(0, LBP_BINS),
        )
        histogram = histogram.astype(np.float32)
        histogram_sum = histogram.sum()
        if histogram_sum > 0:
            histogram /= histogram_sum

        features.append(histogram)

    if not features:
        return np.empty((0, LBP_BINS), dtype=np.float32)

    return np.asarray(features, dtype=np.float32)


def extract_hog_features(images):
    """Extract Histogram of Oriented Gradients features from grayscale images."""
    images = _as_image_batch(images)
    features = []

    for image in images:
        feature_vector = hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        features.append(feature_vector.astype(np.float32))

    if not features:
        return np.empty((0, _hog_feature_dim(images.shape[1:])), dtype=np.float32)

    return np.asarray(features, dtype=np.float32)


def extract_orb_features(images):
    """Extract fixed-length ORB descriptor vectors from grayscale images."""
    images = _as_image_batch(images)
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    features = []

    for image in images:
        uint8_image = _to_uint8_image(image)
        _, descriptors = orb.detectAndCompute(uint8_image, None)

        feature_vector = np.zeros(ORB_FEATURE_DIM, dtype=np.float32)
        if descriptors is not None:
            descriptors = descriptors[:ORB_N_FEATURES].astype(np.float32)
            descriptor_values = descriptors.ravel()
            feature_vector[: descriptor_values.size] = descriptor_values
            feature_vector /= 255.0

        features.append(feature_vector)

    if not features:
        return np.empty((0, ORB_FEATURE_DIM), dtype=np.float32)

    return np.asarray(features, dtype=np.float32)
