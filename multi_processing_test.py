import os
import time
import math
import logging
import numpy as np
import multiprocessing
from multiprocessing import Pool
import SimpleITK as sitk
from radiomics import featureextractor

# Set up logging
logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

# Define image dimensions
channels, height, width = 1, 600, 450

# Generate random image and mask
image = sitk.GetImageFromArray(np.random.randint(255, size=(height, width)))
mask = sitk.GetImageFromArray(np.random.randint(2, size=(height, width)))

# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Function to extract features from image and mask
def extract_features(image_path, mask_path):
    result = extractor.execute(image_path, mask_path, label=1)
    time.sleep(1)
    return result

# Define number of images and paths
num_images = 100
image_paths = [image] * num_images
mask_paths = [mask] * num_images

# Define number of processes
num_processes = multiprocessing.cpu_count() - 1

# Define loops to execute
loops = [0, 0, 1]

# ---------------------------------------------------
# 10_000 -> 310s
# 15 images/processes per batch,
# ~aprox 666 batches
# 0.031 s/img
# ---------------------------------------------------
if loops[0]:
    print("\nfor loop batchsize")
    num_batches = math.ceil(len(image_paths) / num_processes)

    start_time = time.time()
    results = []
    for i in range(num_batches + 1):
        start_index = i * num_processes
        end_index = min((i + 1) * num_processes, len(image_paths))
        batch_image_paths = image_paths[start_index:end_index]
        batch_mask_paths = mask_paths[start_index:end_index]

        with Pool(num_processes) as pool:
            results.append(pool.starmap(extract_features, zip(batch_image_paths, batch_mask_paths)))
    
    end_time = time.time()
    print(f"{(end_time - start_time) / num_images:.3f} s/img")
    print(f"{end_time - start_time:.3f} s\n")


# ---------------------------------------------------
# 10_000 -> 170s
# ~666 images and 15 processes per batch,
# 15 batches
# 0.017 s/img
# ---------------------------------------------------
if loops[1]:
    print("for loop chunksize")

    chunk_size = math.ceil(len(image_paths) / num_processes)
    image_chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]
    mask_chunks = [mask_paths[i:i+chunk_size] for i in range(0, len(mask_paths), chunk_size)]

    start_time = time.time()
    results = []
    for i in range(len(image_chunks)):
        with Pool(num_processes) as pool:
            results.append(pool.starmap(extract_features, zip(image_chunks[i], mask_chunks[i])))

    end_time = time.time()
    print(f"{(end_time - start_time) / num_images:.3f} s/img")
    print(f"{end_time - start_time:.3f} s\n")


# ---------------------------------------------------
# 10_000 -> 163s
# 0.0163 s/img
# ---------------------------------------------------
if loops[2]:
    print("no for loop chunksize")
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = pool.starmap(extract_features, zip(image_paths, mask_paths), chunksize=None)

    end_time = time.time()
    print(f"{(end_time - start_time) / num_images:.3f} s/img")
    print(f"{end_time - start_time:.3f} s\n")
