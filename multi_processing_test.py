import os
import time
import logging
import numpy as np
import multiprocessing
import SimpleITK as sitk
from multiprocessing import Pool
from radiomics import featureextractor


c, h, w = 1, 224, 224
image = sitk.GetImageFromArray(np.random.randint(255, size=(h, w)))
mask = sitk.GetImageFromArray(np.random.randint(2, size=(h, w)))
extractor = featureextractor.RadiomicsFeatureExtractor()

logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

def extract_features(image_path, mask_path):
    result = extractor.execute(image_path, mask_path, label=1)
    return result


cpu_count = multiprocessing.cpu_count()
print(cpu_count)
cpu_count = os.cpu_count()
print(cpu_count)

n = 10000
image_paths = [image] * n
mask_paths = [mask] * n
num_processes = multiprocessing.cpu_count() - 1


# ---------------------------------------------------
# 10_000 -> 310s
# 15 images/processes per batch,
# ~aprox 666 batches
# 0.031 s/img
# ---------------------------------------------------

if 0:
    num_batches = len(image_paths) // num_processes

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
    print(end_time - start_time)


# ---------------------------------------------------
# 10_000 -> 170s
# ~666 images and 15 processes per batch,
# 15 batches
# 0.017 s/img
# ---------------------------------------------------
if 0:
    chunk_size = round(len(image_paths) / num_processes)
    image_chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]
    mask_chunks = [mask_paths[i:i+chunk_size] for i in range(0, len(mask_paths), chunk_size)]

    start_time = time.time()
    results = []
    for i in range(len(image_chunks)):
        with Pool(num_processes) as pool:
          results.append(pool.starmap(extract_features, zip(image_chunks[i], mask_chunks[i])))

    end_time = time.time()
    print(end_time - start_time)