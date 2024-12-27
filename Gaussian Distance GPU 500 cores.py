import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm
from numba import cuda
import os

# Set GPU 1 as the active device
cp.cuda.Device(1).use()

@cuda.jit
def compute_distance_matrix_gpu(data, result_matrix):
    """Compute the distance matrix using GPU cores."""
    i, j = cuda.grid(2)
    n = data.shape[0]

    if i < n and j < n:
        x1, y1, x2, y2 = data[i], data[j]

        def gaussian_dist(t):
            return cp.linalg.norm(
                (x1[0] * t + x1[1] * (1 - t)) * norm.pdf(t, 0.5, 0.304) -
                (x2[0] * t + x2[1] * (1 - t)) * norm.pdf(t, 0.5, 0.304)
            )

        result, _ = quad(gaussian_dist, -np.inf, np.inf)
        result_matrix[i, j] = result
        

if __name__ == "__main__":
    # File paths
    input_file = "northern-ireland dataset.xlsx"
    output_file = "distance_matrix_gpu.xlsx"

    # Load data
    data = pd.read_excel(input_file).to_numpy()
    data = data[:500]  # Limit to 20 rows for testing

    n = 500

    # Transfer data to GPU
    data_gpu = cp.array(data)
    distance_matrix_gpu = cp.zeros((n, n), dtype=cp.float32)

    # Define grid and block dimensions for GPU parallelization (500 cores)
    threads_per_block = (25, 20)  # ~500 threads per block
    blocks_per_grid = (
        (n + threads_per_block[0] - 1) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Start computation
    with tqdm(total=n * n, desc="Computing distances") as pbar:
        compute_distance_matrix_gpu[blocks_per_grid, threads_per_block](data_gpu, distance_matrix_gpu)
        pbar.update(n * n)

    # Transfer results back to CPU
    distance_matrix = cp.asnumpy(distance_matrix_gpu)

    # Save results
    distance_matrix_df = pd.DataFrame(distance_matrix,
                                       columns=[f'Line {i+1}' for i in range(n)],
                                       index=[f'Line {i+1}' for i in range(n)])
    distance_matrix_df.to_excel(output_file)

    print(f"Distance matrix saved to {output_file}")
