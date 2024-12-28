import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import cuda, float32
import math

# Set GPU 1 as the active device
cp.cuda.Device(1).use()

@cuda.jit
def compute_distance_matrix_gpu(data, result_matrix, s, start_idx, t_values, dt):
    """Compute the distance matrix using GPU cores for a chunk."""
    i, j = cuda.grid(2)
    n = data.shape[0]
    num_points = t_values.shape[0]

    if start_idx + i >= n or j >= n:
        return

    x1, y1 = data[start_idx + i]
    x2, y2 = data[j]

    dist = 0.0
    for k in range(num_points):
        t = t_values[k]
        norm_pdf = (1.0 / (s * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((t - 0.5) / s) ** 2)
        term1 = (x1 * t + y1 * (1 - t)) * norm_pdf
        term2 = (x2 * t + y2 * (1 - t)) * norm_pdf
        dist += math.sqrt((term1 - term2) ** 2) * dt

    result_matrix[i, j] = dist

if __name__ == "__main__":
    # File paths
    input_file = "northern-ireland dataset.xlsx"
    output_file = "distance_matrix_gpu.xlsx"

    # Load data
    data = pd.read_excel(input_file).to_numpy()
    data = data[:500]  # Limit to 500 rows for testing

    n = 500
    s = 0.304  # Adjustable constant
    num_points = 1000
    T = 10.0  # Truncation limit for integration
    t_values_gpu = cp.linspace(-T, T, num_points, dtype=cp.float32)
    dt = (2 * T) / num_points

    # Chunk size for progress tracking
    chunk_size = 50

    # Transfer data to GPU
    data_gpu = cp.array(data, dtype=cp.float32)
    distance_matrix_gpu = cp.zeros((n, n), dtype=cp.float32)

    # Define grid and block dimensions for GPU parallelization
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (chunk_size + threads_per_block[0] - 1) // threads_per_block[0],
        (n + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Start computation in chunks
    for start_idx in tqdm(range(0, n, chunk_size), desc="Computing distances"):
        try:
            compute_distance_matrix_gpu[blocks_per_grid, threads_per_block](
                data_gpu, distance_matrix_gpu, s, start_idx, t_values_gpu, dt
            )
            cuda.synchronize()  # Ensure all GPU tasks for this chunk are complete
        except cuda.CudaSupportError as e:
            print(f"CUDA kernel launch error: {e}")
            break

    # Transfer results back to CPU
    distance_matrix = cp.asnumpy(distance_matrix_gpu)

    # Save results
    distance_matrix_df = pd.DataFrame(distance_matrix,
                                       columns=[f'Line {i+1}' for i in range(n)],
                                       index=[f'Line {i+1}' for i in range(n)])
    distance_matrix_df.to_excel(output_file)

    print(f"Distance matrix saved to {output_file}")
