import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import pandas as pd
import time
import os
from multiprocessing import Pool, Value, Lock
from tqdm import tqdm  # Import tqdm

os.chdir('/user1/postdoc/miu/paramita2000_p/Gaussian-Distance/')


# Global variables for multiprocessing
counter = None
lock = None
progress_bar = None  # Add a progress bar

def gaussian_dist(t, x1, y1, s1, x2, y2, s2):
    return np.linalg.norm((x1 * t + y1 * (1 - t)) * norm.pdf(t, 0.5, s1) -
                          (x2 * t + y2 * (1 - t)) * norm.pdf(t, 0.5, s2))

def sigma(alpha=0.05):
    z_95 = norm.ppf(1 - alpha)
    z_5 = norm.ppf(alpha)
    return 1 / (z_95 - z_5)

def distance(x1, y1, x2, y2, s):
    result, _ = quad(gaussian_dist, -np.inf, np.inf, args=(x1, y1, s, x2, y2, s))
    return result

def compute_distance_row(args):
    """Compute one row of the distance matrix and update the counter."""
    global counter, lock, progress_bar  # Access the global variables
    i, data, s = args
    n = len(data)
    row = np.zeros(n)
    x11, y11, x12, y12 = data[i]
    x1 = np.array([x11, y11])
    y1 = np.array([x12, y12])
    for j in range(n):
        x21, y21, x22, y22 = data[j]
        x2 = np.array([x21, y21])
        y2 = np.array([x22, y22])
        row[j] = distance(x1, y1, x2, y2, s)

    # Update counter and progress bar
    with lock:
        counter.value += 1
        progress_bar.update(1)  # Update the tqdm progress bar
    return i, row

def init_worker(shared_counter, shared_lock, shared_progress_bar):
    """Initialize global variables in worker processes."""
    global counter, lock, progress_bar
    counter = shared_counter
    lock = shared_lock
    progress_bar = shared_progress_bar

if __name__ == "__main__":
    file_path = "northern-ireland dataset.xlsx"
    data = pd.read_excel(file_path).to_numpy()
    s = sigma()
    n = len(data)
    distance_matrix = np.zeros((n, n))

    # Shared counter and lock initialized globally
    shared_counter = Value('i', 0)
    shared_lock = Lock()

    # Prepare multiprocessing
    start = time.time()
    with tqdm(total=n, desc="Processing rows", position=0, leave=True) as progress_bar:
        with Pool(processes=200, initializer=init_worker, initargs=(shared_counter, shared_lock, progress_bar)) as pool:
            results = pool.map(compute_distance_row, [(i, data, s) for i in range(n)])

    end = time.time()
    
    # Collect results
    for i, row in results:
        distance_matrix[i, :] = row

    # Save results
    distance_matrix_df = pd.DataFrame(distance_matrix,
                                       columns=[f'Line {i+1}' for i in range(n)],
                                       index=[f'Line {i+1}' for i in range(n)])
    output_file = "distance_matrix.xlsx"
    distance_matrix_df.to_excel(output_file)
    print(f"Distance matrix saved to {output_file}")
    print(f"Total computation time: {end - start:.2f} seconds")
