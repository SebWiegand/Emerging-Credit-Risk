"""
tune_lsh.py

Standalone script to tune FAISS LSH hyperparameters using
Cong et al.'s NeighborFinder and eval_index_accuracy.

Usage:
    1. Run your main script once so it saves `embedding_matrix.npy`.
    2. Run this file:
           python tune_lsh.py
    3. Read off the best (bits, tables) and paste them into
       N_BITS and N_TABLES in Main_with...py
"""

import numpy as np
from TextualFactors import NeighborFinder


# ============================================================
# 1. Load the embedding matrix
# ============================================================
# Make sure your main script saved this:
#   np.save("embedding_matrix.npy", embedding_matrix)
embedding_matrix_path = "embedding_matrix.npy"

print(f"Loading embeddings from: {embedding_matrix_path}")
embedding_matrix = np.load(embedding_matrix_path)
print("Embedding matrix shape:", embedding_matrix.shape)


# ============================================================
# 2. Build NeighborFinder (their class)
# ============================================================
nf = NeighborFinder(
    embedding_matrix,
    random_state=42,
    num_queries=1000,  # number of random queries used in eval_index_accuracy
)

print("NeighborFinder initialized.")


# ============================================================
# 3. Define a safe grid of LSH hyperparameters to test
# ============================================================
# bits = number of hash bits = number of random hyperplanes per table
# tables = number of independent hash tables
grid_bits = [64, 128, 256]          # adjust if you want finer/coarser search
grid_tables = [2, 4, 8, 16, 32]     # avoid too large values to prevent segfaults

print("\nStarting LSH hyperparameter grid search...")
print(f"Bits grid:    {grid_bits}")
print(f"Tables grid:  {grid_tables}\n")

results = []  # will store tuples (accuracy, bits, tables)


# ============================================================
# 4. Evaluate each (bits, tables) combo using their metric
# ============================================================
for bits in grid_bits:
    for tables in grid_tables:
        print(f"Evaluating LSH with bits={bits}, tables={tables}...")

        # Use Cong et al.'s factory for FAISS LSH
        index = nf.create_lsh_index(bits, tables)

        # Use their evaluation function (k=2 is their default)
        acc = nf.eval_index_accuracy(index, k=2)

        print(f"  -> accuracy = {acc:.4f}")
        results.append((acc, bits, tables))


# ============================================================
# 5. Pick and report the best configuration
# ============================================================
best_acc, best_bits, best_tables = max(results, key=lambda x: x[0])

print("\n================ LSH TUNING RESULT ================")
print(f"Best accuracy (Cong metric): {best_acc:.4f}")
print(f"Best bits per table:         {best_bits}  (hyperplanes/table)")
print(f"Best number of hash tables:  {best_tables}")
print("===================================================")
print("Paste these values into N_BITS and N_TABLES in your Main script.")
