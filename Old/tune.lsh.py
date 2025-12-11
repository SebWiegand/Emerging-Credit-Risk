"""
tune_lsh.py

Standalone script to tune FAISS LSH hyperparameters using
Cong et al.'s NeighborFinder.optimize_lsh_hyperparameters,
which internally calls eval_index_accuracy.

Usage:
    1. Run your main script once so it saves `embedding_matrix.npy`.
    2. Run this file:
           python tune_lsh.py
    3. Read off the chosen (bits, tables) and paste them into
       N_BITS and N_TABLES in Main_with...py
"""

import os
import numpy as np
from TextualFactors import NeighborFinder


# ============================================================
# 1. Load the embedding matrix
# ============================================================
embedding_matrix_path = "embedding_matrix.npy"  # adjust if saved elsewhere

print(f"Loading embeddings from: {embedding_matrix_path}")
if not os.path.exists(embedding_matrix_path):
    raise FileNotFoundError(
        f"Could not find {embedding_matrix_path}. "
        "Run your main script first so it saves the embedding matrix."
    )

embedding_matrix = np.load(embedding_matrix_path)
print("Embedding matrix shape:", embedding_matrix.shape)


# ============================================================
# 2. Build NeighborFinder (their class)
# ============================================================
nf = NeighborFinder(
    embedding_matrix,
    random_state=42,
    num_queries=1000,  # number of random queries used inside eval_index_accuracy
)

print("NeighborFinder initialized.")


# ============================================================
# 3. Use THEIR hyperparameter optimizer
# ============================================================
# Note: optimize_lsh_hyperparameters internally:
#   - loops over bits & tables
#   - creates LSH indices
#   - calls eval_index_accuracy
#   - stops once accuracy >= target_accuracy
#
# Important: Their accuracy metric is very strict, so realistic targets
# are low (e.g. 0.03–0.06). Here we use 0.06 and cut off max_trials to
# avoid huge indexes that might crash FAISS.
target_accuracy = 0.99
max_trials = 8   # don't go beyond 2**8 bits per table
k_eval = 2       # matches their default k=2

print(
    f"\nRunning Cong et al.'s optimize_lsh_hyperparameters with "
    f"target_accuracy={target_accuracy}, max_trials={max_trials}, k={k_eval}"
)

res = nf.optimize_lsh_hyperparameters(
    target_accuracy=target_accuracy,
    max_trials=max_trials,
    k=k_eval,
)

# ============================================================
# 4. Report the result
# ============================================================
if isinstance(res, tuple) and len(res) == 2:
    ok, info = res
    if ok:
        print("\n================ LSH TUNING RESULT (Cong et al.) ================")
        print(f"Achieved accuracy:           {info['Accuracy']:.4f}")
        print(f"Best bits per table:         {info['Bits per Table']}  (hyperplanes/table)")
        print(f"Best number of hash tables:  {info['Hash Tables']}")
        print("====================================================")
        print("Paste these values into N_BITS and N_TABLES in your Main script.")
    else:
        print(
            "\nCong's optimizer returned but did not reach the target accuracy.\n"
            "Check the printed trials above and, if needed, rerun with a lower "
            "target_accuracy or different max_trials."
        )
else:
    print(
        "\nCong's optimize_lsh_hyperparameters did not return a (ok, info) tuple.\n"
        "This usually means the target_accuracy was never reached within max_trials.\n"
        "Try lowering target_accuracy or inspecting the printed trial accuracies."
    )
