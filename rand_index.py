import numpy as np

# Define counting matrix here. Each row represents a cluster Z and each column represents a cluster Q.
matrix1 = np.array([
    [4, 0, 0],
    [4, 1, 1]

])

# FOr qeustion 8 on spring 2023 exam, we can define the matrix as follows:
matrix = np.array([
    [4, 6]
])



def calculate_s(matrix):
    # Calculate S using the formula: S = sum(sum(n_km * (n_km - 1) / 2 for each m) for each k)
    return np.sum((matrix * (matrix - 1)) / 2)


def calculate_d(matrix, total_N):
    # Calculate D using the provided formulas
    S = calculate_s(matrix)
    # Sum of elements in each row
    n_k_Z = np.sum(matrix, axis=1)
    # Sum of elements in each column
    n_m_Q = np.sum(matrix, axis=0)

    # Calculate pairs within each cluster of Z
    sum_n_k_Z = np.sum(n_k_Z * (n_k_Z - 1) / 2)
    # Calculate pairs within each cluster of Q
    sum_n_m_Q = np.sum(n_m_Q * (n_m_Q - 1) / 2)

    # Total pairs in N
    total_pairs_N = total_N * (total_N - 1) / 2

    # Calculate D
    D = total_pairs_N - (sum_n_k_Z + sum_n_m_Q) + S
    return D, S


def calculate_rand_index(matrix):
    total_N = np.sum(matrix)
    D, S = calculate_d(matrix, total_N)
    total_pairs = total_N * (total_N - 1) / 2
    rand_index = (S + D) / total_pairs
    return rand_index, S, D


def calculate_jaccard_similarity(matrix):
    total_N = np.sum(matrix)
    D, S = calculate_d(matrix, total_N)
    total_pairs = total_N * (total_N - 1) / 2
    jaccard_similarity = S / (total_pairs - D) if (total_pairs - D) != 0 else 0
    return jaccard_similarity

rand_index, S, D = calculate_rand_index(matrix)
jaccard_similarity = calculate_jaccard_similarity(matrix)

print(f"S = {S}")
print(f"D = {D}")
print(f"Rand Index = {rand_index:.4f}")
print(f"Jaccard Similarity = {jaccard_similarity:.4f}")

