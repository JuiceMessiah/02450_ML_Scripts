import numpy as np

d = np.array([[-0.4, -0.8],
              [-0.9, 0.3],
              [0, 0.9],
              [1, -0.1],
              [0.8, -0.7],
              [0.1, 0.8]])


def p_norm(d: np.array, p: int) -> np.array:
    """
    Calculate the p-norm of a matrix d

    :param d: np.array. Array of x and y coordinates
    :param p: int. The p value for the p-norm
    :return: np.array. p-distance of the matrix d
    """
    if p == -1:
        # p = infinity
        return np.max(np.abs(d), axis=1)
    else:
        return np.linalg.norm(d, ord=p, axis=1)


print(p_norm(d, 1))
print(p_norm(d, 2))
print(p_norm(d, -1))
