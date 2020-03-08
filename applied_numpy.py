import numpy as np


def func_1(n):
    return 2*n + 1


def func_2(n):
    return 50 - 5*n


def build_sequences(min_value, max_value, sequence_number):
    """
    Write a function that can generate the following sequences:
        sequence #1: 2 * n + 1
        sequence #2: 50 - 5 * n
        sequence #3: 2 ** n

    Although this exercises can easily be done with list
    comprehensions, it can be more efficient to use numpy
    (the arange method can be handy here).

    Start by generating all 50 first values for the sequence that
    was selected by sequence_number and return a numpy array
    filtered so that it only contains values in
    [min_value, max_value] (min and max being included)

    :param min_value: minimum value to use to filter the arrays
    :param max_value: maximum value to use to filter the arrays
    :param sequence_number: number of the sequence to return
    :returns: the right sequence as a np.array
    """
    n_min = 0
    n_max = 50
    if sequence_number == 1:
        seq = np.arange(func_1(n_min), func_1(n_max), 2)
    elif sequence_number == 2:
        seq = np.arange(func_2(n_min), func_2(n_max), -5)
    elif sequence_number == 3:
        seq = np.arange(0, 51, 1)
        seq = 2**seq
    flags = [min_value <= x <= max_value for x in seq]
    return np.array(seq[flags])


def moving_averages(x, k):
    """
    Given a numpy vector x of n > k, compute the moving averages
    of length k.  In other words, return a vector z of length
    m = n - k + 1 where z_i = mean([x_i, x_i-1, ..., x_i-k+1])

    Note that z_i refers to value of z computed from index i
    of x, but not z index i. z will be shifted compared to x
    since it cannot be computed for the first k-1 values of x.

    Example inputs:
    - x = [1, 2, 3, 4]
    - k = 3

    the moving average of 3 is only defined for the last 2
    values: [3, 4].
    And z = np.array([mean([1,2,3]), mean([2,3,4])])
        z = np.array([2.0, 3.0])

    :param x: numpy array of dimension n > k
    :param k: length of the moving average
    :returns: a numpy array z containing the moving averages.
    """
    moving_avg = []
    for i in range(k-1, len(x)):
        moving_avg.append(np.mean(x[i-k+1:i+1]))

    return np.array(moving_avg)


def block_matrix(A, B):
    """
    Given two numpy matrices A and B of arbitrary dimensions,
    return a new numpy matrix of the following form:
        [A,0]
        [0,B]

    Example inputs:
        A = [1,2]    B = [5,6]
            [3,4]        [7,8]

    Expected output:
        [1,2,0,0]
        [3,4,0,0]
        [0,0,5,6]
        [0,0,7,8]

    :param A: numpy array
    :param B: numpy array
    :returns: a numpy array with A and B on the diagonal.
    """
    row_a, col_a = A.shape
    row_b, col_b = B.shape

    z_top_right = np.zeros([row_a, col_b])
    z_bottom_left = np.zeros([row_b, col_a])

    a_z = np.hstack((A, z_top_right))
    b_z = np.hstack((z_bottom_left, B))
    block = np.vstack((a_z, b_z))

    return np.array(block)


# A = np.array([[1, 2, 5, 5], [3, 7, 7, 8], [6, 9, 1, 8]])
# B = np.array([[5, 3], [7, 8], [2, 5], [1, 2]])
# print(block_matrix(A, B))
