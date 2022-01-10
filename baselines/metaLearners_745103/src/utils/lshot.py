import os.path as osp
import numpy as np
import math

def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    N = Y_in.shape[0]
    size_limit = 150000
    if N > size_limit:
        batch_size = 1280
        Y_out = []
        num_batch = int(math.ceil(1.0 * N / batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            tmp = np.exp(Y_in[start:end, :])
            tmp = tmp / (np.sum(tmp, axis=1)[:, None])
            Y_out.append(tmp)
        del Y_in
        Y_out = np.vstack(Y_out)
    else:
        Y_out = np.exp(Y_in)
        Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out

def normalize_2(Y_in):
    Y_in_sum = Y_in.sum(1)[:, np.newaxis]
    Y_in = np.divide(Y_in, Y_in_sum)
    return Y_in


def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    if batch == False:
        temp = (unary * Y) + (-bound_lambda * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
            E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

    return E

def bound_update(unary, kernel, bound_lambda, bound_iteration=20, batch=False):
    """
    """
    oldE = float('inf')
    Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel, bound_lambda, batch)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy()

    l = np.argmax(Y, axis=1)
    return l