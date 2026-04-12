
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import sklearn.cluster
from munkres import Munkres
import numpy as np
import torch
import scipy.io as sio
import os
import copy

def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while not stop:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp

def post_proC(C, K, d, alpha):
    C = 0.5 * (C + C.T)
    r = min(d*K + 1, C.shape[0]-1) 
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = sklearn.cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                                  assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def missing_data_generation(input, num_nans):
    # Make a copy of the input array to avoid modifying the original array
    input_copy = input.copy().astype(float)  # Ensure the array is of float type

    # Randomly choose indices to assign NaN values
    indices = np.random.choice(input_copy.size, size=num_nans, replace=False)

    # Assign NaN values to the chosen indices
    input_copy.ravel()[indices] = np.nan

    return input_copy

def convert_nan(input):
    nan_mask = torch.isnan(input)
    x_omega = torch.where(nan_mask, torch.zeros_like(input), input)
    mask_tensor = torch.where(nan_mask, torch.zeros_like(input), torch.ones_like(input))
    return x_omega, mask_tensor


def generate_data(m=100, n = 50, r = 2 , k = 2, noise = 10):
    X = []

    for i in range(k):
        # Generate data for class i
        U = np.random.randint(10, size=(m, r))
        V = np.random.randint(10, size=(n, r))
        x = U @ V.T
        Y = x + noise*np.random.randint(10, size=(m, n))
        # Append the data
        X.append(Y)

    # for i in range(k):
    #     # Generate data for class i
    #     U = np.random.rand(m, r)
    #     V = np.random.rand(n, r)
    #     x = U @ V.T
    #     Y = x + noise*np.random.rand(m, n)
    #     # Append the data
    #     X.append(Y)

    # Stack data vertically for each class
    X = np.hstack(X)
    data = X.T

    full_data = copy.deepcopy(data)
    input_shape = data.shape
    batch_size = input_shape[0]
    flat_layer_size = [input_shape[0]]
    enc_layer_size = [40]
    deco_layer_size = [input_shape[-1]]
    total_datapoints = input_shape[0] * input_shape[1]
    K = num_class = k
    reg1 = 1  # recon loss
    reg2 = 0.001  # auto-encoder loss
    alpha1, alpha2 = 1, 1
    d = r
    lr = 0.005

    # Generate labels according to the number of samples
    labels = np.concatenate([np.repeat(i + 1, n) for i in range(k)])
    true_labels = labels.flatten()

    print("Shape of the generated data:", data.shape)
    print("True labels shape:", true_labels.shape)
    return data, full_data, input_shape, batch_size, total_datapoints, flat_layer_size, enc_layer_size,\
        deco_layer_size, None, None, K, reg1, reg2, alpha1, alpha2, d, lr, num_class * d, true_labels

