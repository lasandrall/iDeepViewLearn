# python function
import numpy as np
# import matplotlib.pyplot as plt
import torch
import math
import random
from sklearn import svm
import itertools
import os
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored


# set random seed
def torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# normalize training or testing data
def myNormalize(input, train=None):
    data = copy.copy(input)
    h, w = data.shape[0], data.shape[1]
    if train is None:
        for i in range(w):
            tmp_mean = np.mean(data[:, i])
            tmp_std = np.std(data[:, i])
            data[:, i] = (data[:, i] - tmp_mean) / (tmp_std + 1e-8)
    else:
        assert w == train.shape[1], 'Testing data should have the same number of features as the training data!'
        for i in range(w):
            tmp_mean = np.mean(train[:, i])
            tmp_std = np.std(train[:, i])
            data[:, i] = (data[:, i] - tmp_mean) / (tmp_std + 1e-8)
    return data

# this works
# Input format
# Xdata       list of D views
# edges       is a list of D (D is number of views) elements of edge information (M x 2, M is number of edges) for the views.
#            if no edge information, set as [] (empty). This will use original optimization to Z and reconstructed data
# vweight     is a list of D weight information for the weight of the vertices
#            (variables) for each D.

def normalized_Laplacian(Xdata, edges, vweight):
    D = len(Xdata)  # list of D views
    mynormalizedLaplacian = list(range(D))

    for d in range(D):
        nRows, nVars = Xdata[d].size()

        # if edge information is empty, then no graph information
        # utilizes original coding
        if len(edges[d]) != 0:
            edgesd = edges[d]
            vweightd = vweight[d]
            
            
            # if there are weights, we consider the normalized laplacian of a weighted graph
            if len(vweightd) != 0:
                WeightM = torch.zeros(nVars, nVars)  # to be used to form the weights of the matrix
                for j in range(len(edgesd)):
                    indI = int(edgesd.iloc[j, 0])
                    indJ = int(edgesd.iloc[j, 1])
                    WeightM[indI - 1, indJ - 1] = vweightd[j]
                    WeightM[indJ - 1, indI - 1] = vweightd[j]

                Dv = torch.sum(WeightM, 1)
                L = torch.diag(Dv) - WeightM  # think about how to store this as a sparse matrix
                notZero = Dv != 0
                Dv2 = torch.zeros(len(Dv))
                Dv2[notZero] = Dv[Dv != 0] ** (-0.5)

                Dv = torch.diag(Dv2)
                normalizedLaplacian = torch.matmul(Dv, torch.matmul(L, Dv))  # normallized Laplacian of a weighted graph

                mynormalizedLaplacian[d] = normalizedLaplacian
            # if there are no weights, we consider the normalized laplacian of a  graph
            elif len(vweightd) == 0:
                AdjM = torch.zeros(nVars, nVars)
                for j in range(len(edgesd)):
                    indI = int(edgesd.iloc[j, 0])
                    indJ = int(edgesd.iloc[j, 1])
                    AdjM[indI - 1, indJ - 1] = 1
                    AdjM[indJ - 1, indI - 1] = 1

                Dv = torch.sum(AdjM, 1)
                L = torch.diag(Dv) - AdjM
                notZero = Dv != 0
                Dv2 = torch.zeros(len(Dv))
                Dv2[notZero] = Dv[Dv != 0] ** (-0.5)

                Dv = torch.diag(Dv2)
                normalizedLaplacian = torch.matmul(Dv, torch.matmul(L, Dv))  # normallized Laplacian of a weighted graph

                mynormalizedLaplacian[d] = normalizedLaplacian
        elif len(edges[d]) == 0:
            mynormalizedLaplacian[d] = torch.eye(nVars)

    return mynormalizedLaplacian

def run(X_train, y_train, comb, X_tune=None, y_tune=None, important=None, outtype=None, top_rate=0.1, edged=None, vWeightd=None, epochs=1000, plot=False, train=False, device='cpu', normalization=True):
    # hyper-parameters
    K = comb[0]
    lr_net = comb[1]
    lr_z = comb[2]
    #C = comb[3], for classifier
    l1 = comb[4]

    # data dimension
    D = len(X_train)
    n = X_train[0].shape[0]

    # normalized training
    X_train_normalized = []

    # numbers of important features in each view
    gt = []

    # numbers of features in each view
    p = []
    for i in range(D):
        if normalization:
            X_train_normalized.append(torch.tensor(myNormalize(X_train[i])).to(device))
        else:
            X_train_normalized.append(torch.tensor(X_train[i]).to(device))
        p.append(X_train[i].shape[1])
        if important != None:
            if isinstance(important, int):
                gt.append(important)
            else:
                gt.append(important[i])
        else:
            gt.append(int(X_train[i].shape[1] * top_rate))

    if edged != None:
        lap = normalized_Laplacian(X_train_normalized, edged, vWeightd)

    def norm_21(x):
        return torch.norm(x, dim=0).sum()

    def unsupervised_loss_1(X, Z_pred, model):
        loss = 0
        if edged != None:
            for i in range(D):
                loss += norm_21(X[i] - model[i](Z_pred)) + l1 * norm_21(model[i](Z_pred) @ lap[i].to(device))
        else:
            for i in range(D):
                loss += norm_21(X[i] - model[i](Z_pred)) + l1 * norm_21(model[i](Z_pred))
        return loss / X[0].shape[0]

    def unsupervised_loss_2(X, Z_pred, model):
        loss = 0
        for i in range(D):
            loss += torch.norm(X[i] - model[i](Z_pred)) ** 2
        return loss / X[0].shape[0]

    unsupervised_history_1 = []
    unsupervised_history_2 = []

    # first-stage reconstruction using data with all features
    # initial Z, NNs
    Z_1 = torch.randn(n, K, requires_grad=True, device=device)
    model_1 = list(range(D))
    optimizer_1 = list(range(D))
    for i in range(D):
        model_1[i] = torch.nn.Sequential(
            torch.nn.Linear(K, 64),
            torch.nn.GroupNorm(16, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 256),
            torch.nn.GroupNorm(64, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, p[i]),
        ).to(device)
        optimizer_1[i] = torch.optim.Adam(model_1[i].parameters(), lr_net)
    optimizer_z_1 = torch.optim.Adam([Z_1], lr_z)

    for epoch in range(epochs):
        unsupervised = unsupervised_loss_1(X_train_normalized, Z_1, model_1)
        unsupervised_history_1.append(unsupervised.item())
        for i in range(D):
            optimizer_1[i].zero_grad()
        optimizer_z_1.zero_grad()
        unsupervised.backward()
        for i in range(D):
            optimizer_1[i].step()
        optimizer_z_1.step()

    # if plot:
    #     plt.plot(unsupervised_history_1)
    #     plt.title("Unsupervised History 1")
    #     plt.show()
    #     plt.close()

    # feature selection
    features = []
    X_train_fs = []

    for d in range(D):
        pre = abs(model_1[d](Z_1))
        norm_pre = torch.norm(pre, dim=0)
        features_pre = (sorted(range(len(norm_pre)), key=lambda i: norm_pre[i]))[::-1]
        features_pre = features_pre[:gt[d]]
        features_pre = sorted(features_pre)
        features.append(features_pre)
        X_train_fs.append(X_train_normalized[d][:, features_pre])

    # second-stage reconstruction using data without unimportant features
    model_2 = list(range(D))
    optimizer_2 = list(range(D))
    for i in range(D):
        model_2[i] = torch.nn.Sequential(
            torch.nn.Linear(K, 64),
            torch.nn.GroupNorm(16, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 256),
            torch.nn.GroupNorm(64, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, gt[i]),
        ).to(device)
        optimizer_2[i] = torch.optim.Adam(model_2[i].parameters(), lr_net)
    Z_2 = torch.randn(n, K, requires_grad=True, device=device)
    optimizer_z_2 = torch.optim.Adam([Z_2], lr_z)
    for epoch in range(epochs):
        unsupervised = unsupervised_loss_2(X_train_fs, Z_2, model_2)
        unsupervised_history_2.append(unsupervised.item())
        for i in range(D):
            optimizer_2[i].zero_grad()
        optimizer_z_2.zero_grad()
        unsupervised.backward()
        for i in range(D):
            optimizer_2[i].step()
        optimizer_z_2.step()
        # projected gradient descent (PGD)
        with torch.no_grad():
            tmp_norm = torch.norm(Z_2, dim=1).reshape(-1, 1)
            tmp_norm = torch.clamp(tmp_norm, min=1)
            Z_2.data = Z_2.data / tmp_norm
    # if plot:
    #     plt.plot(unsupervised_history_2)
    #     plt.title("Unsupervised History 2")
    #     plt.show()
    #     plt.close()
    
    GZ = []
    RZprime = []
    if device != 'cpu':
        Ztrain = Z_1.cpu().detach().numpy()
        Zprime = Z_2.cpu().detach().numpy()
        for d in range(D):
            GZ.append(model_1[d](Z_1).cpu().detach().numpy())
            RZprime.append(model_2[d](Z_2).cpu().detach().numpy())
    else: 
        Ztrain = Z_1.detach().numpy()
        Zprime = Z_2.detach().numpy()
        for d in range(D):
            GZ.append(model_1[d](Z_1).detach().numpy())
            RZprime.append(model_2[d](Z_2).detach().numpy())

    #### USE OUTTPYE
    # outtype = NULL => no tuning, discard the rest of the code and return here
    if outtype is None:
        clf = None
        return features, model_2, clf, Ztrain, GZ, Zprime, RZprime
    
    # Otherwise, tune
    # Create estimators for categorical(clf), continuous(regr), and survival(estimator)
    if outtype == "categorical":
        C = comb[3]
        # train a SVM classifier using learned latent code Z of training data
        clf = svm.SVC(kernel='rbf', C=C, gamma='scale')
        if device != 'cpu':
            clf.fit(Z_2.cpu().detach().numpy(), y_train)
        else:
            clf.fit(Z_2.detach().numpy(), y_train)
    elif outtype == "continuous":
        C = comb[3]
        # train a SVM.SVR
        regr = svm.SVR(kernel='rbf', C=C, gamma='scale')
        if device != 'cpu':
            regr.fit(Z_2.cpu().detach().numpy(), y_train)
        else:
            regr.fit(Z_2.detach().numpy(), y_train)
    elif outtype == "survival":
        y_train_new = np.array([(status, survival) for status, survival in y_train], dtype=np.dtype([('Status','?'), ('Time','<f8')]))
        alpha = comb[3]
        estimator = FastKernelSurvivalSVM(alpha=alpha, rank_ratio=1, kernel='rbf',
                                  max_iter=1000, tol=1e-5)
        if device != 'cpu':
            estimator.fit(Z_2.cpu().detach().numpy(), y_train_new)
        else:
            estimator.fit(Z_2.detach().numpy(), y_train_new)
            
    if X_tune == None:
        if outtype == "categorical":
            return features, model_2, clf, Ztrain, GZ, Zprime, RZprime
        elif outtype == "continuous":
            return features, model_2, regr, Ztrain, GZ, Zprime, RZprime
        elif outtype == "survival":
            return features, model_2, estimator, Ztrain, GZ, Zprime, RZprime
        elif outtype == "nothing":
            return features, model_2, unsupervised_history_2[-1], Ztrain, GZ, Zprime, RZprime
    
    # testing stage
    n_tune = X_tune[0].shape[0]

    # normalized tuning data
    X_tune_normalized = []
    for i in range(D):
        #X_tune_normalized.append(torch.tensor(myNormalize(X_tune[i], X_train[i])).to(device))
        if normalization:
            X_tune_normalized.append(torch.tensor(myNormalize(X_tune[i], X_train[i])).to(device))
        else:
            X_tune_normalized.append(torch.tensor(X_tune[i]).to(device))
            
    # normalized tuning data without unimportant features
    X_tune_fs = []
    for d in range(D):
        X_tune_fs.append(X_tune_normalized[d][:, features[d]])
    # get Z from testing data
    Z_tune = torch.randn(n_tune, K, requires_grad=True, device=device)
    optimizer_z_tune = torch.optim.Adam([Z_tune], lr_z)
    Z_history = []
    

    for i in range(epochs):
        loss_z = unsupervised_loss_2(X_tune_fs, Z_tune, model_2)
        Z_history.append(loss_z.item())
        optimizer_z_tune.zero_grad()
        loss_z.backward()
        optimizer_z_tune.step()
        # projected gradient descent (PGD)
        with torch.no_grad():
            tmp_norm = torch.norm(Z_tune, dim=1).reshape(-1, 1)
            tmp_norm = torch.clamp(tmp_norm, min=1)
            Z_tune.data = Z_tune.data / tmp_norm
    # if plot:
    #     plt.plot(Z_history)
    #     plt.title("Z History")
    #     plt.show()
    #     plt.close()
    
    if outtype == "categorical":
        # pass the learned Z of testing data to the trained classifer to do prediction
        if device != 'cpu':
            prediction = clf.predict(Z_tune.cpu().detach().numpy())
        else:
            prediction = clf.predict(Z_tune.detach().numpy())
        acc = (prediction.reshape(-1) == y_tune).sum().item() / n_tune
        return acc, features
    elif outtype == "continuous":
        if device != 'cpu':
            prediction = regr.predict(Z_tune.cpu().detach().numpy())
        else:
            prediction = regr.predict(Z_tune.detach().numpy())
        mse = mean_squared_error(y_tune, prediction)
        #print("mse,", mse)
        return -mse, features
    elif outtype == "survival":
        y_tune_new = np.array([(status, survival) for status, survival in y_tune], dtype=np.dtype([('Status','?'), ('Time','<f8')]))
        if device != 'cpu':
            prediction = estimator.predict(Z_tune.cpu().detach().numpy())
        else:
            prediction = estimator.predict(Z_tune.detach().numpy())
        result = concordance_index_censored(y_tune_new["Status"], y_tune_new["Time"], prediction)[0]
        return result, features
    elif outtype == "nothing":
        BIC = Z_tune.shape[1] * math.log(Z_tune.shape[0]) + 2 * Z_history[-1]
        #print("comb", comb, "last loss", Z_history[-1], "BIC", BIC)
        return -BIC, features
    
    

def train(X_train, y_train, X_tune=None, y_tune=None, comb_num=0, best_comb=None, important=None, outtype=None, top_rate=0.1, edged=None, vWeightd=None, epochs=1000, fold=5, plot=False, verbose=True, gpu=False, normalization=True, myseed=1234):
    torch_seed(seed=myseed)
    D = len(X_train)
    # choose to use GPU or CPU
    if gpu == True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if verbose:
            print("Training using GPU", flush=True)
    else:
        device = 'cpu'
        if verbose:
            print("Training using CPU", flush=True)
    # upper bound of the dimension of the latent code K
    if important is not None:
        feature_num = important
        if isinstance(feature_num, int):
            max_K = important
        else:
            max_K = min(feature_num)
    else:
        min_p = X_train[0].shape[1]
        for i in range(D):
            if X_train[i].shape[1] < min_p:
                min_p = X_train[i].shape[1]
        max_K = int(min_p * top_rate) 
    # 0211 force max_K to be at least 20
    if max_K < 20:
        print("Overriding impt or top_rate, set max_K = 20")
        max_K = 20

    # search the best hyper-parameter combination
    if best_comb == None:
        # default combination
        if comb_num == 0:
            if outtype == "categorical":
                print("Using default hyper-parameters with categorical outcome.")
                best_comb = [int(max_K / 2), 1e-4, 1, 10, 0.01]
            elif outtype == "continuous":
                print("Using default hyper-parameters with continuous outcome.")
                best_comb = [int(max_K / 2), 1e-4, 1, 1, 0.01]
            elif outtype == "survival":
                print("Using default hyper-parameters with survival outcome.")
                best_comb = [int(max_K / 2), 1e-4, 1, 1, 0.01]
            elif outtype == "nothing":
                print("Using default hyper-parameters with no outcome.")
                best_comb = [int(max_K / 2), 1e-4, 1, 1, 0.01]
            #else: raise an error
        else:
            best_idx = -1
            best_acc = float('-inf')
            
            # iDVL hyper-parameters
            K_list = range(0, max_K, int(max_K / 5))[1:]
            lr_list = [1e-1, 1e-2, 1e-3, 1e-4]
            lrz_list = [10, 1, 1e-1, 1e-2]
            l1_list = [0.001, 0.01, 0.1]
            
            if outtype == "categorical":
                print("Provided categorical outcome; using SVM Classifier:")
                # SVM hyper-parameters
                C_list = [0.1, 1, 10, 100]
                # Search Space
                max_num = len(K_list) * len(lr_list) * len(lrz_list) * len(C_list) * len(l1_list)
                comb = random.sample(sorted(set(itertools.product(K_list, lr_list, lrz_list, C_list, l1_list))), max_num)
            elif outtype == "continuous":
                print("Provided continuous outcome; using SVM Regression:")
                # SVM hyper-parameters
                C_list = [0.1, 1, 10, 100]
                # Search Space
                max_num = len(K_list) * len(lr_list) * len(lrz_list) * len(C_list) * len(l1_list)
                comb = random.sample(sorted(set(itertools.product(K_list, lr_list, lrz_list, C_list, l1_list))), max_num)
            elif outtype == "survival":
                print("Provided survival outcome; using SVM Survival:")
                # SVM hyper-parameters
                alpha_list = [1e-4, 1e-2, 0.1, 1, 10, 100]
                # Search Space
                max_num = len(K_list) * len(lr_list) * len(lrz_list) * len(alpha_list) * len(l1_list)
                comb = random.sample(sorted(set(itertools.product(K_list, lr_list, lrz_list, alpha_list, l1_list))), max_num)
            elif outtype == "nothing":
                print("Provided no outcome; using BIC:")
                # Search Space
                parm_list = [1]
                max_num = len(K_list) * len(lr_list) * len(lrz_list) * 1 * len(l1_list)
                comb = random.sample(sorted(set(itertools.product(K_list, lr_list, lrz_list, parm_list, l1_list))), max_num)
            # Loop over combinations
            for idx in range(min(comb_num, max_num)):
                # validating in the tuning dataset if provided
                #if X_tune is not None and y_tune is not None:
                if X_tune is not None:
                    if verbose == True and idx == 0:
                        print('Using validation dataset...', flush=True)
                    #print("comb:" ,comb[idx])
                    tmp_acc, _ = run(X_train, y_train, comb[idx], X_tune=X_tune, y_tune=y_tune, important=important, outtype = outtype,
                                     top_rate=top_rate, edged=edged, vWeightd=vWeightd,
                                     epochs=epochs, plot=plot, device=device, normalization=normalization)
                # k-fold cross validation if no tuning dataset
                else:
                    # Commented out 0217 no need y_tune
                    # if X_tune is not None and y_tune is None and idx == 0:
                    #     print('Outcome for tuning tuning set (y_tune) is not available. Sent to cross validation.')
                    if verbose == True and idx == 0:
                        print('Using {}-fold cross validation...'.format(fold), flush=True)

                    fold_acc = []
                    if outtype == "categorical":
                        tmp_indices = StratifiedKFold(n_splits=fold, shuffle=True, random_state=myseed).split(X_train[0], y_train)
                    elif outtype == "nothing":
                        tmp_indices = ShuffleSplit(n_splits=fold, test_size=1/fold, random_state=myseed).split(X_train[0])
                    else:
                        tmp_indices = ShuffleSplit(n_splits=fold, test_size=1/fold, random_state=myseed).split(X_train[0], y_train)
                    for train_index, tune_index in tmp_indices:
                        tmp_X_train = []
                        tmp_X_tune = []
                        if outtype == "nothing":
                            tmp_y_train = None
                            tmp_y_tune = None
                        else:
                            tmp_y_train = np.array(y_train)[train_index]
                            tmp_y_tune = np.array(y_train)[tune_index]
                        for d in range(len(X_train)):
                            tmp_X_train.append(X_train[d][train_index, :])
                            tmp_X_tune.append(X_train[d][tune_index, :])
                        
                        tmp_fold_acc, _ = run(X_train=tmp_X_train, y_train=tmp_y_train, comb=comb[idx], X_tune=tmp_X_tune, y_tune=tmp_y_tune,
                                              important=important, outtype=outtype, top_rate=top_rate, edged=edged, vWeightd=vWeightd,
                                              epochs=epochs, plot=plot, device=device, normalization=normalization)
                        fold_acc.append(tmp_fold_acc)
                    tmp_acc = np.mean(fold_acc)
                # update the best hyper-parameter combination
                if tmp_acc > best_acc:
                    best_idx = idx
                    best_acc = tmp_acc
            best_comb = comb[best_idx]

    if verbose:
        print('*****************************', flush=True)
        print('Best combination of hyper-parameters:', flush=True)
        print('K:{}, lr:{}, lrz:{}, Classifier:{}, l1:{}'.format(best_comb[0], best_comb[1], best_comb[2],
                                                 best_comb[3], best_comb[4]), flush=True)
        print('*****************************', flush=True)
        print()
    features, model_2, clf, Ztrain, GZ, Zprime, RZprime = run(X_train=X_train, y_train=y_train, comb=best_comb, important=important, outtype=outtype,
                                                              top_rate=top_rate, edged=edged, vWeightd=vWeightd, 
                                                              epochs=epochs, plot=plot, device=device, normalization=normalization)
    return features, model_2, clf, best_comb, Ztrain, GZ, Zprime, RZprime


def test(X_train, y_train, X_tune, comb, features, model_2, clf, y_tune=None, outtype = None, top_rate=0.1, edged=None, vWeightd=None, epochs=1000, plot=False, verbose=False, gpu=False, normalization=True, myseed=1234):
    torch_seed(seed=myseed)
    # choose to use GPU or CPU
    if gpu == True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if verbose:
            print("Testing using GPU", flush=True)
    else:
        device = 'cpu'
        if verbose:
            print("Testing using CPU", flush=True)

    # hyper-parameters
    K = comb[0]
    #lr_net = comb[1]
    lr_z = comb[2]
    #C = comb[3]
    #l1 = comb[4]

    D = len(X_train)
    n = X_train[0].shape[0]
    n_tune = X_tune[0].shape[0]

    X_train_normalized = []
    X_tune_normalized = []
    p = []
    for i in range(D):
        #X_train_normalized.append(torch.tensor(myNormalize(X_train[i])).to(device))
        #X_tune_normalized.append(torch.tensor(myNormalize(X_tune[i], X_train[i])).to(device))
        if normalization:
            X_train_normalized.append(torch.tensor(myNormalize(X_train[i])).to(device))
            X_tune_normalized.append(torch.tensor(myNormalize(X_tune[i], X_train[i])).to(device))
        else:
            X_train_normalized.append(torch.tensor(X_train[i]).to(device))
            X_tune_normalized.append(torch.tensor(X_tune[i]).to(device))
        p.append(X_train[i].shape[1])

    if edged != None:
        lap = normalized_Laplacian(X_train_normalized, edged, vWeightd)

    def norm_21(x):
        return torch.norm(x, dim=0).sum()

    def unsupervised_loss_1(X, Z_pred, model):
        loss = 0
        if edged != None:
            for i in range(D):
                loss += norm_21(X[i] - model[i](Z_pred)) + l1 * norm_21(model[i](Z_pred) @ lap[i].to(device))
        else:
            for i in range(D):
                loss += norm_21(X[i] - model[i](Z_pred)) + l1 * norm_21(model[i](Z_pred))
        return loss / X[0].shape[0]

    def unsupervised_loss_2(X, Z_pred, model):
        loss = 0
        for i in range(D):
            loss += torch.norm(X[i] - model[i](Z_pred)) ** 2
        return loss / X[0].shape[0]

    X_tune_fs = []
    for d in range(D):
        X_tune_fs.append(X_tune_normalized[d][:, features[d]])

    # get Z from testing data
    Z_tune = torch.randn(n_tune, K, requires_grad=True, device=device)
    optimizer_z_tune = torch.optim.Adam([Z_tune], lr_z)
    Z_history = []

    for i in range(epochs):
        loss_z = unsupervised_loss_2(X_tune_fs, Z_tune, model_2)
        Z_history.append(loss_z.item())
        optimizer_z_tune.zero_grad()
        loss_z.backward()
        optimizer_z_tune.step()
        # projected gradient descent (PGD)
        with torch.no_grad():
            tmp_norm = torch.norm(Z_tune, dim=1).reshape(-1, 1)
            tmp_norm = torch.clamp(tmp_norm, min=1)
            Z_tune.data = Z_tune.data / tmp_norm
    # if plot:
    #     plt.plot(Z_history)
    #     plt.title("Z History")
    #     plt.show()
    #     plt.close()
    # Get Z' and RZ' for testing
    RZprimetest = []
    if device != 'cpu':
        Zprimetest = Z_tune.cpu().detach().numpy()
        for d in range(D):
            RZprimetest.append(model_2[d](Z_tune).cpu().detach().numpy())
    else: 
        Zprimetest = Z_tune.detach().numpy()
        for d in range(D):
            RZprimetest.append(model_2[d](Z_tune).detach().numpy())
    
    prediction = None
    performance = None
    if X_tune is not None and clf is not None:
        # print("0217 check can BIC be here?")
        if outtype == "nothing":
            # clf should be second stage training loss, last element
            performance = Z_tune.shape[1] * math.log(Z_tune.shape[0]) + 2 * clf
            print("Testing on data with no outcome, returning BIC:", performance)
        else: 
            if gpu == True:
                prediction = clf.predict(Z_tune.cpu().detach().numpy())
            else:
                prediction = clf.predict(Z_tune.detach().numpy())
            #print("prediction", prediction)
            if y_tune is not None:
                if outtype == "categorical":
                    performance = (prediction.reshape(-1) == y_tune).sum().item() / len(y_tune)
                    print("Testing on categorical data, returning classification accuracy:", performance)
                elif outtype == "continuous":
                    performance = mean_squared_error(y_tune, prediction)
                    print("Testing on continuous data, returning mean squared error:", performance)
                elif outtype == "survival":
                    #y_train_new = np.array([(status, survival) for status, survival in y_train], dtype=np.dtype([('Status','?'), ('Time','<f8')]))
                    y_tune_new = np.array([(status, survival) for status, survival in y_tune], dtype=np.dtype([('Status','?'), ('Time','<f8')]))
                    performance = concordance_index_censored(y_tune_new["Status"], y_tune_new["Time"], prediction)[0]
                    print("Testing on survival data, returning C-statistics:", performance)


    return prediction, performance, Zprimetest, RZprimetest
