import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset,Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.stats.multitest as smt
import pickle
##===================================================================================================
#### Build dataset
class slide_target_dataset(Dataset):
    ## input: features_list[n_slides](slide_name, features[n_tiles,n_features])
    ## target[n_slides, n_target]
    
    def __init__(self, features, targets):
        
        self.features = features
        self.targets = targets    
        self.dim = self.features[0][1].shape[1] ## n_features

    def __getitem__(self, index):
        sample = torch.Tensor(self.features[index][1]).float()
        target = torch.Tensor(self.targets[index]).float()
        
        return sample, target

    def __len__(self):
        return len(self.features)

##===================================================================================================
def load_dataset(path2features, path2target, path2split, ik_fold, il_fold, target_cols):
    
    ## load image feature
    if path2features.endswith(".npy"):
        features = np.load(path2features, allow_pickle=True)
    else:
        with open(path2features, 'rb') as f:
            features = pickle.load(f)

    print("len(features):", len(features))
    
    ## load target
    if path2target.endswith(".csv"):
        df_target = pd.read_csv(path2target, index_col=None, usecols=target_cols)[target_cols]
    else:
        df_target = pd.read_pickle(path2target)

    targets = df_target[target_cols].values
    print("targets.shape:", targets.shape)
    
    ## create dataset
    dataset = slide_target_dataset(features, targets)
    
    ## load_train_valid_test_idx:
    train_valid_test_idx = np.load(path2split, allow_pickle=True)

    train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    test_idx = train_valid_test_idx["test_idx"][ik_fold]
    
    ## split train, valid, test dataset
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)
    
    return train_set, valid_set, test_set

##===================================================================================================
def compute_coefs(labels, preds):
    return np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])

def compute_slope(labels, preds):
    return np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])

def compute_coef_slope(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    
    return coef, slope

##------------------------------------------------------------------
## R and p_1side values
def pearson_r_and_p(label, pred):
    R,p = pearsonr(label, pred)
    if R> 0:
        p_1side = p/2.
    else:
        p_1side = 1-p/2

    return p_1side

##------------------------------------------------------------------
## number of genes with Holm-Sidak correlated p-val<0.05
def number_predictable_genes(labels, preds):
    p = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    return np.sum(p<0.05)

def holm_sidak_p(labels, preds):
    return np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

def compute_coef_slope_p(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    return coef, slope, p_value

def compute_coef_slope_padj(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    p_adj = smt.multipletests(p_value, alpha=0.05, method='hs', is_sorted=False, returnsorted=False)[1]

    return coef, slope, p_adj

##===================================================================================================
def norm_percentiles(y_train, y_test):
    ## mean and std over samples for each gene
    mean = np.mean(y_train,axis=0)
    std = np.std(y_train,axis=0)
    
    ## y_test[n_samples, n_genes]
    perc_test = np.zeros_like(y_test).astype(float)
    for i in range(y_test.shape[1]):
        ## for each gene
        perc_test[:,i] = norm.cdf(y_test[:,i], mean[i], std[i])
    
    return perc_test
##===================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##===================================================================================================


