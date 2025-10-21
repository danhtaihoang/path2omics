# %%
import numpy as np
import pandas as pd
import torch
#from torch import optim
from torch.utils.data import DataLoader,ConcatDataset
import os,sys,time
from model_MLP import *
from utils import *
import pickle

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

init_random_seed(random_seed=42)

# %%
##================================================================================================
project = "BRCA"

##-------------------------------------------------
n_inputs = 768
n_hiddens = 512
dropout = 0.2
batch_size = 32
learning_rate = 0.0001
print("dropout:", dropout)
print("batch_size:", batch_size)
print("learning_rate:", learning_rate)

ik_fold = int(sys.argv[1])
il_fold = int(sys.argv[2])
i_gene_min = int(sys.argv[3])
i_gene_step = int(sys.argv[4])
max_epochs,patience = 500,50

print("ik_fold:", ik_fold)
print("il_fold:", il_fold)
print("i_gene_min:", i_gene_min)
print("i_gene_step:", i_gene_step)
print("max_epochs: {}, patience: {}".format(max_epochs, patience))

# %%
##================================================================================================
path2split = f"../10metadata/{project}_train_valid_test_idx.npz"
path2features = f"../11features/{project}_features.npy"
path2target = f"../12target/{project}_actual.pkl"
genes = np.loadtxt(f"../12target/{project}_genes.txt", dtype="str")

i_gene_max = int(i_gene_min + i_gene_step)
genes = genes[i_gene_min:i_gene_max][:,0]
print("genes:", genes)

n_outputs = len(genes)
print("n_outputs:", n_outputs)

## create result directory
result_dir = f"results/result_{ik_fold}_{il_fold}_{i_gene_min}/"
os.makedirs(result_dir,exist_ok=True)

# %%
## load data
train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split,
                                              ik_fold, il_fold, genes)

# %%
##================================================================================================
## model
bias_init = torch.nn.Parameter(torch.Tensor(np.mean([sample[1].detach().cpu().numpy()\
                         for sample in train_set], axis=0)).to(device))

model = MLP_regression(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
##================================================================================================
print(" ")
print(" --- fit --- ")
start_time = time.time()

model,train_loss,train_coef,train_slope,\
valid_loss,valid_coef,valid_slope,valid_labels,valid_preds = \
fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size)

print("fit -- completed -- time: {:.2f}".format(time.time() - start_time))

# %%
##================================================================================================
#print(" ")
#print(" --- analyze_result --- ")
start_time = time.time()

analyze_result(result_dir,genes,model,train_loss,train_coef,train_slope, \
               valid_loss,valid_coef,valid_slope,valid_labels, valid_preds, test_set)

print(f"analyze_result -- completed -- time: {(time.time() - start_time):.2f}s")
##================================================================================================

print("--- completed ---")

# %%
