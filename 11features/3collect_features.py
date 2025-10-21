import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
##=============================================================================

project = sys.argv[1]
task_name = sys.argv[2]


path2meta = "../10metadata/"

data_augmentation = False
model_name = "ctrans"

path2inputs = f"{project}_features_{model_name}/"

metadata = pd.read_csv(f"{path2meta}{project}_slide_selected.csv")
##=============================================================================

slide_files = metadata.slide_file.values

n_slides = len(slide_files)

print(f"n_slides: {n_slides}")

##--------
features = []
for i_slide in range(n_slides):
    slide_file = slide_files[i_slide]
    x = np.load(f"{path2inputs}{slide_file}.npy")

    print(i_slide, x.shape)

    features.append((slide_file, x))

print("-- completed collect --")


np.save(f"{project}_features.npy", np.array(features, dtype=object), allow_pickle=True)
print(f"len(features): {len(features)}")

##----------
if data_augmentation:
    ## add features_rot into the features
    for k in [1,2,3]:
        for i_slide in range(n_slides):
            slide_file = slide_files[i_slide]
            x = np.load(f"{path2inputs}{slide_file}_rot{k}.npy")
            features.append((slide_file, x))
        
    np.save(f"{path2outputs}{project}_features_rot.npy", features)

    print(f"len(features_rot): {len(features)}")
    print("--- completed ---  collect features from every slide")

##=============================================================================
## n_tiles in each slide
print("number of tiles in each slide:")

slide_files = np.array([features[i][0] for i in range(n_slides)])
n_tiles = np.array([len(features[i][1]) for i in range(n_slides)])
print("min, max, n>8000:", np.min(n_tiles), np.max(n_tiles), sum(n_tiles>8000))

np.savetxt(f"{project}_n_tiles.txt", np.array((slide_files, n_tiles)).T, fmt="%s %s")

## plot
bins = np.linspace(0,16000,40, endpoint=True)
nx,ny = 1,1
fig, ax = plt.subplots(ny,nx,figsize=(nx*3.5,ny*3))
ax.hist(n_tiles,bins=bins,histtype='bar',color="lightblue",edgecolor="black",rwidth=0.85)
ax.set_xlabel("Number of tiles per slide")
ax.set_ylabel("Number of slides")
        
plt.tight_layout(h_pad=1, w_pad= 1.5)
plt.savefig(f"{project}_n_tiles.pdf", format="pdf", dpi=50)

print("--- completed ---  n_tiles")

##=============================================================================
