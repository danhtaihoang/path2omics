# %%
import numpy as np
import os,sys,time,platform

#from torchvision.models import resnet50
from utils_preprocessing import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

init_random_seed(random_seed=42)

# %%
##====================================================================================================== 
project = sys.argv[1]
i_slide = int(sys.argv[2])
edge_mag_thrsh = float(sys.argv[3])

model_names = ["ctrans"]
print(f"project: {project}, i_slide: {i_slide}")
##---------------------------------------
## hyper-parameters
mag_assumed = 20
evaluate_edge = True
save_tile_file = False
extract_pretrained_features = True

mag_selected = 20
tile_size = 512
mask_downsampling = 16

model_tile_size = 224
batch_size = 32

## evaluate tile
edge_fraction_thrsh = 0.5

##---------------------------------------
path2storage = "../"    

# %%
##====================================================================================================== 
path2slide = path2storage + f"{project}_slides_data/"
print("path2slide:", path2slide)
    
path2meta = "../10metadata/"
path2mask = f"{project}_mask/"
path2coordinates = f"{project}_coordinates/"
path2report = f"{project}_report/"

metadata = pd.read_csv(f"{path2meta}{project}_slide_selected.csv")

##--------------------------
os.makedirs(path2mask, exist_ok=True)
os.makedirs(path2coordinates,exist_ok=True)
os.makedirs(path2report, exist_ok=True)

# %%
##======================================================================================================
slide_files = metadata.slide_file.values
slide_names = metadata.slide_name.values

slide_file = slide_files[i_slide]
slide_name = slide_names[i_slide]
print(f"slide_file: {slide_file}, slide_name: {slide_name}")

if save_tile_file:
    ## create tile_folder:
    tile_folder = f"{project}_tiles/" + slide_name
    print(f"tile_folder: {tile_folder}")

    os.makedirs(tile_folder,exist_ok=True)

# %%
tiles_list = slide2tiles(path2slide, slide_name, slide_file, mag_assumed, mag_selected, tile_size, 
                         mask_downsampling,edge_mag_thrsh,edge_fraction_thrsh,save_tile_file,
                        path2mask,path2coordinates)
# %%
report_df = pd.DataFrame({"slide_file": [slide_file], "slide_name": [slide_name], 
                          "edge_mag_thrsh": [edge_mag_thrsh]})
print(report_df)
report_df.to_csv(f"{path2report}{slide_file}.csv", index=None)

# %%
##======================================================================================================
# Extract features from tiles
if extract_pretrained_features:
    for model_name in model_names:
        print(" ")
        print("model_name:", model_name)

        if extract_pretrained_features:
            path2features = f"{project}_features_{model_name}/"
            os.makedirs(path2features,exist_ok=True)

        start_time = time.time()
        ## feature extraction
        features = tiles2features(tiles_list, model_name, batch_size)

        run_time = int(time.time() - start_time)
        print(f"finished -- i_slide: {i_slide}, total time: {run_time}")

        np.save(f"{path2features}{slide_file}.npy", features)

# %%

# %%
