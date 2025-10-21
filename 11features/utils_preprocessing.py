import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
from PIL import Image
import czifile

import cv2
import torch
from torch import nn
import torchvision
#from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTModel
from timm.models.vision_transformer import VisionTransformer
import timm
from ctrans_model import CTransPath

import utils_color_norm
color_norm = utils_color_norm.macenko_normalizer()

## check available device
#device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device = "cpu"
print("device:", device)
##======================================================================================================
class resnet50_feature_extraction(nn.Module):
    def __init__(self, model_type="load_from_saved_file"):
        super().__init__()

        if model_type == "load_from_internet":
            self.resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_type == "load_from_saved_file":
            self.resnet = resnet50(weights=None)
        else:
            print("cannot find model_type can only be load_from_internet or load_from_saved_file")

        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

##======================================================================================================
def evaluate_tile_edge(img_np, edge_mag_thrsh, edge_fraction_thrsh):

    #select = 1  ## initial value
    
    #img_np = np.array(img_RGB)
    tile_size = img_np.shape[0]
        
    ##---------------------------------------
    ## 0) exclude if edge_mag > 0.5
    img_gray=cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Remove noise using a Gaussian filter
    #img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)

    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    edge_frac = (mag < edge_mag_thrsh).mean()

    return (edge_frac <= edge_fraction_thrsh)

##======================================================================================================
def evaluate_tile_color(img_np,black_thrsh,black_pct_thrsh,blue_level_thrsh,red_level_thrsh,
                        H_min,H_max,S_min,S_max,V_min,V_max,select):

    #img_np = np.array(img_RGB)

    L, A, B = cv2.split(cv2.cvtColor((img_np), cv2.COLOR_RGB2LAB)) 

    ##---------------------------------------
    ## 1) remove if percentage of black spot > 0.01
    black_pct = np.mean(L < black_thrsh)
    if black_pct > black_pct_thrsh:
        select = 0
        return select
    ##---------------------------------------
    ## 2) remove if too blue (heavy mark), or too red (blood)
    red,green,blue = np.mean(img_np[:,:,0]),np.mean(img_np[:,:,1]),np.mean(img_np[:,:,2])
    blue_level = blue/(red + green)
    blue_level2 = blue*blue_level

    if blue_level2 > blue_level_thrsh:
        select = 0
        return select

    ##---
    red_level = red/(green + blue)
    red_level2 = red*red_level

    if red_level2 > red_level_thrsh:
        select = 0
        return select

    ##---------------------------------------
    ## 3) remove if tile has the same color suggested (using color detection)
    H,S,V = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))    
    H,S,V = np.mean(H),np.mean(S),np.mean(V)

    if (H_min <= H and H <= H_max and S_min <= S and S <= S_max and V_min <= V and V <= V_max):
        select = 0
        return select
    
    return select

##================================================================================================
## 2024.05.02: for NZI files
def slide2tiles(path2slide, slide_name, slide_file, mag_assumed, mag_selected, tile_size, 
    mask_downsampling,edge_mag_thrsh,edge_fraction_thrsh,save_tile_file,
    path2mask,path2coordinates):
    
    if slide_file.split(".")[-1] == "czi":
        slide_format = "czi"
    else:
        slide_format = "others"
    
    ##-------------------------
    ##-------------------------
    if slide_format == "czi":
        ## read CZI image file:
        slide = czifile.imread(f"{path2slide}{slide_file}")

        if np.ndim(slide) == 6:
            slide = slide[0][0][0]

        elif np.ndim(slide) == 5:
            slide = slide[0][0]
        print("slide.shape:", slide.shape)

        mag_max = mag_assumed
        mag_original = 0
    ##-------------------------
    else:
        ## open slide
        slide = openslide.OpenSlide(f"{path2slide}{slide_file}")

        ## magnification max
        if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
            mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            print("mag_max:", mag_max)
            mag_original = mag_max
        else:
            print("[WARNING] mag not found, assuming: {mag_assumed}")
            mag_max = mag_assumed
            mag_original = 0
    ##-------------------------
        

    ## downsample_level
    downsampling = int(int(mag_max)/mag_selected)
    print(f"downsampling: {downsampling}")

    mask_tile_size = int(np.ceil(tile_size/mask_downsampling))
    #print("mask_tile_size:", mask_tile_size)

    ##------------------------------------------------------------------
    ## slide partitioning
    ## slide size at largest level (level=0)
    if slide_format == "czi":
        py0, px0 = slide.shape[0], slide.shape[1]
    else:
        px0, py0 = slide.level_dimensions[0]

    tile_size0 = int(tile_size*downsampling)
    print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

    n_rows,n_cols = int(py0/tile_size0), int(px0/tile_size0)
    print(f"n_rows: {n_rows}, n_cols: {n_cols}")

    n_tiles_total = n_rows*n_cols
    print(f"n_tiles_total: {n_tiles_total}")

    ##-----------------------
    img_mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)
    mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)

    i_tile = 0
    tiles_list = []

    col_list = []
    row_list = []
    i_tile_list = []
    for row in range(n_rows):
        print(f"row: {row}/{n_rows}")
        for col in range(n_cols):

            if slide_format == "czi":
                ## 2024.05.02: CZI file
                tile = slide[int(row*tile_size0):int((row+1)*tile_size0),\
                             int(col*tile_size0):int((col+1)*tile_size0),:]
                tile = Image.fromarray(tile, "RGB")
            else:
                tile = slide.read_region((col*tile_size0, row*tile_size0),\
                                     level=0, size=[tile_size0, tile_size0]).convert("RGB") ## RGBA image --> RGB
                

            if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:
                # downsample to target tile size
                tile = tile.resize((tile_size, tile_size))

                mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))

                img_mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                         int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile

                tile = np.array(tile)
                #print(tile.shape)

                ## evaluate tile
                select = evaluate_tile_edge(tile, edge_mag_thrsh, edge_fraction_thrsh)

                if select == 1:
                    ## 2022.09.08: color normalization:
                    tile_norm = Image.fromarray(color_norm.transform(tile))

                    mask_tile_norm = np.array(tile_norm.resize((mask_tile_size, mask_tile_size)))

                    mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                         int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile_norm    

                    #tiles_list.append(np.array(tile_norm).astype(np.uint8))
                    tiles_list.append(tile_norm)

                    if save_tile_file:
                        tile_name = "tile_" + str(row).zfill(5)+"_" + str(col).zfill(5) + "_" \
                                 + str(i_tile).zfill(5) + "_" + str(downsampling).zfill(3)

                        tile_norm.save(f"{tile_folder}/{tile_name}.png")

                    ## 2023.05.27: tile information
                    col_list.append(col)
                    row_list.append(row)
                    i_tile_list.append(i_tile)

            i_tile += 1

    ## 2023.05.27: save tile coordinates:
    downsampling_list = [downsampling]*len(row_list)
    df_coordinates = pd.DataFrame({"row": row_list, "col": col_list, "i_tile": i_tile_list, "downsampling": downsampling})
    #df_coordinates.to_csv(f"{path2coordinates}{slide_name}.csv", index_label="tile_idx")
    df_coordinates.to_csv(f"{path2coordinates}{slide_file}.csv", index_label="tile_idx")
    
    ##====================================================================================================== 
    ## plot: draw color lines on the mask
    line_color = [0,255,0]

    n_tiles = len(tiles_list)

    #img_mask[:,::mask_tile_size,:] = line_color
    #img_mask[::mask_tile_size,:,:] = line_color
    mask[:,::mask_tile_size,:] = line_color
    mask[::mask_tile_size,:,:] = line_color

    fig, ax = plt.subplots(1,2,figsize=(30,15))
    #fig, ax = plt.subplots(1,2,figsize=(50,30))
    ax[0].imshow(img_mask)
    ax[1].imshow(mask)

    ax[0].set_title(f"{slide_name}, mag_original: {mag_original}, mag_assumed: {mag_assumed}")
    ax[1].set_title(f"edge: {edge_mag_thrsh}, n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}")

    plt.tight_layout(h_pad=0.4, w_pad=0.5)
    #plt.savefig(f"{path2mask}{slide_name}.pdf", format="pdf", dpi=50)
    plt.savefig(f"{path2mask}{slide_file}.pdf", format="pdf", dpi=50)
    plt.close()

    img_mask = 0 ; mask = 0

    print("completed cleaning")
    
    return tiles_list

##======================================================================================================
def tile_transform(tiles_list, data_mean, data_std):
    data_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=data_mean, std=data_std)])

    ## data transform:
    n_tiles = len(tiles_list)
    print("n_tiles:", n_tiles)

    tiles = []
    for i in range(n_tiles):
        tiles.append(data_transform(tiles_list[i]).unsqueeze(0))
    tiles = torch.cat(tiles, dim=0)
    print("tiles.shape:", tiles.shape)
    tiles_list = 0
    
    return tiles  ## [n_tiles,3,224,224]

##================================================================================================
def tiles2features(tiles_list, model_name, batch_size):

    ##----------------------------------------
    ## model config
    if model_name == "vit":
        path2model = "../vit-base-patch16-224-in21k"
        model = ViTModel.from_pretrained(path2model)
        model.to(device)
        data_mean=[0.5, 0.5, 0.5] ; data_std = [0.5, 0.5, 0.5]

    if model_name == "dino":
        path2model = "../dino_vit_small_patch16_ep200.pt"
        model = VisionTransformer(img_size=224, patch_size=16, 
                                  embed_dim=384, num_heads=6, num_classes=0)
        model.to(device)
        model.load_state_dict(torch.load(path2model,map_location=device))
        data_mean=[0.485, 0.456, 0.406] ; data_std = [0.229, 0.224, 0.225]

    if model_name == "ctrans":
        path2model = "ctranspath.pth"
        model = CTransPath(num_classes=0)
        model.to(device)
        model.load_state_dict(torch.load(path2model)['model'])
        data_mean=[0.485, 0.456, 0.406] ; data_std = [0.229, 0.224, 0.225]

    if model_name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        path2model = "pytorch_model_conch.bin"

        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=f"{path2model}")

        model.to(device)
        model_tile_size = 512
        data_mean = [0.48145466, 0.4578275, 0.40821073] ; data_std = [0.26862954, 0.26130258, 0.27577711]

    ##-----------------------------------
    if model_name == "uni":
        print(f"model_name: {model_name}")

        path2model = "pytorch_model_uni.bin"

        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, 
        num_classes=0, dynamic_img_size=True)

        model.to(device)
        model.load_state_dict(torch.load(path2model,map_location=device))

        model_tile_size = 224
        data_mean=[0.485, 0.456, 0.406] ; data_std = [0.229, 0.224, 0.225]

    ##-----------------------------------
    if model_name == "virchow2":
        print(f"model_name: {model_name}")
        path2model = f"vit_huge_patch14_224_virchow_v2.bin"
            
        model = timm.create_model(
            "vit_huge_patch14_224",
            checkpoint_path=path2model,
            img_size=224,
            patch_size=14,
            init_values=1e-5,
            num_classes=0,
            mlp_ratio=5.3375,
            global_pool="",
            reg_tokens=4,
            dynamic_img_size=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

        model.to(device)
        model.load_state_dict(torch.load(path2model,map_location=device))

        model_tile_size = 224
        data_mean=[0.485, 0.456, 0.406] ; data_std = [0.229, 0.224, 0.225]



    model.eval()
    
    ## tile transform
    tiles = tile_transform(tiles_list, data_mean, data_std)

    ## extract features from tiles
    n_tiles = tiles.shape[0]
    features = []
    for idx_start in range(0, n_tiles, batch_size):
        idx_end = idx_start + min(batch_size, n_tiles - idx_start)

        #with torch.no_grad():
        with torch.inference_mode():
            if model_name == "conch":    
                y = model.encode_image(tiles[idx_start:idx_end], proj_contrast=False, normalize=False)
            else:
                y = model(tiles[idx_start:idx_end])

        if model_name == "virchow2":     
            y = torch.cat([y[:, 0], y[:, 5:].mean(1)], dim=-1)

        #if model_name == "vit":
        #    y = y.last_hidden_state[:, 0]

        features.append(y.detach().cpu().numpy())

    features = np.concatenate(features)
    print("features.shape:", features.shape)
    
    return features
##================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
