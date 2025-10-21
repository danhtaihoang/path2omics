import os,sys
import numpy as np
import pandas as pd
##from PyPDF2 import PdfFileMerger, PdfFileReader
from PyPDF2 import PdfMerger, PdfReader
##======================================================================================================================

project = "BRCA_FFPE"

path2meta = "../10metadata/"
path2inputs = f"{project}_mask/"
path2outputs = ""

## find nume files within a folder
slide_files = []
for f in os.listdir(path2inputs):
    if f.endswith(".pdf"):
        slide_files.append(f)

## alphabet sort
slide_files = sorted(slide_files)

##mergedObject = PdfFileMerger()
mergedObject = PdfMerger()
for slide_file in slide_files:    
    mergedObject.append(PdfReader(f"{path2inputs}{slide_file}", "rb"))
    
mergedObject.write(f"{project}_mask.pdf")

print("--- completed collecting mask--- ")
##-----------------------------------
df = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")
slide_files_all = df["slide_file"].values

slide_files_short = np.array([x[:-4] for x in slide_files]) ## without .pdf

slide_files_missing = np.setdiff1d(slide_files_all, slide_files_short)

print("slide_files_missing:", slide_files_missing)
print(slide_files_missing.shape)

if len(slide_files_missing) > 0:
    slide_idxs = np.array([np.argwhere(slide_files_all == x)[0][0] for x in slide_files_missing])
    print("slide missing idx:", slide_idxs)

    np.savetxt(f"{project}_missing_slides.txt", np.array((slide_idxs,slide_files_missing)).T, fmt="%s")

print("--- completed all taks --- ")











