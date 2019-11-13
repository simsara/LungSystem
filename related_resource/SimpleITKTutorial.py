# world coordinates x,y,z 有现实世界的实际涵义
# voxel coordinates i,j,k 可能没有现实世界涵义
# This Tutorial will show how to: 
# - Open and read a .mhd image 
# - Visualize a .mhd image 
# - Read a list of candidates from a .csv file 
# - Transform from world coordinates to voxel coordinates 
# - Extract some features / patches of candidates and visualize them

import SimpleITK as sitk 
import numpy as np 
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt

# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('192.168.1.112', 40068), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage,numpyOrigin,numpySpacing

def readCSV(filename):
    lines = []
    with open(filename,"r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord,origin,spacing):
    # 一些解释 http://www.freesion.com/article/563621885/
    # transform from world coordinates to voxel coordinates
    stretchedVoxelCoord = np.absolute(worldCoord -origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def normalizePlanes(npzarray):
    # extract views from the candidates

    maxHU = 400.
    minHU = -1000. 

    npzarray = (npzarray - minHU)/(maxHU - minHU)
    npzarray[npzarray>1] = 1. # 超过阈值的全归一化为1
    npzarray[npzarray<0] = 0. # 不超过阈值的全归一化为0
    return npzarray

img_path = "/home/liubo/data/LUNA16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.mhd"
cand_path = "/home/liubo/data/LUNA16/CSVFILES/annotations.csv"

# load image
numpyImage,numpyOrigin,numpySpacing = load_itk_image(img_path)
# print(numpyImage.shape)
# print(numpyOrigin)
# print(numpySpacing)
"""
输出结果
(140, 512, 512)
[-237.240005 -143.600006 -171.      ]
[2.         0.66406202 0.66406202]
"""

# load candidates
cands = readCSV(cand_path)
# print(cands)
# get candiates
for idx in range(1,len(cands)):
    # TODO 这里还有问题
    cand = cands[idx]
    print("processing %d"%idx)
    worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    voxelCoord = worldToVoxelCoord(worldCoord,numpyOrigin,numpySpacing)
    voxelWidth = 65
    patch = numpyImage[int(voxelCoord[0]),int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2),int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]
    patch = normalizePlanes(patch)
    print("data")
    print(worldCoord)
    print(voxelCoord)
    print(patch)
    outputDir = "patches/"
    plt.imshow(patch, cmap="gray")
    im = Image.fromarray((patch*255).astype('uint8')).convert("RGB")
    img_path = os.path.join(outputDir, 'patch_' + "_".join([str(coord) for coord in worldCoord]) + ".png")
    print(img_path)
    im.save(img_path)
