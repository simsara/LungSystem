
"""
step1 is processing luna16
"""
import numpy
import pydicom
import os
import cv2
import glob
import pandas
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi

import configs.graduate.config_training_gradute as config 

