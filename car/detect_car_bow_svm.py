import cv2
import numpy as np
import os

if not os.path.isdir('CarData'):
    print(
        'CarData folder not found. Please download and unzip'
        'http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz'
        'into the same folder as this script.'
    )
    exit(1)

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100

sift = cv2.xfeatures2d.SIFT