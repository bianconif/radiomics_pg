import os

import SimpleITK as sitk
import six

from radiomics import firstorder, imageoperations

from functions import *

#Set up a pyradiomics directory variable:
dataDir = '../output'

#Store the path of your image and mask in two variables
image = sitk.ReadImage('../input/test/001/CT/image.nrrd')
mask = sitk.ReadImage('../input/test/001/CT/mask.nii')
window = (125 - 150, 125 + 150)
windowedMask = imageoperations.resegmentMask(image, mask,
                                             resegmentRange = window)

#Test compute_features()
features_class = 'firstorder'
features_names = ['Mean', 'Maximum', 'Energy']
results = compute_features(image, mask, features_class, features_names)

#Calculate the first-order features
firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, windowedMask)
firstOrderFeatures.enableAllFeatures()
firstOrderFeatures.calculateFeatures()

#Print the features
for key, val in six.iteritems(firstOrderFeatures.featureValues):
    print("\t%s: %s" %(key, val))

