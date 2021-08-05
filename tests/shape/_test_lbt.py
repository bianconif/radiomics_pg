import os

from utilities.misc import Roi
from features.shape.lbt import lbt_index

lbt_indices_to_compute = ['cailleux-flatness', 'disc-rod']
modes = ['aabb', 'mask', 'signal']

source_folder = '../../zingg_shape_radiomics/cache/SPN-01'
source_files = os.listdir(source_folder)


for source_file in source_files:
    
    #Read the Roi
    roi = Roi.from_pickle(source = f'{source_folder}/{source_file}')
    
    for index in lbt_indices_to_compute:
        print(f'[{source_file}] {index} =', end = '')
        for mode in modes:
            value = lbt_index(roi, index, mode)
            print(f' {value:.1f} ({mode})', end = '')
        print('\r', end = '')
    
    a = 0

#features = {'surface_to_volume_ratio' : surface_to_volume_ratio,
            #'compactness_1' : compactness_1,
            #'length_breatdh_thickness' : length_breadth_thickness, 
            #'zingg_ratios' : zingg_ratios,
            #'disc_rod_index' : disc_rod_index,
            #'oblate-prolated-index' : oblate_prolated_index,
            #'volume_density' : volume_density}