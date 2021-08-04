from utilities.misc import Roi

from features.shape.mask.aabb import volume_density, length_breadth_thickness,\
     zingg_ratios, disc_rod_index, oblate_prolated_index
from features.shape.mask.others import compactness_1, surface_to_volume_ratio

patient_id = '115GA'
    
#Read the Roi
roi = Roi.from_pickle(source = f'tests/roi/{patient_id}.pkl')

features = {'surface_to_volume_ratio' : surface_to_volume_ratio,
            'compactness_1' : compactness_1,
            'length_breatdh_thickness' : length_breadth_thickness, 
            'zingg_ratios' : zingg_ratios,
            'disc_rod_index' : disc_rod_index,
            'oblate-prolated-index' : oblate_prolated_index,
            'volume_density' : volume_density}

for key, value in features.items():
    retval = value(roi)
    print(f'{key} = ', end = '')
    try:
        for element in retval:
            print(f' {element:.3f}', end = '')
        print('\n', end = '')
    except TypeError:
        print(f' {retval:.3f}')
