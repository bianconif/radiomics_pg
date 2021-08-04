from utilities.misc import Roi

from features.shape.mask.aabb import volume_density
from features.shape.mask.others import compactness_1, surface_to_volume_ratio

patient_id = '115GA'
    
#Read the Roi
roi = Roi.from_pickle(source = f'tests/roi/{patient_id}.pkl')

features = {'surface_to_volume_ratio' : surface_to_volume_ratio,
            'compactness_1' : compactness_1,
            'volume_density' : volume_density}

for key, value in features.items():
    print(f'{key} = {value(roi):.3f}')
    a = 0
