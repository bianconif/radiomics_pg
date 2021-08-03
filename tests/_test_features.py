from utilities.misc import Roi
import features

patient_id = '115GA'
    
#Read the Roi
roi = Roi.from_pickle(source = f'tests/roi/{patient_id}.pkl')

#Instantiate the shape feature calculator
shape = features.Shape(roi)

features = {'surface_to_volume_ratio' : shape.surface_to_volume_ratio,
            'compactness_1' : shape.compactness_1}

for key, value in features.items():
    print(f'{key} = {value():.3f}')
    a = 0
