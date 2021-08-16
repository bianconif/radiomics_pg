import json
import os

import matplotlib.pyplot as plt

from radiomics_pg.utilities.misc import Roi
from radiomics_pg.utilities.geometry import TriangularMesh

patient_ids = ['31MM', '71VG', '115GA']
base_folder = '../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01'

for patient_id in patient_ids:    
    mask_file = f'{base_folder}/{patient_id}/nifti-scans-and-rois/CT/mask.nii'
    scan_folder = f'{base_folder}/{patient_id}/dicom-scans/CT'
    metadata_source = f'{base_folder}/{patient_id}/metadata.json'
    
    #Read the metadata
    diagnosis = {'Malignancy' : False}
    with open(metadata_source) as f:
        data = json.load(f)
        if data['Malignancy'] == 'P':
            diagnosis['Malignancy'] = True
        diagnosis['Diagnosis'] = data['Diagnosis']
    

    roi = Roi.from_dcm_and_nii(mask_file, scan_folder, diagnosis = diagnosis)
    roi.save(f'tests/roi/{patient_id}.pkl')
    roi_1 = Roi.from_pickle(source = f'tests/roi/{patient_id}.pkl')
    
    #Dump the roi as bitmap
    bitmap_dump_folder = f'tests/roi/bitmap_dump/{patient_id}'
    if not os.path.isdir(bitmap_dump_folder):
        os.makedirs(bitmap_dump_folder)
    roi_1.dump_to_bitmaps(bitmap_dump_folder)

    #Show the triangular mesh
    fig = plt.figure(figsize=(10, 10))
    roi_1.draw_mesh(fig)
    #tm = roi_1.get_mask_mesh()
    #tm.draw(fig)
    plt.tight_layout()
    plt.show()
    
    print(f'Patient-ID: {patient_id}, diagnosis: {roi_1.get_metadata()}')
    print(f'The voxel volume of the ROI is: {roi_1.get_voxel_volume()}')
    print(f'The volume of the ROI (from triangular mesh) is: {roi_1.get_mesh_volume()}')
    print(f'The surface area of the ROI (from triangular mesh) is: {roi_1.get_surface_area()}')
