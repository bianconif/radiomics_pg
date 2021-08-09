import json

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

#Average spacing
#avg_spacing = roi_1.get_average_spacing()
#print(f'Average spacing: {avg_spacing}')

    #Generate triangula mesh on mask
    tm = TriangularMesh.by_marching_cubes(roi_1)
    tm.show()
    
    print(f'Patient-ID: {patient_id}, diagnosis: {roi_1.get_metadata()}')
    print(f'The voxel volume of the ROI is: {roi_1.get_voxel_volume()}')
    print(f'The volume of the ROI (from triangular mesh) is: {roi_1.get_mesh_volume()}')
    print(f'The surface area of the ROI (from triangular mesh) is: {roi_1.get_surface_area()}')
