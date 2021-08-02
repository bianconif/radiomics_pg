from utilities.misc import Roi
from utilities.geometry import TriangularMesh

patient_id = '115GA'
    
mask_file = f'../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/{patient_id}/nifti-scans-and-rois/CT/mask.nii'
scan_folder = f'../../../../Pac_F_Bianco/ProgettiRicerca/Radiomics/Datasets/Sassari/Studies/SPN-01/{patient_id}/dicom-scans/CT'

#roi = Roi.from_dcm_and_nii(mask_file, scan_folder)
#roi.save(f'tests/roi/{patient_id}.pkl')
roi_1 = Roi.from_pickle(source = f'tests/roi/{patient_id}.pkl')

#Average spacing
avg_spacing = roi_1.get_average_spacing()
print(f'Average spacing: {avg_spacing}')

#Generate triangula mesh on mask
tm = TriangularMesh.by_marching_cubes(roi_1)
tm.show()

print(f'The voxel volume of the ROI is: {roi_1.get_voxel_volume()}')
print(f'The surface area of the mesh is: {tm.get_surface_area()}')
print(f'The volume of the mesh is: {tm.get_volume()}')
print(f'The surface area of the ROI is: {roi_1.get_surface_area()}')
a = 0