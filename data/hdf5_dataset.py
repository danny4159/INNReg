import h5py
import numpy as np
import torch
from tqdm import tqdm
from data.base_dataset import BaseDataset


def _register_3d_rigid(fixed_np: np.ndarray, moving_np: np.ndarray, z_pad: int = 20) -> np.ndarray:
    """Rigid 3D registration of moving → fixed using SimpleITK.

    z_pad slices are reflect-padded before registration and cropped after,
    preventing diagonal boundary artifacts caused by z-direction translation/rotation.

    Args:
        fixed_np:  (H, W, D) float32 array, [-1, 1] normalized
        moving_np: (H, W, D) float32 array, [-1, 1] normalized
        z_pad:     number of slices to pad on each end in z before registration
    Returns:
        registered moving volume (H, W, D) float32, background filled with -1
    """
    import SimpleITK as sitk

    D_orig = fixed_np.shape[2]

    fixed_pad  = np.pad(fixed_np,  ((0, 0), (0, 0), (z_pad, z_pad)), mode='edge')
    moving_pad = np.pad(moving_np, ((0, 0), (0, 0), (z_pad, z_pad)), mode='edge')

    # numpy (H, W, D) → SimpleITK expects (D, H, W)
    fixed_sitk  = sitk.GetImageFromArray(fixed_pad.transpose(2, 0, 1).astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_pad.transpose(2, 0, 1).astype(np.float32))

    initial_tf = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200
    )
    R.SetOptimizerScalesFromIndexShift()
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(initial_tf, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_tf = R.Execute(fixed_sitk, moving_sitk)

    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, final_tf,
        sitk.sitkLinear, -1.0, moving_sitk.GetPixelID(),
    )

    # SimpleITK (D, H, W) → numpy (H, W, D), crop back to original z
    result_pad = sitk.GetArrayFromImage(resampled).transpose(1, 2, 0).astype(np.float32)
    return result_pad[:, :, z_pad:z_pad + D_orig]


class HDF5Dataset(BaseDataset):
    """
    2D slice dataset from an HDF5 file with structure:
        f[modality][patient_id] -> (H, W, n_slices)

    Data is assumed to be already normalized to [-1, 1] per patient.

    Optionally applies 3D rigid registration (SimpleITK) of moving → fixed
    per patient at init time and caches the result in memory.

    Usage: --dataset_mode hdf5 --hdf5_path /path/to/file.h5
           --moving_modality T1_moved --fixed_modality T2
           [--apply_rigid_registration]
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hdf5_path', type=str, required=True,
                            help='path to HDF5 file')
        parser.add_argument('--moving_modality', type=str, default='T1_moved_5mm',
                            help='moving image modality key in HDF5 (e.g. T1_moved, T2_moved_3mm, T1)')
        parser.add_argument('--fixed_modality', type=str, default='T2',
                            help='fixed image modality key in HDF5 (e.g. T2, FLAIR)')
        parser.add_argument('--apply_rigid_registration', action='store_true',
                            help='apply 3D rigid pre-registration (SimpleITK) before feeding to model')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.hdf5_path = opt.hdf5_path
        self.moving_key = opt.moving_modality
        self.fixed_key = opt.fixed_modality
        self.apply_rigid_registration = getattr(opt, 'apply_rigid_registration', False)

        with h5py.File(self.hdf5_path, 'r') as f:
            self.patient_ids = sorted(f[self.fixed_key].keys())
            # slice counts may differ per patient
            self.slice_counts = [f[self.fixed_key][pid].shape[2] for pid in self.patient_ids]

        self.cumulative = np.cumsum([0] + self.slice_counts)

        # flat index list
        self.samples = [
            (pid, s)
            for pid, n in zip(self.patient_ids, self.slice_counts)
            for s in range(n)
        ]

        # 3D rigid registration cache: {patient_id: registered_moving_vol (H, W, D)}
        self.reg_cache = {}
        if self.apply_rigid_registration:
            print(f'[RigidReg] Running 3D rigid registration for {len(self.patient_ids)} patients ...')
            with h5py.File(self.hdf5_path, 'r') as f:
                for pid in tqdm(self.patient_ids, desc='[RigidReg]'):
                    fixed_vol  = f[self.fixed_key][pid][...].astype(np.float32)
                    moving_vol = f[self.moving_key][pid][...].astype(np.float32)
                    self.reg_cache[pid] = _register_3d_rigid(fixed_vol, moving_vol)
            print('[RigidReg] Done.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        patient_id, slice_idx = self.samples[index]

        with h5py.File(self.hdf5_path, 'r') as f:
            fixed = f[self.fixed_key][patient_id][:, :, slice_idx].astype(np.float32)
            if self.apply_rigid_registration:
                moving = self.reg_cache[patient_id][:, :, slice_idx]
            else:
                moving = f[self.moving_key][patient_id][:, :, slice_idx].astype(np.float32)

        A = torch.from_numpy(moving).unsqueeze(0)  # (1, H, W), range [-1, 1]
        B = torch.from_numpy(fixed).unsqueeze(0)   # (1, H, W), range [-1, 1]

        return {
            'A': A,
            'B': B,
            'A_paths': f'{patient_id}/{slice_idx:03d}',
            'B_paths': f'{patient_id}/{slice_idx:03d}',
            'patient_id': patient_id,
            'slice_idx': slice_idx,
        }
