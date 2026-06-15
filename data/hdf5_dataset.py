import h5py
import numpy as np
import torch
from data.base_dataset import BaseDataset


class HDF5Dataset(BaseDataset):
    """
    2D slice dataset from an HDF5 file with structure:
        f[modality][patient_id] -> (256, 256, n_slices)

    Data is assumed to be already normalized to [-1, 1] per patient.

    Usage: --dataset_mode hdf5 --hdf5_path /path/to/file.h5
           --moving_modality T1_moved_5mm --fixed_modality T2
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hdf5_path', type=str, required=True,
                            help='path to HDF5 file')
        parser.add_argument('--moving_modality', type=str, default='T1_moved_5mm',
                            help='moving image modality key in HDF5 (e.g. T1_moved, T2_moved_3mm, T1)')
        parser.add_argument('--fixed_modality', type=str, default='T2',
                            help='fixed image modality key in HDF5 (e.g. T2, FLAIR)')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.hdf5_path = opt.hdf5_path
        self.moving_key = opt.moving_modality
        self.fixed_key = opt.fixed_modality

        with h5py.File(self.hdf5_path, 'r') as f:
            self.patient_ids = sorted(f[self.fixed_key].keys())
            first = self.patient_ids[0]
            self.n_slices = f[self.fixed_key][first].shape[2]

        # flat index: (patient_id, slice_idx) in deterministic order
        self.samples = [
            (pid, s)
            for pid in self.patient_ids
            for s in range(self.n_slices)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        patient_id, slice_idx = self.samples[index]

        with h5py.File(self.hdf5_path, 'r') as f:
            moving = f[self.moving_key][patient_id][:, :, slice_idx].astype(np.float32)
            fixed  = f[self.fixed_key][patient_id][:, :, slice_idx].astype(np.float32)

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
