"""
Test script that saves per-patient NIfTI volumes.

Accumulates 2D slice outputs across all slices for each patient,
then writes registered / translated / dvf as .nii.gz files.

Output directory structure:
    <save_dir>/<pro_name>/<patient_id>/
        moving.nii.gz        -- moving image (input A)
        fixed.nii.gz         -- fixed image  (input B)
        translated.nii.gz    -- T(A): INN-translated moving
        registered.nii.gz    -- R(T(A)): registered result
        dvf.nii.gz           -- deformation field (H x W x S x 2)

All volumes are in the original [-1, 1] intensity range.
"""

import os
import json
import numpy as np
import torch
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_np(tensor):
    """(1, C, H, W) tensor → (C, H, W) numpy array on CPU."""
    return tensor.squeeze(0).detach().cpu().numpy()


def _save_nifti(vol, path):
    """Save a numpy array as NIfTI with identity affine."""
    img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
    nib.save(img, path)


def flush_patient(patient_id, accum, save_dir):
    """Stack accumulated slices and write NIfTI files for one patient."""
    patient_dir = os.path.join(save_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    def stack_scalar(key):
        # list of (H, W) arrays → (H, W, S)
        return np.stack(accum[key], axis=-1)

    def stack_vector(key):
        # list of (2, H, W) arrays → (H, W, S, 2)
        arr = np.stack(accum[key], axis=-1)   # (2, H, W, S)
        return arr.transpose(1, 2, 3, 0)      # (H, W, S, 2)

    _save_nifti(stack_scalar('moving'),     os.path.join(patient_dir, 'moving.nii.gz'))
    _save_nifti(stack_scalar('fixed'),      os.path.join(patient_dir, 'fixed.nii.gz'))
    _save_nifti(stack_scalar('translated'), os.path.join(patient_dir, 'translated.nii.gz'))
    _save_nifti(stack_scalar('registered'), os.path.join(patient_dir, 'registered.nii.gz'))
    _save_nifti(stack_vector('dvf'),        os.path.join(patient_dir, 'dvf.nii.gz'))

    print(f'  saved {patient_id} ({len(accum["moving"])} slices) → {patient_dir}')


# ── main test loop ────────────────────────────────────────────────────────────

def test(opt, save_dir):
    opt.num_threads   = 0
    opt.batch_size    = 1
    opt.serial_batches = True   # keeps slice order: patient0/0, patient0/1, ...
    opt.no_flip       = True
    opt.display_id    = -1

    dataset = create_dataset(opt)
    model   = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    current_patient = None
    accum = defaultdict(list)

    for data in tqdm(dataset):
        patient_id = data['patient_id'][0]   # batch_size=1 → first element
        slice_idx  = data['slice_idx'][0].item()

        # flush when patient changes
        if current_patient is not None and patient_id != current_patient:
            flush_patient(current_patient, accum, save_dir)
            accum = defaultdict(list)

        current_patient = patient_id

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        # visuals values are (1, 1, H, W) or (1, 2, H, W) tensors
        accum['moving'].append(_to_np(visuals['real_A'])[0])      # (H, W)
        accum['fixed'].append(_to_np(visuals['real_B'])[0])       # (H, W)
        accum['translated'].append(_to_np(visuals['fake_B'])[0])  # (H, W)
        accum['registered'].append(_to_np(visuals['registered'])[0])  # (H, W)
        accum['dvf'].append(_to_np(visuals['dvf']))               # (2, H, W)

    # flush last patient
    if current_patient is not None:
        flush_patient(current_patient, accum, save_dir)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_path = 'Checkpoint'
    pro_name  = '***'          # ← change to your experiment name

    result_dir = os.path.join('Results_NIfTI', pro_name)
    os.makedirs(result_dir, exist_ok=True)

    opt = TestOptions().parse()
    with open(os.path.join(save_path, pro_name, 'test_setting.json'), 'r') as f:
        opt.__dict__.update(json.load(f))

    opt.name           = pro_name
    opt.checkpoints_dir = save_path

    for arg in vars(opt):
        print(f'{arg:<25} {str(getattr(opt, arg))}')

    test(opt, result_dir)
