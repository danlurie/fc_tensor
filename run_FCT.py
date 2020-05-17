#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import itertools
from nilearn import image
from scipy import stats

## Set paths to files and load the brain mask image.
func_path = '/home/despoB/dlurie/Projects/despolab_lesion/data/patients/derivatives/fmriprep/work/fmriprep_wf/single_subject_117_wf/func_preproc_task_rest_acq_128px_run_01_wf/bold_stc_wf/copy_xform/sub-117_task-rest_acq-128px_run-01_bold_valid_tshift_xform.nii.gz'

brain_mask_img = image.load_img('/home/despoB/dlurie/Projects/despolab_lesion/data/patients/derivatives/fmriprep/work/fmriprep_wf/single_subject_117_wf/func_preproc_task_rest_acq_128px_run_01_wf/bold_bold_trans_wf/bold_reference_wf/enhance_and_skullstrip_bold_wf/combine_masks/ref_image_valid_corrected_brain_mask_maths.nii.gz')

confounds_path = '/home/despoB/dlurie/Projects/despolab_lesion/data/patients/derivatives/postproc/out/sub-117/func/sub-117_task-rest_acq-128px_run-01_desc-confounds_variant-lesionfix_regressors_expansion.tsv'

cf_df = pd.read_csv(confounds_path, sep='\t')
cf_df.fillna(value=0, inplace=True)

reg_cols = ['csf', 'csf_derivative1', 'csf_derivative1_power2', 'csf_power2',
       'global_signal','global_signal_derivative1', 'global_signal_derivative1_power2',
       'global_signal_power2','trans_x','trans_x_derivative1', 'trans_x_power2', 'trans_x_derivative1_power2',
       'trans_y', 'trans_y_derivative1', 'trans_y_power2',
       'trans_y_derivative1_power2', 'trans_z', 'trans_z_derivative1',
       'trans_z_derivative1_power2', 'trans_z_power2', 'rot_x',
       'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
       'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2',
       'rot_y_power2', 'rot_z', 'rot_z_derivative1',
       'rot_z_derivative1_power2', 'rot_z_power2']

## Denoie, smooth, and maks BOLD data.
func_img = image.clean_img(func_path, high_pass=0.008, t_r=2, confounds=cf_df.loc[:, reg_cols].values, mask_img=brain_mask_img)
func_img = image.smooth_img(func_img, 3)

brain_mask_data = brain_mask_img.get_fdata()
brain_coords = np.argwhere(brain_mask_data == 1)

## Create a design matrix to describe direction from each voxel to its neighbors
unit = [1,0,-1]
Mi = np.empty((27,6), dtype='float')
M_orig = np.flip(np.array(list(itertools.product(unit, repeat=3))), axis=1).astype('float')
M = M_orig.copy()

for row_idx, row_vector in enumerate(M_orig):
    row_sum = np.sum(abs(row_vector))
    if row_sum == 2:
        M[row_idx] = row_vector * (1 / np.sqrt(2))
    elif row_sum == 3:
        M[row_idx] = row_vector * (1 / np.sqrt(3))
    Mi[row_idx] = [M[row_idx,0]**2,             # x^2
                   2*M[row_idx,0]*M[row_idx,1], # 2xy
                   2*M[row_idx,0]*M[row_idx,2], # 2xz
                   M[row_idx,1]**2,             # y^2
                   2*M[row_idx,1]*M[row_idx,2], # 2yz
                   M[row_idx,2]**2]             # z^2

Mi = np.delete(Mi, 13, axis=0)
M_orig = np.delete(M_orig, 13, axis=0)

## Estimate FC tensors at each voxel
ts_data = func_img.get_fdata()
tensor_store = []
ev_store = []
fa_store = []
for idx, center_coord in enumerate(brain_coords):
    C = []
    for neighbor_loc in M_orig:
        try:
            i,j,k = center_coord
            neighbor_coord = center_coord + neighbor_loc
            i_,j_,k_ = neighbor_coord.astype(int)
            r, p = stats.spearmanr(ts_data[i,j,k,:], ts_data[i_,j_,k_,:])
            if np.isnan(r):
                r=0
                #print("Zero correlation encountered for voxel {}".format(idx))
        except:
            r=0
        C.append(r**2)
    C = np.array(C)
    T = np.linalg.inv(Mi.T @ Mi) @ Mi.T @ C
    T_t = np.array([[T[0],T[1],T[2]],
                    [T[1],T[3],T[4]],
                    [T[2],T[4],T[5]]])
    tensor_store.append(T_t)
    ev = np.linalg.eig(T_t)
    ev_store.append(ev)
    fa = np.sqrt(0.5 * ((ev[0][0]-ev[0][1])**2 + (ev[0][0]-ev[0][2])**2 + (ev[0][1]-ev[0][2])**2 ) / ((ev[0][0]**2) + (ev[0][1]**2) + (ev[0][2]**2)))
    fa_store.append(fa)

## Create and save FFA map
fa_data = brain_mask_img.get_fdata().copy()
brain_voxels = brain_mask_data == 1
fa_data[~brain_voxels] = 0
fa_data[brain_voxels] = np.array(fa_store)
fa_img = image.new_img_like(brain_mask_img, fa_data)
fa_img.to_filename('/home/despoB/dlurie/Projects/fc_tractography/data/sub-117/space-BOLD/sub-117_task-rest_acq-128px_run-01_space-BOLD_variant-penalize_FFA.nii.gz')

## Create and save FCPD map.
ev_data = func_img.get_fdata().copy()
ev_data = np.empty_like(ev_data)
dat_zero = ev_data == 0
ev_data[dat_zero] = np.nan
ev_data = ev_data[:,:,:,:3]
evals = np.array([list(i[0]) for i in ev_store])
peval_idx = np.argmax(evals, axis=1)
evecs = np.array([list(i[1]) for i in ev_store])

evecs_sorted = []
for idx in range(len(evecs)):
    evecs_sorted.append((evecs[idx][:,peval_idx[idx]]))
evecs_sorted = np.array(evecs_sorted)

ev_data[:,:,:,0][brain_voxels] = evecs_sorted[:][:,0]
ev_data[:,:,:,1][brain_voxels] = evecs_sorted[:][:,1]
ev_data[:,:,:,2][brain_voxels] = evecs_sorted[:][:,2]
ev_img = image.new_img_like(func_img, ev_data)
ev_img.to_filename('/home/despoB/dlurie/Projects/fc_tractography/data/sub-117/space-BOLD/sub-117_task-rest_acq-128px_run-01_space-BOLD_variant-penalize_pFCT.nii.gz')


