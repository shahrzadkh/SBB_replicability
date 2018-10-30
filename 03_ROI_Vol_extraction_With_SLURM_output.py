#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Aug 31 10:08:50 2017

@author: brain
"""
"""

Now after generating the binary maps of significat clusters, this script reads the path of the ROIs and calculates the mean / median GMV within eahch ROI within eahc subject.
These mean values are then written into a CSV table. 
The script is basically calling the fslstats frm nipype "Imagestats"

"""

import os
import numpy as np 
import pandas as pd
from GLM_and_ROI_generation import vol_extraction


#%% Step 1: reading the main table and create phenotypical subsamples (age & sex matched)
run_masch = '/data/'
subsampling_scripts_base_dir = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", "scripts/25_11_2017")
original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/IQ_Results_No_outlier_splits_1000_perm"
Other_important_variables =['T1_weighted_useful']
Diagnosis_exclusion_criteria = 'loose'
run ='_Run_0'
Image_top_DIR = run_masch + "BnB2/Team_Eickhoff/eNKI_DATA/DATA/eNKI"
Mask_file_complete = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Masks/binned_FZJ100_all_c1meanT1.nii.gz')
Template_Slurm_Submision_script_path = os.path.join(subsampling_scripts_base_dir , 'SLURM_que_template_modified')
Test_sample_p = [0.3, 0.5, 0.7]
    
n_jobs = 30   
 

# For Age only ROIS:
    
ROI_lists_file = os.path.join(original_base_folder, 'all_ROI_lists_path_cog_tests.txt') ## This is a file, in each line has the ROI_list_full_path for different tests, ...
Confounders_names =['Age_current', 'Sex', 'EDU_years']

f = open(os.path.join(original_base_folder, 'all_ROI_base_name_path_cog_tests.txt'), 'w+')
S = open(os.path.join(original_base_folder, 'all_Sample_table_with_GMV_info_path_cog_tests.txt'), 'w+')



try:
    with open(ROI_lists_file) as L:
        lines = L.read().splitlines()
except:
    pass
for ROI_list_full_path in lines:
    D=os.path.dirname(ROI_list_full_path).split('/')
    print(D)
    Test_base_DIR = os.path.join('/'+os.path.join(*D[:-7]))
    print(Test_base_DIR)
    Test_var = D[-6]
    print(Test_var)
    merged_file_full_path = os.path.join(Test_base_DIR, '4D_images/grouped_main_sample_'+ Test_var +'.nii.gz')
    Sample_info_table_CSV_full_path = os.path.join(Test_base_DIR, 'sample_CSV/grouped_main_sample_'+ Test_var + '.csv')
    where_to_save_ROI_Table = os.path.join(Test_base_DIR, 'Secondary_CSVs_for_correlations')
    ROI_table_name_suffix = Test_var
    Sample_table_with_GMV_info_full_path, ROI_base_names_text_file_full_path, ROI_effects  = vol_extraction(ROI_list_full_path,merged_file_full_path,\
                                                                                                            Sample_info_table_CSV_full_path,\
                                                                                                            where_to_save_ROI_Table, ROI_table_name_suffix,\
                                                                                                            what_to_extract='mean',T_stats_Flag =1, n_jobs =n_jobs)
    ROIS_T_max_Full_path = ROI_effects['ROIS_T_max_Full_path']
    ROIS_Cohen_d_Full_path = ROI_effects['ROIS_Cohen_d_Full_path']
    ROIS_r_efect_size_Full_path = ROI_effects['ROIS_r_efect_size_Full_path']
    
    with open(ROI_base_names_text_file_full_path) as k:
        rois = k.read().splitlines()
    if len(rois) >0:
        T = open(os.path.join(where_to_save_ROI_Table, 'Test_vars.txt'), 'w+')
        T.write(Test_var+'\n')
        T.close()
        T = open(os.path.join(where_to_save_ROI_Table, 'Covariates.txt'), 'w+')
        for i in Confounders_names:
            
            T.write(i+'\n')
        T.close()
        print(ROI_base_names_text_file_full_path)
        f.write(ROI_base_names_text_file_full_path +'\n')
    
        S.write(Sample_table_with_GMV_info_full_path +'\n')
        
S.close()
f.close()
    
    


