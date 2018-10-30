#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:30:39 2018

@author: skharabian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After running the 01_GLM.....py code (or additionally finished the submissions to SLURM) , we will have randomise with predefienedspecification run on each o the Discovery subsamples (for each behavioral score, each split, each sample size)
Now to collect the location of the positive/negaive significantly correlated ROIs with the behavioral score, we run the following code. 
It basically goes through the folders of randomise results and applies the cluster command of FSL and creates binary ROIs. Here one can have ROIs' ids that are either in order of Maximum T-vale or Maximum size. 

The path to these binary ROIs are svaed separately for positive /negatively correlated ROIs in a Text file. 


"""
import os
import numpy as np 
import pandas as pd
from joblib import Parallel, delayed 
from GLM_and_ROI_generation import importing_table_from_csv,\
                                    create_test_var_specific_subsamples,\
                                    several_split_train_test_index_generation,\
                                    load_NIFTI_DATA_FROM_TABLE, GLM_on_Sample,\
                                    Create_binned_ROIS

def _ROI_binarizing_one_split(Working_Dir, GLM_Settings, ROIS_stats, test_variable_name, R,x):     
    Split_working_dir = os.path.join(Working_Dir, 'Split_' + str(x))
    
    

    
    #%% Stage 5:
    
    # Initialization to "Create_binned_ROIS":
    #ROIS_stats = {'ROIS_stats_type': 'FWE_corrp','thresh_corrp': 0.05, 'min_cluster_size_corr': 10, \
    #              'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
    #              'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
   
    GLM_method = GLM_Settings['GLM_method']#'nilearn' # or 
    P_correction_method = GLM_Settings['P_correction_method']  # 'FWE' # or 
    
    # Template Inputs
    
    Working_dir = Split_working_dir
    design_text_full_path = os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method,'design.txt')
    # Template Outputs:
    # os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, 'ROI', $ROIS_stats.ROIS_stats_type, 'tstat1', list_of_ROIs_based_on_', $ROIS_stats.Order_ROIs, '.txt')
    # os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, 'ROI', $ROIS_stats.ROIS_stats_type, 'tstat2', list_of_ROIs_based_on_', $ROIS_stats.Order_ROIs, '.txt')
    
         
    # Input from inside the script: 
    if ROIS_stats['ROIS_stats_type'] == 'uncorr':
        
        Stats_map_full_path1 = os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, test_variable_name +'_' + GLM_method + '_stats_tstat1.nii.gz')
        Stats_map_full_path2 = os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, test_variable_name +'_' + GLM_method + '_stats_tstat2.nii.gz')
    else:
        Stats_map_full_path1 = os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, test_variable_name +'_' + GLM_method + '_stats_' + P_correction_method + '_corrp_tstat1.nii.gz')
        Stats_map_full_path2 = os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method, test_variable_name +'_' + GLM_method + '_stats_' + P_correction_method + '_corrp_tstat2.nii.gz')
      
        
    ROI_list_full_path1 = Create_binned_ROIS(Stats_map_full_path1, design_text_full_path, Stats_type = ROIS_stats['ROIS_stats_type'], \
                                             thresh_corrp = ROIS_stats['thresh_corrp'],T_Threshold = ROIS_stats['T_Threshold'], \
                                             min_cluster_size_corr = ROIS_stats['min_cluster_size_corr'], thresh_uncorrp =  ROIS_stats['thresh_uncorrp'], \
                                             min_cluster_size_uncorr = ROIS_stats['min_cluster_size_uncorr'], test_side = ROIS_stats['test_side'], \
                                             Order_ROIs = ROIS_stats['Order_ROIs'], Limit_num_ROIS = ROIS_stats['Limit_num_ROIS'], binned_ROIS = ROIS_stats['binned_ROIS'])
   
    
    
    #f_file.write(ROI_list_full_path1+'\n')  # python will convert \n to os.linesep
    R1 = ROI_list_full_path1 
    ROI_list_full_path2 = Create_binned_ROIS(Stats_map_full_path2, design_text_full_path, Stats_type = ROIS_stats['ROIS_stats_type'], \
                                             thresh_corrp = ROIS_stats['thresh_corrp'],T_Threshold = ROIS_stats['T_Threshold'], \
                                             min_cluster_size_corr = ROIS_stats['min_cluster_size_corr'], thresh_uncorrp =  ROIS_stats['thresh_uncorrp'], \
                                             min_cluster_size_uncorr = ROIS_stats['min_cluster_size_uncorr'], test_side = ROIS_stats['test_side'], \
                                             Order_ROIs = ROIS_stats['Order_ROIs'], Limit_num_ROIS = ROIS_stats['Limit_num_ROIS'], binned_ROIS = ROIS_stats['binned_ROIS'])
   
    empty_flag1 = np.load(os.path.join(os.path.dirname(Stats_map_full_path1), 'empty_flag.npy'))
    empty_flag2 = np.load(os.path.join(os.path.dirname(Stats_map_full_path2), 'empty_flag.npy'))
    
    np.save(os.path.join(os.path.join(Working_dir,'GLM_Stats_dir', test_variable_name, GLM_method), 'general_empty_flag.npy'), empty_flag1*empty_flag2)
    
    #f_file.write(ROI_list_full_path2+'\n')  # python will convert \n to os.linesep    
    R2 = ROI_list_full_path2
    R = {'R1':[R1], 'R2':[R2]}
    return R
    
    
           
    
    
if __name__ == "__main__":
    #### load Original table, and generate test specific table and folder:
    
    #run_masch = '/media/sf_Volumes/'
    run_masch = '/data/'
    Main_Sample_info_table_CSV_full_path = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Python_processed_original_assessment_data_csv/Table_for_scripts/Outliers_removed_eNKI_MRI_cog_anxiety_Learning_roun1.csv')
    subsampling_scripts_base_dir = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", "scripts/25_11_2017")
    original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/IQ_Results_No_outlier_splits_1000_perm"
    Other_important_variables =['T1_weighted_useful']
    Diagnosis_exclusion_criteria = 'loose'
    run ='_Run_0'
    Image_top_DIR = run_masch + "BnB2/Team_Eickhoff/eNKI_DATA/DATA/eNKI"
    Mask_file_complete = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Masks/binned_FZJ100_all_c1meanT1.nii.gz')
    Template_Slurm_Submision_script_path = os.path.join(subsampling_scripts_base_dir , 'SLURM_que_template_modified')
    Test_sample_p = [0.3, 0.5, 0.7]
    
    
    GLM_Settings = {'Flag_TFCE': 1,'n_GLM_core':30, 'n_GLM_perm': 1000, 'GLM_method':'fsl', 'P_correction_method' : 'tfce', 'Mask_file_complete': Mask_file_complete,'SLURM_Que' : True, 'Template_Submision_script_path' : Template_Slurm_Submision_script_path}
    
    n_jobs = GLM_Settings['n_GLM_core']
    n_perm = GLM_Settings['n_GLM_perm']
    
    
#    ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 0.00001, 'min_cluster_size_corr': 100, \
#                  'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
#                  'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
    ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 3.00001, 'min_cluster_size_corr': 100, \
                  'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
                  'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
#    ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 8, 'min_cluster_size_corr': 100, \
#                  'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
#                  'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
    
    
    Confounders_names =['Age_current', 'Sex', 'EDU_years']
    Tests_list_file_full = os.path.join(original_base_folder, "Cog_list.txt")
    try:
        with open(Tests_list_file_full) as L:
            Tests = L.read().splitlines()
    except:
        pass
    
    
     
    Tests = ['BMI']#, 'WASI_Total_IQ', 'WASI_Perceptual', 'WASI_Vocabulary']
    f_file = open(os.path.join(original_base_folder, 'all_ROI_lists_path_cog_tests_3.txt'), 'w+')
    locator_of_cog_specific_Pos_file = open(os.path.join(original_base_folder, 'locator_of_cog_specific_positive_ROIS_3.txt'), 'a+')
    locator_of_cog_specific_Neg_file = open(os.path.join(original_base_folder, 'locator_of_cog_specific_negative_ROIS_3.txt'), 'a+')
    
    #Tests = ['ANT_01']
    
    for test_variable_name in Tests:
        for percent_test_size in Test_sample_p:
                
            Split_settings = {'test_sample_size' : percent_test_size, 'Age_step_size': 10, 'gender_selection' : None, 'n_split' : 100, 'Sex_col_name': 'sex_mr', 'Age_col_name' : 'Age_current'}
            
            test_sample_size= Split_settings['test_sample_size']
            min_ROI_size = ROIS_stats['min_cluster_size_corr']
            Base_dir = os.path.join(original_base_folder, test_variable_name + '_' + str(test_sample_size) + 'test_sample_' + str(n_perm)+ '_perm_' + str(min_ROI_size) + '_voxelminClust')
            
            Working_dir = os.path.join(Base_dir, test_variable_name +'_partial_corr'+ run)
                 
            Cog_specific_Pos_file = open(os.path.join(Working_dir, 'positive_ROI_lists_path_all_splits_3.txt'), 'w+')
    
            Cog_specific_Neg_file = open(os.path.join(Working_dir, 'negative_ROI_lists_path_all_splits_3.txt'), 'w+')
            
        
            n_split = Split_settings['n_split']
            #for i_split in np.arange(Split_settings['n_split']):
         # Parallel(n_jobs= n_jobs)(delayed(_multiple_stages_to_run_GLM_for_one_split)(Working_dir,Table_main_filtered_full_path,Train_index, Test_index,NIFTI_loading_Setting, GLM_Settings,x) for x in np.arange(n_split))
                    
            R = [{'R1':[], 'R2': []} for i in np.arange(n_split)] # x100
            collected_R = Parallel(n_jobs= n_jobs)(delayed(_ROI_binarizing_one_split)(Working_dir, GLM_Settings,ROIS_stats,test_variable_name, R[x], x) for x in np.arange(n_split))
            
            
            for item in collected_R:
                f_file.write("%s\n" % item['R1'][0])
                Cog_specific_Pos_file.write("%s\n" % item['R1'][0])
            Cog_specific_Pos_file.close()
            for item in collected_R:
                f_file.write("%s\n" % item['R2'][0])
                Cog_specific_Neg_file.write("%s\n" % item['R2'][0])
            Cog_specific_Neg_file.close()
    
            locator_of_cog_specific_Pos_file.write(os.path.join(Working_dir, 'positive_ROI_lists_path_all_splits_3.txt') + "\n")
            locator_of_cog_specific_Neg_file.write(os.path.join(Working_dir, 'negative_ROI_lists_path_all_splits_3.txt')+ "\n")
                    
    f_file.close()
    locator_of_cog_specific_Pos_file.close()
    locator_of_cog_specific_Neg_file.close()                