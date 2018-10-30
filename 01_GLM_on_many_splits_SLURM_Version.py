#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:47:28 2018

@author: skharabian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the initial script that is used to generate slirm submission files for randomise. 
Here basically for a given origianl sample (eNKI) and for every psychological score (names are cbased on the columns of the eNKI table),...
The origial sample is splitted in to Disocvery (sample1) and test sample (Sample2). The number of the participants in each sample is defined based on "test sample_p".
For example, for TEst_Sample_p = 0.3, Discovery sample consists of 70% of the original sample and test sample will have 30%.

This procedure is done #Split (here 100) times and in each split, Dicosvery and test subsamples are chosen to be matched for age (defined within a limit (e.g. here the limit is 10 years) and sex
This splitting procedure is performed using the fnction: several_split_train_test_index_generation

Then within eahc split, For eahc Discovery and Test subsample, Preprocessed T1-weighted images are merged together and the design matrix is generated (with covariates (here: age, sex) and the respective behavioral score)
Then depending on the Slurm_que Flag and other GLM specifications, either randomise command will run for each Discovery group OR the submission script for Slurm is generated. 

In case of Slurm submission script, these should be then submitted manually to Slurm through BATCH.

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

def _multiple_stages_to_run_GLM_for_one_split(Working_Dir,Table_main_filtered_full_path,Train_index, Test_index,NIFTI_loading_Setting, GLM_Settings,x):     
    Split_working_dir = os.path.join(Working_Dir, 'Split_' + str(x))
    
    try:
        os.makedirs(Split_working_dir)
    except OSError:
        if not os.path.isdir(Split_working_dir):
            raise
        
    main_Table = importing_table_from_csv(Table_main_filtered_full_path)
    main_Table.loc[np.array(Train_index['train_' + str(x)]) , 'Which_sample'] = 1
    main_Table.loc[np.array(Test_index['test_'  + str(x)]) , 'Which_sample'] = 2
    
    First_group = main_Table[main_Table['Which_sample'] ==1]
    Second_group = main_Table[main_Table['Which_sample'] ==2]
    First_group = First_group.reset_index(drop= True)
    Second_group = Second_group.reset_index(drop= True)
    
    new_samples_base_name = os.path.basename(Table_main_filtered_full_path).strip("main_sample_")
    #main_table_gender_specific_full_path = Table_main_filtered_full_path
    Split_sample_CSV_dir= os.path.join(Split_working_dir,'sample_CSV')
    try:
        os.makedirs(Split_sample_CSV_dir)
    except OSError:
        if not os.path.isdir(Split_sample_CSV_dir):
            raise
    
    grouped_main_sample_full_path = os.path.join(Split_sample_CSV_dir, 'grouped_main_sample_' + new_samples_base_name)
    main_Table.to_csv(grouped_main_sample_full_path, index = False)
    first_group_full_path = os.path.join(Split_sample_CSV_dir,'Sample_1_' + new_samples_base_name)
    First_group.to_csv(first_group_full_path, sep=',', index = False)
    second_group_full_path = os.path.join(Split_sample_CSV_dir,'Sample_2_' + new_samples_base_name)
    Second_group.to_csv(second_group_full_path, sep=',', index = False)
    
    
    #####
    '''
    At the end of this stage I have generated: /Shahrzad's Folder/Base_analysis_forder/Test_specific_folder/
                                                                                                            sample_CSV/main_sample_+test_variable_name+.csv
                                                                                                            Split_X/
                                                                                                                    sample_CSV/
                                                                                                                                grouped_main_sample_+test_variable_name+.csv
                                                                                                                                Sample_1_+test_variable_name+.csv
                                                                                                                                Sample_2_+test_variable_name+.csv
                                                                                                                                
     So I will have the following variables:
         first_group_full_path, second_group_full_path, grouped_main_sample_full_path and Split_working_dir
    '''
    
    
    #%% Stage 2: 4D file generation:
       
    
    # Initializing to "Create merged files for subsamples"
    
    Image_top_DIR = NIFTI_loading_Setting['Image_top_DIR']
    Mask_file_complete = NIFTI_loading_Setting['Mask_file_complete']
    Base_Sample_name= NIFTI_loading_Setting['Base_Sample_name']
    mod_method = NIFTI_loading_Setting['mod_method']
    ##### End of fix input
    #Template Input:
    Sample_info_table_path = first_group_full_path #os.path.join(Base_working_dir, test_variable_name +'partial_corr'+ run, 'sample_CSV','Sample_1_'+test_variable_name+'nii.gz')
    ## Template Output:
    # os.path.join(Base_dir, test_variable_name +'partial_corr'+ run, '4D_images', 'Sample_1_'+test_variable_name+'.nii.gz')
    # This should be set-up inside the script running load_NIFTI_DATA_FROM_TABLE:
    merged_sample_dir = os.path.join(os.path.split((os.path.dirname(Sample_info_table_path)))[0], '4D_images')  
    merged_image_name = os.path.splitext(os.path.basename(Sample_info_table_path))[0]
    Sample_info_table = pd.read_csv(Sample_info_table_path)
    try:
        os.makedirs(merged_sample_dir)
    except OSError:
        if not os.path.isdir(merged_sample_dir):
            raise
    __ = load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Sample_info_table, merged_sample_dir,\
                                      merged_image_name, Mask_file_complete, Base_Sample_name, \
                                      modulation_method = mod_method)
        
    # repeate Step 2 for sample 2: The code that cosnsits of the load_Nifti function, should return the full path of the merged file again
    #Image_top_DIR = "/data/BnB2/Team_Eickhoff/eNKI_DATA/DATA/eNKI_all"
    Image_top_DIR = NIFTI_loading_Setting['Image_top_DIR']
    Mask_file_complete = NIFTI_loading_Setting['Mask_file_complete']
    Base_Sample_name= NIFTI_loading_Setting['Base_Sample_name']
    mod_method = NIFTI_loading_Setting['mod_method']
    ##### End of fix input
    #Template Input:
    Sample_info_table_path2 = grouped_main_sample_full_path #os.path.join(Base_working_dir, test_variable_name +'partial_corr'+ run, 'Sample_2_'+test_variable_name+'.nii.gz')
    ## Template Output:
    # os.path.join(Base_dir, test_variable_name +'partial_corr'+ run, '4D_images', 'Sample_1_'+test_variable_name+'.nii.gz')
    # This should be set-up inside the script running load_NIFTI_DATA_FROM_TABLE:
    # To exclude the "sample_CSV" folder 
    merged_sample_dir2 = os.path.join(os.path.split((os.path.dirname(Sample_info_table_path2)))[0], '4D_images') 
    try:
        os.makedirs(merged_sample_dir2)
    except OSError:
        if not os.path.isdir(merged_sample_dir2):
            raise
    # To exclude the ".csv" from name of the file.
    merged_image_name2 = os.path.splitext(os.path.basename(Sample_info_table_path2))[0]
    Sample_info_table2 = pd.read_csv(Sample_info_table_path2)
     
    __ = load_NIFTI_DATA_FROM_TABLE(Image_top_DIR, Sample_info_table2, merged_sample_dir2,\
                                      merged_image_name2, Mask_file_complete, Base_Sample_name, \
                                      modulation_method = mod_method)
    
    
    
    
        
     
    ######################## 
        
    #%% Stage 3: GLM on Sample1:
    Flag_TFCE = GLM_Settings['Flag_TFCE']
    n_GLM_core = GLM_Settings['n_GLM_core']
    n_GLM_perm = GLM_Settings['n_GLM_perm']
    GLM_method = GLM_Settings['GLM_method']# or 'nilearn'
    P_correction_method = GLM_Settings['P_correction_method'] #'FWE' # or 
    Mask_file_complete = GLM_Settings['Mask_file_complete']
    SLURM_Que = GLM_Settings['SLURM_Que']
    Template_Submision_script_path =GLM_Settings['Template_Submision_script_path']
    ##### End of fix input
    # Template Inputs
    GLM_Sample_info_table_CSV_full_path = first_group_full_path
    Working_dir = Split_working_dir
    # os.path.join(Base_working_dir, test_variable_name +'partial_corr'+ run)
    # Inputs inside the script
    # Create design_matrix (for FSL TFCE)  based on Table csv and selected confounders and 
    
    GLM_merged_image_name = os.path.splitext(os.path.splitext(os.path.basename(GLM_Sample_info_table_CSV_full_path))[0])[0]
    
        
    #Template output:
    #t1_full_path:          os.path.join(Working_dir,'GLM_Stats_dir', $test_var_name, $GLM_method, $test_var_name +'_' + $GLM_method + '_stats_tstat1.nii.gz')     
    #p1_full_path:          os.path.join(Working_dir,'GLM_Stats_dir', $test_var_name, $GLM_method, $test_var_name +'_' + $GLM_method + '_stats_' + $P_correction_method + '_corrp_tstat1.nii.gz')     
    #t2_full_path:          os.path.join(Working_dir,'GLM_Stats_dir', $test_var_name, $GLM_method, $test_var_name +'_' + $GLM_method + '_stats_tstat2.nii.gz')     
    #p2_full_path:          os.path.join(Working_dir,'GLM_Stats_dir', $test_var_name, $GLM_method, $test_var_name +'_' + $GLM_method + '_stats_' + $P_correction_method + '_corrp_tstat2.nii.gz')     
    #design_text_full_path: os.path.join(Working_dir,'GLM_Stats_dir', $test_var_name, $GLM_method,'design.txt')
    
    
    # here we call actual permutations
    t1_full_path, p1_full_path, t2_full_path, p2_full_path, design_text_full_path = GLM_on_Sample(Working_dir,\
                                                                                                  GLM_Sample_info_table_CSV_full_path ,\
                                                                                                  Confounders_names,\
                                                                                                  test_variable_name,\
                                                                                                  GLM_merged_image_name,Mask_file_complete,\
                                                                                                  merged_Flag = 1, Image_top_DIR ="",\
                                                                                                  mod_method = mod_method,\
                                                                                                  Base_Sample_name =Base_Sample_name, n_perm = n_GLM_perm,\
                                                                                                  n_core = n_GLM_core, Flag_TFCE = Flag_TFCE, SLURM_Que = SLURM_Que, Template_Submision_script_path = Template_Submision_script_path)
    
    
    return 

    
    
           
    
    
if __name__ == "__main__":
    #### load Original table, and generate test specific table and folder:
    
    #run_masch = '/media/sf_Volumes/'
    run_masch = '/data/'
    Main_Sample_info_table_CSV_full_path = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Python_processed_original_assessment_data_csv/Table_for_scripts/Outliers_removed_eNKI_MRI_cog_anxiety_Learning_BMI_IQ_roun1.csv')
    subsampling_scripts_base_dir = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", "scripts/25_11_2017")
    original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/TMT_Results_No_outlier_splits_1000_perm"
    Other_important_variables =['T1_weighted_useful']
    Diagnosis_exclusion_criteria = 'loose'
    run ='_Run_0'
    Image_top_DIR = run_masch + "BnB2/T1_DB/DATA/eNKI"
    Mask_file_complete = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Masks/binned_FZJ100_all_c1meanT1.nii.gz')
    Template_Slurm_Submision_script_path = os.path.join(subsampling_scripts_base_dir , 'SLURM_que_template_modified')
    Test_sample_p = [0.3, 0.5, 0.7]
    
    NIFTI_loading_Setting = {'Image_top_DIR': Image_top_DIR, 'Mask_file_complete': Mask_file_complete, 'Base_Sample_name': 'NKI','mod_method' : 'non_linearOnly'}
    
    GLM_Settings = {'Flag_TFCE': 1,'n_GLM_core':30, 'n_GLM_perm': 1000, 'GLM_method':'fsl', 'P_correction_method' : 'tfce', 'Mask_file_complete': Mask_file_complete,'SLURM_Que' : True, 'Template_Submision_script_path' : Template_Slurm_Submision_script_path}
    
    n_jobs = GLM_Settings['n_GLM_core']
    n_perm = GLM_Settings['n_GLM_perm']
    
    ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 0.00001, 'min_cluster_size_corr': 100, \
                  'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
                  'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
#    ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 8, 'min_cluster_size_corr': 100, \
#                  'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
#                  'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'
    
    
    ##### End of fix input
    ## End of "Initializing for main table importing"
    # Inputs, which need to be defined in the script:
    #               test_variable_name = $test_variable_name ### Not sure here
    # Template of output: 
    #               os.path.join(Base_dir, test_variable_name +'partial_corr'+ run,'sample_CSV', 'main_sample_'+$test_variable_name+'.csv')    
    
    Table = pd.read_csv(Main_Sample_info_table_CSV_full_path)
    Confounders_names =['Age_current', 'Sex', 'EDU_years']
    Tests_list_file_full = os.path.join(original_base_folder, "Cog_list.txt")
    try:
        with open(Tests_list_file_full) as L:
            Tests = L.read().splitlines()
    except:
        pass
    
    
     
    #Tests = ['BMI', 'WASI_Total_IQ', 'WASI_Perceptual', 'WASI_Vocabulary'] 
    for test_variable_name in Tests:
        for percent_test_size in Test_sample_p:
                
            Split_settings = {'test_sample_size' : percent_test_size, 'Age_step_size': 10, 'gender_selection' : None, 'n_split' : 100, 'Sex_col_name': 'sex_mr', 'Age_col_name' : 'Age_current'}
            
            test_sample_size= Split_settings['test_sample_size']
            min_ROI_size = ROIS_stats['min_cluster_size_corr']
            Base_dir = os.path.join(original_base_folder, test_variable_name + '_' + str(test_sample_size) + 'test_sample_' + str(n_perm)+ '_perm_' + str(min_ROI_size) + '_voxelminClust')
            
            
            try:
                os.makedirs(Base_dir)
            except OSError:
                if not os.path.isdir(Base_dir):
                    raise
                 
        
            
        
            Table_main_filtered_full_path, Working_dir = create_test_var_specific_subsamples(Base_dir,\
                                                                                             Main_Sample_info_table_CSV_full_path,\
                                                                                             Confounders_names, test_variable_name,\
                                                                                             run, Other_important_variables,\
                                                                                             Diagnosis_exclusion_criteria)
        
        
            Sex_col_name = Split_settings['Sex_col_name']
            Age_col_name = Split_settings['Age_col_name']
            Age = Age_col_name
            Age_step_size = Split_settings['Age_step_size']
            gender_selection =Split_settings['gender_selection']
            n_split = Split_settings['n_split']
            Train_index, Test_index = several_split_train_test_index_generation(subsampling_scripts_base_dir,\
                                                                                Table_main_filtered_full_path,\
                                                                                n_split,\
                                                                                Sex_col_name,\
                                                                                Age_col_name,\
                                                                                Age_step_size,\
                                                                                test_size = test_sample_size,\
                                                                                gender_selection = gender_selection)
            Original_data_table = importing_table_from_csv(Table_main_filtered_full_path)
                
                
            #for i_split in np.arange(Split_settings['n_split']):
            
            
                
                
                
                
            Parallel(n_jobs= n_jobs)(delayed(_multiple_stages_to_run_GLM_for_one_split)(Working_dir,Table_main_filtered_full_path,Train_index, Test_index,NIFTI_loading_Setting, GLM_Settings,x) for x in np.arange(n_split))
                    
                            