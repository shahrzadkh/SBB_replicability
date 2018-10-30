#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:42 2018

@author: skharabian
"""
"""

This is simpler ltrenative to "functional_corr_generation_multiple_splits_partII.py"



"""



# Calculate correlations with the test:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np 
from GLM_and_ROI_generation import Functional_profiling_scikit_boot_bca, Functional_profiling_simple_boot, merging_dataframes_columnwise
import pandas as pd

run_masch = '/data/'
subsampling_scripts_base_dir = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", "scripts/25_11_2017")
#original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/IQ_Results_No_outlier_splits_1000_perm"
original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/ADNI/analysis/Immediate_recall_dummy_diagn_dummy_site"

run ='_Run_0'
 

# For Age only ROIS:
#Confounders_names =['Sex', 'EDU_years']
    
#Confounders_names = ['Age_current', 'Sex', 'EDU_years']

ROI_base_names = os.path.join(original_base_folder, 'all_ROI_base_name_path_cog_tests.txt')
Samples_CSV_paths = os.path.join(original_base_folder, 'all_Sample_table_with_GMV_info_path_cog_tests.txt')

#Cog_list_full_path_for_profiling = os.path.join(Base_Dir, 'Cog_list_full_path_for_profiling_no_age.txt')
#Cog_list_full_path_for_profiling = os.path.join(Base_Dir, 'Cog_list_full_path_for_profiling_with_age.txt')
#Cog_list_full_path_for_profiling = os.path.join(Base_Dir, 'Cog_list_full_path_for_profiling_no_age.txt')
#Cog_list_full_path_for_profiling = "/data/BnB2/USER/Shahrzad/eNKI_modular/Test_Results_no_outlier_multiple_splits/Age_current_partial_corr_Run_0/Split_16/Secondary_CSVs_for_correlations/Test_vars.txt"





Group_selection_column_name = 'Which_sample'
n_groups = ['all', '1', '2']
n_job = 31 # number of simple bootsraps at once
n_boot =0
correlation_methods = ['sPartial'] 
Bootstrap_method = ['simple'] 

means_list_Full_path = open(os.path.join(original_base_folder, 'means_list_Full_path_cog_tests.txt'), 'w+')  
P_Value_list_Full_path = open(os.path.join(original_base_folder, 'P_Value_list_Full_path_cog_tests.txt'), 'w+')  

with open(ROI_base_names) as f:
    ROI_list = f.read().splitlines()
        
       
with open(Samples_CSV_paths) as f:
    Samples = f.read().splitlines()

unique_ROIS= list(set(ROI_list))





## This part is only for the ROI_stability script, otherwise delete it ####
#text_refering_to_location_of_list_of_packed_means = open(os.path.join(Base_Dir, 'packed_means_list_for_stability_Cog_tests.txt'), 'w+')
#text_refering_to_location_of_list_of_packed_err_CIs = open(os.path.join(Base_Dir, 'packed_err_CIs_list_for_stability_Cog_tests.txt'), 'w+')
#text_refering_to_location_of_list_of_packed_avs = open(os.path.join(Base_Dir, 'packed_avs_list_for_stability_Cog_tests.txt'), 'w+')

text_refering_to_location_of_list_of_packed_means = open(os.path.join(original_base_folder, 'packed_means_list_for_stability_cog_tests.txt'), 'w+')

text_refering_to_location_of_list_of_packed_P_Value = open(os.path.join(original_base_folder, 'packed_P_Value_list_for_stability_cog_tests.txt'), 'w+')



## End of: This part is only for the ROI_stability script, otherwise delete it ####        




for i in np.arange(len(unique_ROIS)):
    
    ROI_base_names_text_file_full_path = unique_ROIS[i]#os.path.join(Base_Dir, 'Age_current_partial_corr_Run_0/Secondary_CSVs_for_correlations/Age_current.txt')
    
    ## This part is only for the ROI_stability script, otherwise delete it ####
    name_of_the_test_var = os.path.splitext(os.path.basename(unique_ROIS[i]))[0]
    file_for_roi_stability = open(os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'COG_name_for_stability.txt'), 'w+')
    file_for_roi_stability.write(name_of_the_test_var+'\n')
    file_for_roi_stability.close()
    Cog_test_name_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'COG_name_for_stability.txt')
    ## This part is only for the ROI_stability script, otherwise delete it ####
    
    
    
    
    #** This can be problematic, as we uniqued the ROIS
    #Sample_table_with_GMV_info__full_path = Samples[i]
    Sample_table_with_GMV_info__full_path = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path),'grouped_main_sample_' + os.path.splitext(os.path.basename(unique_ROIS[i]))[0] +'_mean_GMV_' + os.path.splitext(os.path.basename(unique_ROIS[i]))[0] +'.csv')    #os.path.join(Base_Dir, 'Age_current_partial_corr_Run_0/Secondary_CSVs_for_correlations/grouped_main_sample_Age_current_Median_GMV_Age_current.csv')
    Confounders_list_full_path = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Covariates.txt')

    if len(ROI_base_names_text_file_full_path)>0:
        print(ROI_base_names_text_file_full_path)
        print(Sample_table_with_GMV_info__full_path)
        test_specific_means_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_means_list_Full_path.txt')
        test_specific_P_Value_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_P_Value_list_Full_path.txt')
        
        test_specific_means_list_file = open(test_specific_means_list_Full_path, 'w+')  
        test_specific_P_Value_list_file = open(test_specific_P_Value_list_Full_path, 'w+')
        
        ## This part is only for the ROI_stability script, otherwise delete it ####
        Test_specific_mean_dictionary = {key: [] for key in n_groups}
        Test_specific_P_Value_dictionary = {key: [] for key in n_groups}
        
        
        for g in n_groups:
            Test_specific_mean_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_P_Value_dictionary[g] = {key: [] for key in correlation_methods}
            for correlation_method in correlation_methods:
                Test_specific_mean_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_P_Value_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
        ## End of: This part is only for the ROI_stability script, otherwise delete it ####        
        with open(ROI_base_names_text_file_full_path) as f:
            ROIS = f.read().splitlines()
        for roi_id in np.arange(len(ROIS)):
            ROI_name = ROIS[roi_id]
            stats_dir = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), ROI_name)
            try:
                os.makedirs(stats_dir)
            except OSError:
                if not os.path.isdir(stats_dir):
                    raise    
            for correlation_method in correlation_methods:
                for boot_meth in Bootstrap_method:
                    
                    
                    ROI_specific_means_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_means_list_file.txt')
                    ROI_specific_P_Value_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_P_Value_list_file.txt')
                    means_list_file = open(ROI_specific_means_list_file_path, 'w+') 
                    P_val_list_file = open(ROI_specific_P_Value_list_file_path, 'w+') 
                    
                    for g in n_groups: # I have 3 groups.
                        
                    ### Here I modified the input of the function to meet my requirements
                    
                        if boot_meth == 'BCA':
                            mean= Functional_profiling_scikit_boot_bca(ROI_name, stats_dir, Sample_table_with_GMV_info__full_path,\
                                                                        Cog_list_full_path_for_profiling = Cog_test_name_file, Confounders_list_full_path = Confounders_list_full_path,\
                                                                        Group_selection_column_name = Group_selection_column_name, Group_division = True,\
                                                                        Group_selection_Label = g, Sort_correlations = True, correlation_method = correlation_method,\
                                                                        alpha = 0.05, n_boot = n_boot)[0]
                        elif boot_meth == 'simple':
                            mean,_, _, _, P_Value_full_path, _, _ = Functional_profiling_simple_boot(ROI_name, stats_dir, Sample_table_with_GMV_info__full_path,\
                                                                                                     Cog_list_full_path_for_profiling = Cog_test_name_file,Confounders_list_full_path = Confounders_list_full_path,\
                                                                                                     Group_selection_column_name = Group_selection_column_name, Group_division = True,\
                                                                                                     Group_selection_Label = g, Sort_correlations = True, correlation_method = correlation_method,\
                                                                                                     alpha = 0.05, n_boot = n_boot, n_jobs = n_job)
                            
            
                        
                        means_list_file.write(mean+'\n')
                        P_val_list_file.write(P_Value_full_path+'\n')
                        
                        ## This part is only for the ROI_stability script, otherwise delete it ####
                        Test_specific_mean_dictionary[g][correlation_method][boot_meth].append(mean)
                        Test_specific_P_Value_dictionary[g][correlation_method][boot_meth].append(P_Value_full_path)
                        
                        ## End of: This part is only for the ROI_stability script, otherwise delete it ####       
                        
                        
                        
                    means_list_file.close()
                    P_val_list_file.close()
                    ## This part is only for the ROI_stability script, otherwise delete it ####
                    test_specific_means_list_file.write(ROI_specific_means_list_file_path+'\n')
                    test_specific_P_Value_list_file.write(ROI_specific_P_Value_list_file_path+'\n')
                    
                    ## End of: This part is only for the ROI_stability script, otherwise delete it ####       
        
        
         ## This part is only for the ROI_stability script, otherwise delete it ####
        ### Now in order to have the means of all ROIS of a test, concatenated together:
        # NOTE: ofcourse I loose the ordering of the correlations and the tests:

        
        
        for correlation_method in correlation_methods:
            for boot_meth in Bootstrap_method:
                PATH_of_all_samples_for_each_boot_corr_means_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_means_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_means = open(PATH_of_all_samples_for_each_boot_corr_means_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_P_Value_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_P_Value_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_P_Value = open(PATH_of_all_samples_for_each_boot_corr_P_Value_file, 'w+')
                
                for g in n_groups:
                    Full_name_and_path_of_the_new_dataframe_mean = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_means.pkl')
                    PACKED_means = merging_dataframes_columnwise(Test_specific_mean_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_mean)
                    PATH_of_all_samples_for_each_boot_corr_means.write(PACKED_means+'\n')
                    Full_name_and_path_of_the_new_dataframe_P_Value = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_P_Values.pkl')
                    PACKED_P_Value = merging_dataframes_columnwise(Test_specific_P_Value_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_P_Value)
                    PATH_of_all_samples_for_each_boot_corr_P_Value.write(PACKED_P_Value+'\n')
                    
                
                PATH_of_all_samples_for_each_boot_corr_means.close()
                PATH_of_all_samples_for_each_boot_corr_P_Value.close()
                text_refering_to_location_of_list_of_packed_means.write(PATH_of_all_samples_for_each_boot_corr_means_file+'\n')
                text_refering_to_location_of_list_of_packed_P_Value.write(PATH_of_all_samples_for_each_boot_corr_P_Value_file+'\n')
        ## End of: This part is only for the ROI_stability script, otherwise delete it ####
       
        
            
    test_specific_means_list_file.close()       
    test_specific_P_Value_list_file.close()
    means_list_Full_path.write(test_specific_means_list_Full_path+'\n')
    P_Value_list_Full_path.write(test_specific_P_Value_list_Full_path+'\n')
means_list_Full_path.close() # this referes to the text files with address of text files 
P_Value_list_Full_path.close()
## This part is only for the ROI_stability script, otherwise delete it ####
text_refering_to_location_of_list_of_packed_means.close()
text_refering_to_location_of_list_of_packed_P_Value.close()

## End of: This part is only for the ROI_stability script, otherwise delete it ####