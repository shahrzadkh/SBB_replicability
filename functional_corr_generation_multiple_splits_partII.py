#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:22:42 2018

@author: skharabian
"""

""" 
This code is basically trying to create partial correlation (spearman or pearson) for eahc sample, between mean GMV in each ROI and the specific behavioral score, for which the ROI was derived from the exploratory susample.
This script additionally runs the associations for 1000 boostrap resamples and generates measures of stability, Confidence intervals (90%) and the error bars (for matplotlib), as well as p-values. 
Moreover, it saves the masked vesion of the P-value and sample correlation. The masking is performed by setting the p-value or the sample correlation equal to zero, if CI crossess zero. 

Last but not least, this scripts saves the percentage of the bootsrap samples (out of 1000 resmaples), that resulted in "SIGNIFICANT" (i.e. p<0.05) correlation coefficient, 

A simpler alternative to this script (without all the boostrapping and stability stuff, but with useful correlation coeffiecient between mean GMV in eac ROI and behaviral score, in each sample (discovery, Test, separately) is the "functional_corr_generation_multiple_splits.py")

Most important outcome:
At the end of this script, for each ROI, I will have txt files, refering to the path of ".pkl" files (tables) "partial correlation" of the behavioral score and mean GMV in each ROI




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
original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/IQ_Results_No_outlier_splits_1000_perm"
#original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/ADNI/analysis/Immediate_recall_dummy_diagn_dummy_site"

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
n_boot =1000
correlation_methods = ['sPartial'] 
Bootstrap_method = ['simple'] 

means_list_Full_path = open(os.path.join(original_base_folder, 'means_list_Full_path_cog_tests2.txt'), 'w+')  
P_Value_list_Full_path = open(os.path.join(original_base_folder, 'P_Value_list_Full_path_cog_tests2.txt'), 'w+')  
err_list_Full_path = open(os.path.join(original_base_folder, 'err_list_Full_path_cog_tests2.txt'), 'w+')  
masked_P_Value_list_Full_path = open(os.path.join(original_base_folder, 'masked_P_Value_list_Full_path_cog_tests2.txt'), 'w+')  
masked_means_list_Full_path = open(os.path.join(original_base_folder, 'masked_means_list_Full_path_cog_tests2.txt'), 'w+')  
CI_list_Full_path = open(os.path.join(original_base_folder, 'CI_list_Full_path_cog_tests2.txt'), 'w+')
perc_Sig_bootstrap_list_Full_path = open(os.path.join(original_base_folder, 'perc_Sig_bootstrap_list_Full_path_cog_tests2.txt'), 'w+')  
  
with open(ROI_base_names) as f:
    ROI_list = f.read().splitlines()
        
       
with open(Samples_CSV_paths) as f:
    Samples = f.read().splitlines()

unique_ROIS= list(set(ROI_list))





## This part is only for the ROI_stability script, otherwise delete it ####
text_refering_to_location_of_list_of_packed_means = open(os.path.join(original_base_folder, 'packed_means_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_err = open(os.path.join(original_base_folder, 'packed_err_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_masked_means = open(os.path.join(original_base_folder, 'packed_masked_means_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_P_Value = open(os.path.join(original_base_folder, 'packed_P_Value_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_masked_P_Value = open(os.path.join(original_base_folder, 'packed_masked_P_Value_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_CIs = open(os.path.join(original_base_folder, 'packed_CIs_list_for_stability_cog_tests2.txt'), 'w+')
text_refering_to_location_of_list_of_packed_perc_Sig_bootstrap = open(os.path.join(original_base_folder, 'packed_perc_Sig_bootstrap_list_for_stability_cog_tests2.txt'), 'w+')


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
        test_specific_err_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_err_list_Full_path.txt')
        test_specific_masked_means_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_masked_means_list_Full_path.txt')
        test_specific_P_Value_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_P_Value_list_Full_path.txt')
        test_specific_masked_P_Value_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_masked_P_Value_list_Full_path.txt')
        test_specific_CI_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_CI_list_Full_path.txt')
        test_specific_perc_Sig_bootstrap_list_Full_path= os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'test_specific_perc_Sig_bootstrap_list_Full_path.txt')
   
        test_specific_means_list_file = open(test_specific_means_list_Full_path, 'w+')  
        test_specific_err_list_file = open(test_specific_err_list_Full_path, 'w+')  
        test_specific_masked_means_list_file = open(test_specific_masked_means_list_Full_path, 'w+')  
        test_specific_P_Value_list_file = open(test_specific_P_Value_list_Full_path, 'w+')
        test_specific_masked_P_Value_list_file = open(test_specific_masked_P_Value_list_Full_path, 'w+')
        test_specific_CI_list_file = open(test_specific_CI_list_Full_path, 'w+')
        test_specific_perc_Sig_bootstrap_list_file = open(test_specific_perc_Sig_bootstrap_list_Full_path, 'w+')
        
        ## This part is only for the ROI_stability script, otherwise delete it ####
        Test_specific_mean_dictionary = {key: [] for key in n_groups}
        Test_specific_err_dictionary = {key: [] for key in n_groups}
        Test_specific_masked_mean_dictionary = {key: [] for key in n_groups}
        Test_specific_P_Value_dictionary = {key: [] for key in n_groups}
        Test_specific_masked_P_Value_dictionary = {key: [] for key in n_groups}
        Test_specific_CI_dictionary = {key: [] for key in n_groups}
        Test_specific_perc_Sig_bootstrap_dictionary = {key: [] for key in n_groups}
        
        for g in n_groups:
            Test_specific_mean_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_err_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_masked_mean_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_P_Value_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_masked_P_Value_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_CI_dictionary[g] = {key: [] for key in correlation_methods}
            Test_specific_perc_Sig_bootstrap_dictionary[g] = {key: [] for key in correlation_methods}
            for correlation_method in correlation_methods:
                Test_specific_mean_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_err_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_masked_mean_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_P_Value_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_masked_P_Value_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_CI_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
                Test_specific_perc_Sig_bootstrap_dictionary[g][correlation_method] = {key: [] for key in Bootstrap_method}
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
                    ROI_specific_err_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_err_list_file.txt')
                    ROI_specific_masked_mean_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_masked_mean_list_file.txt')
                    ROI_specific_P_Value_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_P_Value_list_file.txt')
                    ROI_specific_masked_P_Value_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_masked_P_Value_list_file.txt')
                    ROI_specific_CI_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_CI_list_file.txt')
                    ROI_specific_perc_Sig_bootstrap_list_file_path = os.path.join(stats_dir, boot_meth + '_' + correlation_method + '_perc_Sig_bootstrap_list_file.txt')
                    
                    
                    means_list_file = open(ROI_specific_means_list_file_path, 'w+') 
                    err_list_file = open(ROI_specific_err_list_file_path, 'w+') 
                    masked_means_list_file = open(ROI_specific_masked_mean_list_file_path, 'w+') 
                    P_val_list_file = open(ROI_specific_P_Value_list_file_path, 'w+') 
                    masked_P_val_list_file = open(ROI_specific_masked_P_Value_list_file_path, 'w+') 
                    CI_list_file = open(ROI_specific_CI_list_file_path, 'w+') 
                    perc_Sig_bootstrap_list_file = open(ROI_specific_perc_Sig_bootstrap_list_file_path, 'w+') 
                    
                    for g in n_groups: # I have 3 groups.
                        
                    ### Here I modified the input of the function to meet my requirements
                    
                        if boot_meth == 'BCA':
                            mean= Functional_profiling_scikit_boot_bca(ROI_name, stats_dir, Sample_table_with_GMV_info__full_path,\
                                                                        Cog_list_full_path_for_profiling = Cog_test_name_file, Confounders_list_full_path = Confounders_list_full_path,\
                                                                        Group_selection_column_name = Group_selection_column_name, Group_division = True,\
                                                                        Group_selection_Label = g, Sort_correlations = True, correlation_method = correlation_method,\
                                                                        alpha = 0.05, n_boot = n_boot)[0]
                        elif boot_meth == 'simple':
                            mean,err, _, masked_mean, P_Value_full_path,\
                            masked_P_Val, CIs_full_path, perc_Sig_bootstrap_full_path = Functional_profiling_simple_boot(ROI_name, stats_dir, Sample_table_with_GMV_info__full_path,\
                                                                                                                        Cog_list_full_path_for_profiling = Cog_test_name_file,Confounders_list_full_path = Confounders_list_full_path,\
                                                                                                                        Group_selection_column_name = Group_selection_column_name, Group_division = True,\
                                                                                                                        Group_selection_Label = g, Sort_correlations = True, correlation_method = correlation_method,\
                                                                                                                        alpha = 0.05, n_boot = n_boot, n_jobs = n_job)
                                                            
            
                        
                        means_list_file.write(mean+'\n')
                        err_list_file.write(err+'\n')
                        masked_means_list_file.write(masked_mean+'\n')
                        P_val_list_file.write(P_Value_full_path+'\n')
                        masked_P_val_list_file.write(masked_P_Val+'\n')
                        CI_list_file.write(CIs_full_path+'\n')
                        perc_Sig_bootstrap_list_file.write(perc_Sig_bootstrap_full_path+'\n')
                        ## This part is only for the ROI_stability script, otherwise delete it ####
                        Test_specific_mean_dictionary[g][correlation_method][boot_meth].append(mean)
                        Test_specific_err_dictionary[g][correlation_method][boot_meth].append(err)
                        Test_specific_masked_mean_dictionary[g][correlation_method][boot_meth].append(masked_mean)
                        Test_specific_P_Value_dictionary[g][correlation_method][boot_meth].append(P_Value_full_path)
                        Test_specific_masked_P_Value_dictionary[g][correlation_method][boot_meth].append(masked_P_Val)
                        Test_specific_CI_dictionary[g][correlation_method][boot_meth].append(CIs_full_path)
                        Test_specific_perc_Sig_bootstrap_dictionary[g][correlation_method][boot_meth].append(perc_Sig_bootstrap_full_path)
                        
                        ## End of: This part is only for the ROI_stability script, otherwise delete it ####       
                        
                        
                        
                    means_list_file.close()
                    err_list_file.close()
                    masked_means_list_file.close()
                    P_val_list_file.close()
                    masked_P_val_list_file.close()
                    CI_list_file.close()
                    perc_Sig_bootstrap_list_file.close()

                    ## This part is only for the ROI_stability script, otherwise delete it ####
                    test_specific_means_list_file.write(ROI_specific_means_list_file_path+'\n')
                    test_specific_err_list_file.write(ROI_specific_err_list_file_path+'\n')
                    test_specific_masked_means_list_file.write(ROI_specific_masked_mean_list_file_path+'\n')
                    test_specific_P_Value_list_file.write(ROI_specific_P_Value_list_file_path+'\n')
                    test_specific_masked_P_Value_list_file.write(ROI_specific_masked_P_Value_list_file_path+'\n')
                    test_specific_CI_list_file.write(ROI_specific_CI_list_file_path+'\n')
                    test_specific_perc_Sig_bootstrap_list_file.write(ROI_specific_perc_Sig_bootstrap_list_file_path+'\n')
                    
                    
                    ## End of: This part is only for the ROI_stability script, otherwise delete it ####       
        
        
         ## This part is only for the ROI_stability script, otherwise delete it ####
        ### Now in order to have the means of all ROIS of a test, concatenated together:
        # NOTE: ofcourse I loose the ordering of the correlations and the tests:

        
        
        for correlation_method in correlation_methods:
            for boot_meth in Bootstrap_method:
                PATH_of_all_samples_for_each_boot_corr_means_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_means_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_means = open(PATH_of_all_samples_for_each_boot_corr_means_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_err_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_err_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_err = open(PATH_of_all_samples_for_each_boot_corr_err_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_masked_means_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_masked_means_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_masked_means = open(PATH_of_all_samples_for_each_boot_corr_masked_means_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_P_Value_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_P_Value_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_P_Value = open(PATH_of_all_samples_for_each_boot_corr_P_Value_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_masked_P_Value_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_masked_P_Value_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_masked_P_Value = open(PATH_of_all_samples_for_each_boot_corr_masked_P_Value_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_CI_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_CI_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_CI = open(PATH_of_all_samples_for_each_boot_corr_CI_file, 'w+')
                PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap_file = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'packed_' + boot_meth + correlation_method+ '_perc_Sig_bootstrap_list_file.txt')
                PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap = open(PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap_file, 'w+')
                for g in n_groups:
                    Full_name_and_path_of_the_new_dataframe_mean = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_means.pkl')
                    PACKED_means = merging_dataframes_columnwise(Test_specific_mean_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_mean)
                    PATH_of_all_samples_for_each_boot_corr_means.write(PACKED_means+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_err = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_err.pkl')
                    PACKED_err = merging_dataframes_columnwise(Test_specific_err_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_err)
                    PATH_of_all_samples_for_each_boot_corr_err.write(PACKED_err+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_masked_mean = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_masked_means.pkl')
                    PACKED_masked_means = merging_dataframes_columnwise(Test_specific_masked_mean_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_masked_mean)
                    PATH_of_all_samples_for_each_boot_corr_masked_means.write(PACKED_masked_means+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_P_Value = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_P_Values.pkl')
                    PACKED_P_Value = merging_dataframes_columnwise(Test_specific_P_Value_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_P_Value)
                    PATH_of_all_samples_for_each_boot_corr_P_Value.write(PACKED_P_Value+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_P_Value = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_P_Values.pkl')
                    PACKED_P_Value = merging_dataframes_columnwise(Test_specific_P_Value_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_P_Value)
                    PATH_of_all_samples_for_each_boot_corr_P_Value.write(PACKED_P_Value+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_masked_P_Value = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_masked_P_Values.pkl')
                    PACKED_masked_P_Value = merging_dataframes_columnwise(Test_specific_masked_P_Value_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_masked_P_Value)
                    PATH_of_all_samples_for_each_boot_corr_masked_P_Value.write(PACKED_masked_P_Value+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_CI = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_CI.pkl')
                    PACKED_CI = merging_dataframes_columnwise(Test_specific_CI_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_CI)
                    PATH_of_all_samples_for_each_boot_corr_CI.write(PACKED_CI+'\n')
                    
                    Full_name_and_path_of_the_new_dataframe_perc_Sig_bootstrap = os.path.join(os.path.dirname(ROI_base_names_text_file_full_path), 'Sample_' + str(g) +'_' + boot_meth + '_' + correlation_method + '_test_specific_packed_perc_Sig_bootstrap.pkl')
                    PACKED_perc_Sig_bootstrap = merging_dataframes_columnwise(Test_specific_perc_Sig_bootstrap_dictionary[g][correlation_method][boot_meth], Full_name_and_path_of_the_new_dataframe_perc_Sig_bootstrap)
                    PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap.write(PACKED_perc_Sig_bootstrap+'\n')
                
                PATH_of_all_samples_for_each_boot_corr_means.close()
                PATH_of_all_samples_for_each_boot_corr_err.close()
                PATH_of_all_samples_for_each_boot_corr_masked_means.close()
                PATH_of_all_samples_for_each_boot_corr_P_Value.close()
                PATH_of_all_samples_for_each_boot_corr_masked_P_Value.close()
                PATH_of_all_samples_for_each_boot_corr_CI.close()
                PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap.close()

                text_refering_to_location_of_list_of_packed_means.write(PATH_of_all_samples_for_each_boot_corr_means_file+'\n')
                text_refering_to_location_of_list_of_packed_err.write(PATH_of_all_samples_for_each_boot_corr_err_file+'\n')
                text_refering_to_location_of_list_of_packed_masked_means.write(PATH_of_all_samples_for_each_boot_corr_masked_means_file+'\n')
                text_refering_to_location_of_list_of_packed_P_Value.write(PATH_of_all_samples_for_each_boot_corr_P_Value_file+'\n')
                text_refering_to_location_of_list_of_packed_masked_P_Value.write(PATH_of_all_samples_for_each_boot_corr_masked_P_Value_file+'\n')
                text_refering_to_location_of_list_of_packed_CIs.write(PATH_of_all_samples_for_each_boot_corr_CI_file+'\n')
                text_refering_to_location_of_list_of_packed_perc_Sig_bootstrap.write(PATH_of_all_samples_for_each_boot_corr_perc_Sig_bootstrap_file+'\n')

        ## End of: This part is only for the ROI_stability script, otherwise delete it ####
       
        
            
    test_specific_means_list_file.close()       
    test_specific_P_Value_list_file.close()
    test_specific_err_list_file.close()
    test_specific_masked_means_list_file.close()
    test_specific_masked_P_Value_list_file.close()
    test_specific_CI_list_file.close()
    test_specific_perc_Sig_bootstrap_list_file.close()
    
    means_list_Full_path.write(test_specific_means_list_Full_path+'\n')
    P_Value_list_Full_path.write(test_specific_P_Value_list_Full_path+'\n')
    err_list_Full_path.write(test_specific_err_list_Full_path+'\n')
    masked_P_Value_list_Full_path.write(test_specific_masked_P_Value_list_Full_path+'\n')
    masked_means_list_Full_path.write(test_specific_masked_means_list_Full_path+'\n')
    CI_list_Full_path.write(test_specific_CI_list_Full_path+'\n')
    perc_Sig_bootstrap_list_Full_path.write(test_specific_perc_Sig_bootstrap_list_Full_path+'\n')
    
    
    
    
means_list_Full_path.close() # this referes to the text files with address of text files 
P_Value_list_Full_path.close()
err_list_Full_path.close()
masked_P_Value_list_Full_path.close()
masked_means_list_Full_path.close()
CI_list_Full_path.close()
perc_Sig_bootstrap_list_Full_path.close()


## This part is only for the ROI_stability script, otherwise delete it ####
text_refering_to_location_of_list_of_packed_means.close()
text_refering_to_location_of_list_of_packed_P_Value.close()
text_refering_to_location_of_list_of_packed_err.close()
text_refering_to_location_of_list_of_packed_masked_means.close()
text_refering_to_location_of_list_of_packed_masked_P_Value.close()
text_refering_to_location_of_list_of_packed_CIs.close()
text_refering_to_location_of_list_of_packed_perc_Sig_bootstrap.close()


## End of: This part is only for the ROI_stability script, otherwise delete it ####