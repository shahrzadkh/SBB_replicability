## Function to create frequency maps of  
import os

from nilearn.input_data import NiftiMasker
from GLM_and_ROI_generation import create_merged_file
import pandas as pd
import numpy as np
from nipype.interfaces.fsl import BinaryMaths

def multiply_image_by_constant(ROI_image_full_path, constant, file_suffix = 'stability'):
    
    bin_maths = BinaryMaths()
    bin_maths.inputs.in_file = ROI_image_full_path
    bin_maths.inputs.operation = "mul"
    bin_maths.inputs.operand_value = constant
    out_file = os.path.join(os.path.dirname(ROI_image_full_path),file_suffix + '_'+ os.path.basename(ROI_image_full_path))
    bin_maths.inputs.out_file = out_file
    print(bin_maths.cmdline)
    bin_maths.run()
    return out_file

## Help function for create_frequency_map_from_many_binary_images ###
def Create_Path_of_list_of_single_3D_files(file_pointing_to_tsts_ROIs):
    """ 
    
    file_pointing_to_tsts_ROIs is the file that has been created when I run the script "GLM_many_splits.py"
    it either has only text files reffering tstats1 ROIs for one cog-test or tstats2s or it can have both together. 
         
    """
    LIST_of_ROI_PATHS = []
    with open(file_pointing_to_tsts_ROIs) as L:
        lines = L.read().splitlines()
        
    for i in lines:
        with open(i) as K:
            ROI_temp = K.read().splitlines()
            
        for j in ROI_temp:
            LIST_of_ROI_PATHS.append(j)
            
    return LIST_of_ROI_PATHS
        
    
    
def create_frequency_map_from_many_binary_images(Path_of_list_of_single_3D_files ,probability_map_file_name, save_dir, n_total = 0):
    """
    Path_of_list_of_single_3D_files = LIST_of_ROI_PATHS (created from above HELP function)
    
    or it is a text file with list of ROIs taht should be summed together, directly.
    """
    
    if isinstance(Path_of_list_of_single_3D_files, str):
        with open(Path_of_list_of_single_3D_files) as L:
            ThreeD_files_list = L.read().splitlines()
    else: # Then it should be a list already
        ThreeD_files_list = Path_of_list_of_single_3D_files
        
    merged_file_name= "temporary"    
    fourD_file_path = create_merged_file(ThreeD_files_list, save_dir, merged_file_name = "temporary")
    if n_total == 0:
        n_total = len(ThreeD_files_list)
    masker = NiftiMasker(standardize=False)
    four_D_file_maksed = masker.fit_transform(fourD_file_path)
    # Here to calculate the percentage of being detected or not, for wach voxel, I do the following:
    sum_file = four_D_file_maksed.sum(axis = 0)
    sum_file = sum_file[:, np.newaxis]
    sum_file = sum_file.T
    sum_file = sum_file/n_total
    sum_file = sum_file*100
    file_to_save = os.path.join(save_dir, probability_map_file_name)
    masker.inverse_transform(sum_file).to_filename(file_to_save)
    os.remove(os.path.join(save_dir,merged_file_name + '.nii.gz'))
    return file_to_save

def define_replicability_of_correlation_in_2_half(mean_pkl_sample1,mean_pkl_sample2, mean_pkl_sample_all ='',add_P_Value = True,add_CI = False,add_perc_sig_boot = False, T_max_full_path ='', Cohen_d_full_path ='', r_efect_size_full_path ='', which_split = ''):
    """
    This will craete for one split, one replicability dataframe.
    """
    means_df_1 = pd.read_pickle(mean_pkl_sample1)
    means_df_2 = pd.read_pickle(mean_pkl_sample2)
    
    for i in np.arange(len(means_df_1.columns)):
        
        means_df_1[means_df_1.columns.values[i]] = means_df_1[means_df_1.columns.values[i]].str.get(0)
        
    for i in np.arange(len(means_df_2.columns)):
        
        means_df_2[means_df_2.columns.values[i]] = means_df_2[means_df_2.columns.values[i]].str.get(0)
    
    if os.path.isfile(mean_pkl_sample_all):
        means_df_all = pd.read_pickle(mean_pkl_sample_all)
        
        for i in np.arange(len(means_df_all.columns)):
        
            means_df_all[means_df_all.columns.values[i]] = means_df_all[means_df_all.columns.values[i]].str.get(0)
    
        effect_group_all = pd.DataFrame(means_df_all.values, columns=means_df_all.columns, index=['effect_size_group_all'])
    else:
        effect_group_all = pd.DataFrame([], index=['effect_size_group_all'])
    
        
        
        
    divide_df = pd.DataFrame(means_df_1.values/means_df_2.values, columns=means_df_1.columns, index=['mean_divide_Value'])
    
    multiplication_sign_df = pd.DataFrame(np.sign(means_df_1.values*means_df_2.values), columns=means_df_1.columns, index=['mean_Mul_Sign'])
    effect_group1 = pd.DataFrame(means_df_1.values, columns=means_df_1.columns, index=['effect_size_group1'])
    effect_group2 = pd.DataFrame(means_df_2.values, columns=means_df_1.columns, index=['effect_size_group2'])
   
    if add_P_Value == True:
        
        P_Value_df_1 =  pd.read_pickle(mean_pkl_sample1.strip("_means.pkl") + "_P_Values.pkl")
        P_Value_df_2 =  pd.read_pickle(mean_pkl_sample2.strip("_means.pkl") + "_P_Values.pkl")
        
        
        for i in np.arange(len(P_Value_df_1.columns)):
        
            P_Value_df_1[P_Value_df_1.columns.values[i]] = P_Value_df_1[P_Value_df_1.columns.values[i]].str.get(0)
    
        for i in np.arange(len(P_Value_df_2.columns)):
            
            P_Value_df_2[P_Value_df_2.columns.values[i]] = P_Value_df_2[P_Value_df_2.columns.values[i]].str.get(0)
        
        P_Val_group1 = pd.DataFrame(P_Value_df_1.values, columns=P_Value_df_1.columns, index=['P_Val_group1'])
        P_Val_group2 = pd.DataFrame(P_Value_df_2.values, columns=P_Value_df_2.columns, index=['P_Val_group2'])
   
        
        if os.path.isfile(mean_pkl_sample_all):
            
            P_Value_df_all =  pd.read_pickle(mean_pkl_sample_all.strip("_means.pkl") + "_P_Values.pkl")
            
            
            for i in np.arange(len(P_Value_df_all.columns)):
            
                P_Value_df_all[P_Value_df_all.columns.values[i]] = P_Value_df_all[P_Value_df_all.columns.values[i]].str.get(0)
        
            
            
            P_Val_group_all = pd.DataFrame(P_Value_df_all.values, columns=P_Value_df_all.columns, index=['P_Val_group_all'])
        else:
            P_Val_group_all = pd.DataFrame([], index=['P_Val_group_all'])
    else:
        P_Val_group_all = pd.DataFrame([], index=['P_Val_group_all'])
        P_Val_group1 = pd.DataFrame([], index=['P_Val_group1'])
        P_Val_group2 = pd.DataFrame([], index=['P_Val_group2'])
        
        
 ##     CI
    if add_CI == True:
        
        CI_df_1 =  pd.read_pickle(mean_pkl_sample1.strip("_means.pkl") + "_CI.pkl")
        CI_df_2 =  pd.read_pickle(mean_pkl_sample2.strip("_means.pkl") + "_CI.pkl")
        CI_mul_sign_group1 = pd.DataFrame(columns=CI_df_1.columns)
        CI_mul_sign_group2 = pd.DataFrame(columns=CI_df_2.columns)
        
        
        for i in np.arange(len(CI_df_1.columns)):
        
            CI_df_1[CI_df_1.columns.values[i]] = CI_df_1[CI_df_1.columns.values[i]].str.get(0)
            CI_mul_sign_group1[CI_df_1.columns.values[i]] = np.sign(CI_df_1[CI_df_1.columns.values[i]].str.get(0) * CI_df_1[CI_df_1.columns.values[i]].str.get(1))

        for i in np.arange(len(CI_df_2.columns)):
            
            CI_df_2[CI_df_2.columns.values[i]] = CI_df_2[CI_df_2.columns.values[i]].str.get(0)
            CI_mul_sign_group2[CI_df_2.columns.values[i]] = np.sign(CI_df_2[CI_df_2.columns.values[i]].str.get(0) * CI_df_2[CI_df_2.columns.values[i]].str.get(1))

        
        CI_group1 = pd.DataFrame(CI_df_1.values, columns=CI_df_1.columns, index=['CI_group1'])
        CI_group2 = pd.DataFrame(CI_df_2.values, columns=CI_df_2.columns, index=['CI_group2'])
        boot_sign_group1 = pd.DataFrame(CI_mul_sign_group1.values, columns=CI_df_1.columns, index=['boot_sign_group1'])
        boot_sign_group2 = pd.DataFrame(CI_mul_sign_group2.values, columns=CI_df_2.columns, index=['boot_sign_group2'])
   
        
        if os.path.isfile(mean_pkl_sample_all):
            
            CI_df_all =  pd.read_pickle(mean_pkl_sample_all.strip("_means.pkl") +"_CI.pkl")
            CI_mul_sign_all = pd.DataFrame(columns=CI_df_all.columns)
        
            
            for i in np.arange(len(CI_df_all.columns)):
            
                CI_df_all[CI_df_all.columns.values[i]] = CI_df_all[CI_df_all.columns.values[i]].str.get(0)
                CI_mul_sign_all[CI_df_all.columns.values[i]] = np.sign(CI_df_all[CI_df_all.columns.values[i]].str.get(0) * CI_df_all[CI_df_all.columns.values[i]].str.get(1))

            
            
            CI_group_all = pd.DataFrame(CI_df_all.values, columns=CI_df_all.columns, index=['CI_group_all'])
            boot_sign_all = pd.DataFrame(CI_mul_sign_all.values, columns=CI_df_all.columns, index=['boot_sign_all'])

        else:
            CI_group_all = pd.DataFrame([], index=['CI_group_all'])
    else:
        CI_group_all = pd.DataFrame([], index=['CI_group_all'])
        CI_group1 = pd.DataFrame([], index=['CI_group1'])
        CI_group2 = pd.DataFrame([], index=['CI_group2'])
        boot_sign_group1 = pd.DataFrame([], index=['boot_sign_group1'])
        boot_sign_group2 = pd.DataFrame([], index=['boot_sign_group2'])
        boot_sign_all = pd.DataFrame([], index=['boot_sign_all'])


 ##     add_perc_sig_boot
    if add_perc_sig_boot == True:
        
        perc_Sig_bootstrap_df_1 =  pd.read_pickle(mean_pkl_sample1.strip("_means.pkl") + "_perc_Sig_bootstrap.pkl")
        perc_Sig_bootstrap_df_2 =  pd.read_pickle(mean_pkl_sample2.strip("_means.pkl") + "_perc_Sig_bootstrap.pkl")
        
        
        for i in np.arange(len(perc_Sig_bootstrap_df_1.columns)):
        
            perc_Sig_bootstrap_df_1[perc_Sig_bootstrap_df_1.columns.values[i]] = perc_Sig_bootstrap_df_1[perc_Sig_bootstrap_df_1.columns.values[i]].str.get(0)
    
        for i in np.arange(len(perc_Sig_bootstrap_df_2.columns)):
            
            perc_Sig_bootstrap_df_2[perc_Sig_bootstrap_df_2.columns.values[i]] = perc_Sig_bootstrap_df_2[perc_Sig_bootstrap_df_2.columns.values[i]].str.get(0)
        
        perc_Sig_bootstrap_group1 = pd.DataFrame(perc_Sig_bootstrap_df_1.values, columns=perc_Sig_bootstrap_df_1.columns, index=['perc_Sig_bootstrap_group1'])
        perc_Sig_bootstrap_group2 = pd.DataFrame(perc_Sig_bootstrap_df_2.values, columns=perc_Sig_bootstrap_df_2.columns, index=['perc_Sig_bootstrap_group2'])
   
        
        if os.path.isfile(mean_pkl_sample_all):
            
            perc_Sig_bootstrap_df_all =  pd.read_pickle(mean_pkl_sample_all.strip("_means.pkl") + "_perc_Sig_bootstrap.pkl")
            
            
            for i in np.arange(len(perc_Sig_bootstrap_df_all.columns)):
            
                perc_Sig_bootstrap_df_all[perc_Sig_bootstrap_df_all.columns.values[i]] = perc_Sig_bootstrap_df_all[perc_Sig_bootstrap_df_all.columns.values[i]].str.get(0)
        
            
            
            perc_Sig_bootstrap_group_all = pd.DataFrame(perc_Sig_bootstrap_df_all.values, columns=perc_Sig_bootstrap_df_all.columns, index=['perc_Sig_bootstrap_group_all'])
        else:
            perc_Sig_bootstrap_group_all = pd.DataFrame([], index=['perc_Sig_bootstrap_group_all'])
    else:
        perc_Sig_bootstrap_group_all = pd.DataFrame([], index=['perc_Sig_bootstrap_group_all'])
        perc_Sig_bootstrap_group1 = pd.DataFrame([], index=['perc_Sig_bootstrap_group1'])
        perc_Sig_bootstrap_group2 = pd.DataFrame([], index=['perc_Sig_bootstrap_group2'])
        
        

        
        
    if T_max_full_path == '':
        T_max_full_path = os.path.join(os.path.dirname(mean_pkl_sample1), 'ROI_T_Max_table.pkl')
        Cohen_d_full_path = os.path.join(os.path.dirname(mean_pkl_sample1), 'ROI_Cohen_d_table.pkl')
        r_efect_size_full_path = os.path.join(os.path.dirname(mean_pkl_sample1), 'ROI_r_efect_size_table.pkl')
        
        
    if os.path.isfile(T_max_full_path):
        
        T_max_table = pd.read_pickle(T_max_full_path)
    
        T_max_table.index.values[0] = 'ROI_discovery_Tmax'
    else:
        T_max_table = pd.DataFrame([], index = ['ROI_discovery_Tmax'])
        
        
    if os.path.isfile(Cohen_d_full_path):
        
        Cohen_d_table = pd.read_pickle(Cohen_d_full_path)
    
        Cohen_d_table.index.values[0] = 'ROI_discovery_Cohen_d'
    else:
        Cohen_d_table = pd.DataFrame([], index = ['ROI_discovery_Cohen_d'])
    if os.path.isfile(r_efect_size_full_path):
        
        r_efect_size_table = pd.read_pickle(r_efect_size_full_path)
    
        r_efect_size_table.index.values[0] = 'ROI_discovery_r_efect_size'
    else:
        r_efect_size_table = pd.DataFrame([], index = ['ROI_discovery_r_efect_size'])
    
        
    #Replicability_table = pd.concat([effect_group1, effect_group2, multiplication_df, multiplication_sign_df, T_max_table, Cohen_d_table, r_efect_size_table], axis = 0)
    Replicability_table = pd.concat([effect_group1, P_Val_group1, CI_group1, boot_sign_group1, perc_Sig_bootstrap_group1,\
                                     effect_group2, P_Val_group2, CI_group2, boot_sign_group2,perc_Sig_bootstrap_group2,\
                                     effect_group_all, P_Val_group_all, CI_group_all,boot_sign_all,divide_df,perc_Sig_bootstrap_group_all,\
                                     multiplication_sign_df,T_max_table, Cohen_d_table, r_efect_size_table], axis = 0)
    
    replicability_folder = os.path.join(os.path.dirname(T_max_full_path), 'replicability_folder') 
    try:
        os.makedirs(replicability_folder)
    except OSError:
        if not os.path.isdir(replicability_folder):
            raise
    Replicability_table_Full_path = os.path.join(replicability_folder, 'Replicability_table.pkl')
    if len(which_split) == 0:
        Replicability_table = Replicability_table.T

        Replicability_table.to_pickle(Replicability_table_Full_path)
    else:
        Replicability_table = Replicability_table.add_prefix(which_split + '_')
        Replicability_table = Replicability_table.T
        Replicability_table.to_pickle(Replicability_table_Full_path)
    return Replicability_table_Full_path

def merge_replicability_of_correlations_for_all_splits(List_of_Replicability_table_Full_path): 
    
    Test = pd.DataFrame()
    for i in List_of_Replicability_table_Full_path:
        Replicability_table = pd.read_pickle(i)
        Test= Test.append(Replicability_table)
    
    
    return Test# Make it later if needed 
# Otherwise one can for the moment use the following loop, instead of "merge_replicability_of_correlations_for_all_splits" function
#Test = pd.DataFrame()
#for i in np.arange(n_split):
#    replicability_folder_i = os.path.join(XX + str(i), ....)
#    Replicability_table = pd.read_pickle(replicability_folder_i)
#    Test= Test.append(Replicability_table)
# Now Test is a dataframe that one can group based on the 'mean_Mul_Sign' column and compare the 'ROI_discovery_Tmax'. 
    

#
    
def function_create_and_save_split_stability_on_ROI(ROI_lists_file):
    #run_masch = '/data/'
    ##run_env = "/media/sf_Volumes/"
    #Base_dir = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/Test_Results_no_outlier_multiple_splits"
    #
    #ROI_lists_file = os.path.join(Base_dir, 'all_ROI_lists_path_cog_tests.txt') ## This is a file, in each line has the ROI_list_full_path for different tests, ...
    with open(ROI_lists_file) as L:
        lines = L.read().splitlines()
    
    Stability_ROI_full_path_file = open(os.path.join(os.path.dirname(ROI_lists_file), 'List_of_ROI_with_split_stability_full_path.txt'), 'w+')
     
    for ROI_list_full_path in lines:
        D=os.path.dirname(ROI_list_full_path).split('/')
        Test_base_DIR = os.path.join('/'+os.path.join(*D[:-7]))
        Replicability_table_Full_path = os.path.join(Test_base_DIR,"Secondary_CSVs_for_correlations/replicability_folder/Replicability_table.pkl")
        if os.path.isfile(Replicability_table_Full_path):    
            Replicability_table = pd.read_pickle(Replicability_table_Full_path)
            where_to_save_replicability_ROI = os.path.dirname(ROI_list_full_path)
            for roi in Replicability_table.index.values:
                roi_name = '_'.join(roi.split('_')[2::])
                ROI_image_full_path = os.path.join(where_to_save_replicability_ROI, roi_name + '.nii.gz')
                if os.path.isfile(ROI_image_full_path):
                    print(ROI_image_full_path)
                    stable_ROI_full_path = multiply_image_by_constant(ROI_image_full_path, constant = Replicability_table.loc[roi]['mean_Mul_Sign'], file_suffix = 'stability')
                    Stability_ROI_full_path_file.write(stable_ROI_full_path +'\n')
    
    Stability_ROI_full_path_file.close()
    
    return Stability_ROI_full_path_file
    
    

   










