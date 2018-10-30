#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:09:21 2018

@author: brain
"""
"""

This script basically is the last script that generates output figure and summary information about replicability of exploratory and confirmatory analysis-

In parts it also runs some R-codes (i.e. For calculatig the Bayes factors, using the code from ""http://www.josineverhagen.com/?page_id=76" and also alculating the replication power using "pwr" function)



"""



import subprocess
import seaborn as sns
import matplotlib
matplotlib.pyplot.switch_backend('agg')
from matplotlib.legend import Legend

import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from outcome_generation_for_multiple_splits import Create_Path_of_list_of_single_3D_files,\
                                                    create_frequency_map_from_many_binary_images,\
                                                    define_replicability_of_correlation_in_2_half,\
                                                    function_create_and_save_split_stability_on_ROI,\
                                                    merge_replicability_of_correlations_for_all_splits
                                                      

#run_masch = '/media/sf_Volumes/'
run_masch = '/data/'
Main_Sample_info_table_CSV_full_path = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", 'Python_processed_original_assessment_data_csv/Table_for_scripts/Outliers_removed_eNKI_MRI_cog_anxiety_Learning_roun1.csv')
subsampling_scripts_base_dir = os.path.join(run_masch + "BnB2/USER/Shahrzad/eNKI_modular", "scripts/25_11_2017")
original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/ADNI/analysis/Immediate_recall_dummy_diagn_dummy_site/"
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


ROIS_stats = {'ROIS_stats_type': 'tfce_corrp','thresh_corrp': 0.05,'T_Threshold': 0.00001, 'min_cluster_size_corr': 100, \
              'thresh_uncorrp' : 0.005, 'min_cluster_size_uncorr' : 100, 'test_side' : 2, \
              'Order_ROIs' : 'extent', 'Limit_num_ROIS' : np.inf, 'binned_ROIS' : True} # Or: 'tfce_corrp', or 'uncorr'

Split_settings = {'Age_step_size': 10, 'gender_selection' : None, 'n_split' : 100, 'Sex_col_name': 'sex_mr', 'Age_col_name' : 'Age_current'}
#            
#test_sample_size= Split_settings['test_sample_size']
min_ROI_size = ROIS_stats['min_cluster_size_corr']

Confounders_names =['Age_current', 'Sex', 'EDU_years']
Tests_list_file_full = os.path.join(original_base_folder, "Cog_list.txt")
try:
    with open(Tests_list_file_full) as L:
        Tests = L.read().splitlines()
except:
    pass
#


#### Section1    Perform GLM and save ROIs   
#cmd_c = 'python ' + subsampling_scripts_base_dir + '/01_GLM_on_many_splits_SLURM_Version.py'
#subprocess.call(cmd_c, shell=True)
### Create binary ROIS: 

#cmd_c = 'python ' + subsampling_scripts_base_dir + '/02_ROI_create_GLM_on_many_splits_SLURM_Version.py'
#subprocess.call(cmd_c, shell=True)
 

### Section2    Vol_extraction ####
#cmd_c = 'python ' + subsampling_scripts_base_dir + '/03_ROI_Vol_extraction_With_SLURM_output.py'
#subprocess.call(cmd_c, shell=True)

####
####Section4     Caculate the correlations (partial) with mGMV in ROIS:
#cmd_c = 'python ' + subsampling_scripts_base_dir + '/functional_corr_generation_multiple_splits.py'
#subprocess.call(cmd_c, shell=True)



#Base_dir = os.path.join(original_base_folder, test_variable_name + '_' + str(test_sample_size) + 'test_sample_' + str(n_perm)+ '_perm_' + str(min_ROI_size) + '_voxelminClust')
#Working_dir = os.path.join(Base_dir, test_variable_name +'_partial_corr'+ run)
n_split = Split_settings['n_split']
#%% begin of temporary comment
## I have only commented the Section 3 & 5 for now: as I have already run it.....
## Section3    Sig_ROI frequency maps:
        
locator_of_cog_specific_Pos_file = os.path.join(original_base_folder, 'locator_of_cog_specific_positive_ROIS.txt')
with open(locator_of_cog_specific_Pos_file) as f:
     pos_lines = f.read().splitlines()
for i in pos_lines:
    LIST_of_ROI_PATHS = Create_Path_of_list_of_single_3D_files(i)
    save_dir= os.path.join(os.path.dirname(i), "ROI_frequency_maps")
    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    if len(LIST_of_ROI_PATHS)>0:
        file_name = os.path.basename(os.path.dirname(i)).split("_Run")[0]
        file_to_save = create_frequency_map_from_many_binary_images(Path_of_list_of_single_3D_files = LIST_of_ROI_PATHS,\
                                                                    probability_map_file_name = 'Prob_Pos_Sig_ROIs_' + file_name,\
                                                                    save_dir = save_dir, n_total = n_split)
        
            
locator_of_cog_specific_Neg_file = os.path.join(original_base_folder, 'locator_of_cog_specific_negative_ROIS.txt')
with open(locator_of_cog_specific_Neg_file) as L:
     neg_lines = L.read().splitlines()
for j in neg_lines:
    LIST_of_ROI_PATHS = Create_Path_of_list_of_single_3D_files(j)
    save_dir= os.path.join(os.path.dirname(j), "ROI_frequency_maps")
    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    if len(LIST_of_ROI_PATHS)>0:
        file_name = os.path.basename(os.path.dirname(j)).split("_Run")[0]
        file_to_save = create_frequency_map_from_many_binary_images(Path_of_list_of_single_3D_files = LIST_of_ROI_PATHS,\
                                                                    probability_map_file_name = 'Prob_Neg_Sig_ROIs_' + file_name,\
                                                                    save_dir = save_dir, n_total = n_split)
        
    

## . 
##            
##        ### Section5    replicability
#            
#        
packed_means_location = os.path.join(original_base_folder, 'packed_means_list_for_stability_cog_tests.txt')
        
with open(packed_means_location) as K:
    Loc_split_spec_means = K.read().splitlines()
List_of_files_for_replicabilitypaths = []    
for i in Loc_split_spec_means:
    
    with open(i) as L:
        split_spec_means = L.read().splitlines()
    
    
    Replicability_table_Full_path = define_replicability_of_correlation_in_2_half(mean_pkl_sample1 = split_spec_means[1],mean_pkl_sample2= split_spec_means[2],\
                                                                                  mean_pkl_sample_all = split_spec_means[0],\
                                                                                  which_split = os.path.basename(os.path.dirname(os.path.dirname(split_spec_means[1]))))
    ### This is very specific to the folder structure with Splits:
    File_to_save_Replicability_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(split_spec_means[1]))), 'Test_specific_all_splits_replicability_path.txt')
    List_of_files_for_replicabilitypaths.append(File_to_save_Replicability_location)
    with open(File_to_save_Replicability_location, "a+") as myfile:
        myfile.write(Replicability_table_Full_path +'\n')

#        ### Section6    merging the replicability tables: 
#                
unique_replicability_file_locations = list(set(List_of_files_for_replicabilitypaths))
locator_of_merged_replicability_tables = open(os.path.join(original_base_folder, 'locator_of_merged_replicability_tables.txt'), 'a+')
for i in unique_replicability_file_locations:
    with open(i) as f:
        Lines = f.read().splitlines()
        
    merged_rep = merge_replicability_of_correlations_for_all_splits(Lines)
    merged_rep_folder = os.path.join(os.path.dirname(i), 'Overall_split_Replicability_folder')
    
    try:
        os.makedirs(merged_rep_folder)
    except OSError:
        if not os.path.isdir(merged_rep_folder):
            raise
            
    merged_rep_full_file = os.path.join(merged_rep_folder, 'merged_replicability_table.pkl')
    merged_rep.to_pickle(merged_rep_full_file)
    locator_of_merged_replicability_tables.write(merged_rep_full_file +'\n')
locator_of_merged_replicability_tables.close() # This file could be later loaded and read line by line, to do the stats
#%%        End of ""temporary comment        
        ### Now after going to R, and creating a power column for each split_ROI:
        # In R: I have the following code:
# Now I read in each replicability Text file:
with open(os.path.join(original_base_folder, 'locator_of_merged_replicability_tables.txt')) as f:
    replicability_merged_foleder = f.read().splitlines()            
for i in np.arange(len(replicability_merged_foleder)):
    merged_rep_folder = os.path.dirname(replicability_merged_foleder[i])
    merged_rep_full_file = replicability_merged_foleder[i]
    # here I decide what my sample size of replication and discovery was:
    test_name = os.path.basename(os.path.dirname(merged_rep_folder)).split("_partial_corr_Run_0")[0]
    table_temp = pd.read_csv(os.path.join(os.path.dirname(merged_rep_folder), "Split_0/sample_CSV/grouped_main_sample_" + test_name + ".csv"))
    Original_sample_size = str(len(table_temp[table_temp.Which_sample == 1]))
    replication_sample_size = str(len(table_temp[table_temp.Which_sample == 2]))
    #import subprocess
#%% End oif Again temporary commenting     
    # Define command and arguments
    command = 'Rscript'
    path2script = os.path.join(subsampling_scripts_base_dir, "power_calculation_in_R.R")
    
    # Variable number of args in a list
    # I need to change the pkl file to csv:
    merged_rep_full_file_CSV = os.path.splitext(merged_rep_full_file)[0] +'.csv'
    merged_rep = pd.read_pickle(merged_rep_full_file)
    merged_rep.to_csv(merged_rep_full_file_CSV)
    P_value_threshold = str(0.05)
    
    Out_put_csv_name= os.path.join(merged_rep_folder, 'merged_replicability_table_With_power2.csv')
    
    # Build subprocess command
    cmd = command + ' '+ '--vanilla' + ' '+ path2script + ' '+ merged_rep_full_file_CSV + ' '+ replication_sample_size+ ' '+ P_value_threshold + ' '+ Out_put_csv_name+ ' '+ Original_sample_size+ ' '+ subsampling_scripts_base_dir 
    
    subprocess.call(cmd, shell=True)
#%% End oif Again temporary commenting 
    
    
    # Now i need to chnage back the csv to 
    Out_put_csv_name= os.path.join(merged_rep_folder, 'merged_replicability_table_With_power2.csv')
    merged_rep = pd.read_csv(Out_put_csv_name, index_col="X")
    merged_rep["Power"] = merged_rep["Power"].round(decimals=2)
    
    ##########Now create groups of power and continue with plots----- 
    
    merged_rep['new_power_group'] = pd.cut(merged_rep.Power, bins=4, labels=False)
    cut = pd.cut(merged_rep.Power, bins=4)
    grouped_power_list= merged_rep.groupby(cut)['Power'].median().tolist()
    
    
    merged_rep['Mul_sign_group'] = merged_rep['mean_Mul_Sign'].map({1: "Replicated", -1: "Not_replicated"})
    
    merged_rep = merged_rep[~merged_rep.index.duplicated(keep='first')]
    
   
    
    ### Now one additional plot:
    # Here I want to plo
    index=["Moderate-Strong H0", "Anecdotal_H0", "Anecdotal_H1","Moderate-Strong H1"]
    Bayes_factor_df = pd.DataFrame(index=index, columns=["N_cases", "Decision"])
    
    Bayes_factor_df.loc["Moderate-Strong H0"].N_cases =len(merged_rep[(merged_rep['one_sided_BF01'] > 3) | (merged_rep['one_sided_BF01']==3)])
    Bayes_factor_df.loc["Moderate-Strong H0"].Decision = "n_B01>3"
    Bayes_factor_df.loc["Anecdotal_H0"].N_cases = len(merged_rep[(merged_rep['one_sided_BF01'] < 3) & (merged_rep['one_sided_BF01']>1)])
    Bayes_factor_df.loc["Anecdotal_H0"].Decision = "1<n_B01<3"
    Bayes_factor_df.loc["Anecdotal_H1"].N_cases = len(merged_rep[(merged_rep['one_sided_BF10'] < 3) & (merged_rep['one_sided_BF10']>1)])
    Bayes_factor_df.loc["Anecdotal_H1"].Decision = "1<n_B10<3"
    Bayes_factor_df.loc["Moderate-Strong H1"].N_cases =len(merged_rep[(merged_rep['one_sided_BF10'] > 3) | (merged_rep['one_sided_BF10']==3)])
    Bayes_factor_df.loc["Moderate-Strong H1"].Decision = "n_B10>3"


    plt.figure()
    f, axes = plt.subplots(1,1, figsize=(6,6))
    A_colors=["xkcd:deep orange", "xkcd:pumpkin orange","xkcd:robin's egg blue", "xkcd:medium blue"]

    plot0 = Bayes_factor_df.plot(kind='pie',ax= axes, y = 'N_cases',autopct='%.2f%%', pctdistance=1.2, colors=A_colors, shadow=False, labels=None, legend = True, fontsize=10)
    plot0 = plot0.get_figure()
    plot0.savefig(os.path.join(merged_rep_folder, "pie_Bayes_factors.png"))
    
    ## This file can be used to do stats on : example, one can see if there are T_max differences between replicated and not replicated splits.
    # Example1: 
    #merged_rep.plot.scatter(x = 'effect_size_group_all', y = 'effect_size_group2', c='mean_Mul_Sign', colormap= 'Spectral_r', yticks =np.arange(-0.5,0.5, 0.1), xticks =np.arange(0,0.5, 0.1))
    #Example2: Not anymore interesting 
    plt.figure()
    g = sns.lmplot(x="effect_size_group1", y="effect_size_group2", hue="Mul_sign_group",
                   truncate=True, size=5, data=merged_rep)
    
    g.savefig(os.path.join(merged_rep_folder, "effect_size_of_replicatedvs_not_replicated.png"))
    
    ## Example 4: Not anymore interesting 
    #g = sns.jointplot("ROI_discovery_Tmax", "effect_size_group_all",kind='reg', size=5, data=merged_rep)
    
    
    #Example3    I prefere this version:  *****BEST PLOT for the MOMENT*****
    g = sns.JointGrid("effect_size_group1", "effect_size_group2", xlim=(min(merged_rep.effect_size_group1)-0.05, max(merged_rep.effect_size_group1)+0.05), ylim=(min(merged_rep.effect_size_group2)-0.05, max(merged_rep.effect_size_group2)+0.05), data=merged_rep)
    #g = sns.JointGrid("effect_size_group1", "effect_size_group2", data=merged_rep)
    #color=["xkcd:pumpkin orange", "xkcd:medium blue"]
    color=["xkcd:medium blue","xkcd:pumpkin orange"]
    
    i = 0
    for a, not_replic in merged_rep.groupby("Mul_sign_group"):
        
        
        #sns.kdeplot(not_replic["effect_size_group1"], ax=g.ax_marg_x, legend=False)
        #sns.kdeplot(not_replic["effect_size_group2"], ax=g.ax_marg_y, vertical=True, legend=False)
        #print(len(not_replic))
        if len(not_replic)>1:
            
            sns.distplot(not_replic["effect_size_group1"], ax=g.ax_marg_x, axlabel= False, rug=True, hist=False, color=color[i])
            sns.distplot(not_replic["effect_size_group2"], ax=g.ax_marg_y, vertical=True, axlabel= False, rug=True, hist=False, color=color[i])
        
        #sns.distplot(not_replic["effect_size_group2"], ax=g.ax_marg_y, vertical=True, axlabel= False, norm_hist = False, hist=True, kde = False)
        #sns.distplot(not_replic["effect_size_group2"], ax=g.ax_marg_y, vertical=True, axlabel= False)
        
        # Here I change the size of the points based on the power in the original study-
        sns.regplot(not_replic["effect_size_group1"], not_replic["effect_size_group2"],ci=None, ax=g.ax_joint, color=color[i],scatter=True, line_kws = {"lw" : 0.5}, scatter_kws={"alpha":0.6, "s":1*((not_replic["new_power_group"]+2)**2)})
        #sns.regplot(not_replic["effect_size_group1"], not_replic["effect_size_group2"],ci=None, ax=g.ax_joint, color=color[i])
        
        #, palette=dict(Not_replicated="xkcd:medium blue", Replicated="xkcd:pumpkin orange")
        
    
        i = i+1
        
    
    #g.ax_joint.legend(merged_rep["Mul_sign_group"].unique())
    g.ax_joint.legend(['Not_replicated','Replicated'])
    g.ax_joint.axhline(0, color='red', linestyle = 'dashed', alpha= 0.2)
    
    # To now create a new legend for the size of the markers (aka: power) 
    size_marker = merged_rep.new_power_group.unique()
    size_marker.sort()
    grouped_power_list.sort()
    pws = grouped_power_list
    for pw in size_marker:
        g.ax_joint.scatter([], [], s=((pw+2)**2)*1, c="k",label=str(pws[pw]))
    
    
    
    h, l = g.ax_joint.get_legend_handles_labels()
    #leg = Legend(g.ax_joint, h[:], l[:], loc = 'lower left', labelspacing=1.2, title="power") # or add the following: , borderpad=1, frameon=False, framealpha=0.6, edgecolor="k", facecolor="w"
    leg = Legend(g.ax_joint, h[:], l[:], loc = 'lower left', labelspacing=1, title="Power", borderpad=0.2, frameon=False, framealpha=0.5, edgecolor="k", facecolor="w")
                
                
    g.ax_joint.add_artist(leg)
    g.fig.suptitle(test_name + '_' + str(replication_sample_size) + '_ntest',y = 1.01)     
    g.savefig(os.path.join(merged_rep_folder, "effect_size_of_replicatedvs_not_replicated_joint_plot2.pdf"))
    
    ## The following also is nice for plotting the p_values:(it tried to create a one_sided p-value)
    plt.figure()
    merged_rep["one_sided_P_Val_group1"] = merged_rep['P_Val_group1']/2
    merged_rep["one_sided_P_Val_group2"] = merged_rep['P_Val_group2']/2
    df = pd.melt(merged_rep, value_vars=['one_sided_P_Val_group1', 'one_sided_P_Val_group2'], id_vars='Mul_sign_group')
    
    df.columns = ['replication_Status', 'Group', 'P_values']
    df['Group'] = df['Group'].map({"one_sided_P_Val_group1": "Discovery_group", "one_sided_P_Val_group2": "replication_group"})
    #colors=[ "xkcd:pumpkin orange", "xkcd:medium blue"]
    colors=[ "xkcd:medium blue","xkcd:pumpkin orange"]
    if len(df['replication_Status'].unique())>1:
        split_hue = True
    else:
        split_hue = False
    K = sns.violinplot(x='Group', y='P_values', hue='replication_Status',  palette=colors, cut = 0,inner="quartiles",scale= "count",scale_hue =False, split=split_hue, data=df)
    K.axes.axhline(0.05, color='red', linestyle = 'dashed', alpha= 0.5)
    plt.legend(loc='upper left')
    K.set_title(test_name + '_' + str(replication_sample_size) + '_nTest')
    
    fig = K.get_figure()
    #    plt.show()
    fig.savefig(os.path.join(merged_rep_folder, "P_value_of_replicatedvs_not_replicated_viloin_plot2.pdf"))

    
    
    
#    #####This is also nice & informative
#    
#    plt.figure()
#    g = sns.lmplot(x="P_Val_group1", y="P_Val_group2",fit_reg = False, hue="Mul_sign_group",scatter_kws={"alpha":0.6, "s": ((merged_rep["new_power_group"]+1)**3)*1.5}, data=merged_rep)
#    g.axes[0,0].axhline(0.05, color='red', linestyle = 'dashed', alpha= 0.5)
#    size_marker = merged_rep.new_power_group.unique()
#    size_marker.sort()
#    grouped_power_list.sort()
#    pws = grouped_power_list
#    for pw in size_marker:
#        g.axes[0,0].scatter([], [], s=((pw+1)**3)*1.5, c="k",label=str(pws[pw]))
#    
#    
#    
#    
#    h, l = g.axes[0,0].get_legend_handles_labels()
#    #leg = Legend(g.ax_joint, h[:], l[:], loc = 'lower left', labelspacing=1.2, title="power") # or add the following: , borderpad=1, frameon=False, framealpha=0.6, edgecolor="k", facecolor="w"
#    leg = Legend(g.axes[0,0], h[2:], l[2:], loc = 'lower right', labelspacing=1, title="Power", borderpad= 0.001, frameon=False, framealpha=0.5, edgecolor="k", facecolor="w")
#                
#    g.axes[0,0].add_artist(leg)
#    
#    g.axes[0,0].set_xlim(-0.001, 0.025)
##    plt.show()
#    
#    
#    g.savefig(os.path.join(merged_rep_folder, "P_Value_and_power_replicatedvs_not_replicated_joint_plot.png"))
    
    
    
    
    
    
    
    
    # Example:
    from scipy.stats import ttest_ind
    group1_air = merged_rep.where(merged_rep.mean_Mul_Sign == 1).dropna()['ROI_discovery_Tmax'] # OR "ROI_discovery_r_efect_size", "ROI_discovery_Cohen_d"
    group2_air = merged_rep.where(merged_rep.mean_Mul_Sign == -1).dropna()['ROI_discovery_Tmax']
    ttest_independent = ttest_ind(group1_air,group2_air)
#    %matplotlib inline
    
    
    plt.figure()
    plot_0= merged_rep.boxplot(by=['mean_Mul_Sign'], column=['ROI_discovery_Cohen_d'])
    fig_0 = plot_0.get_figure()
    fig_0.savefig(os.path.join(merged_rep_folder, "box_plot_replicability.png"))
#    plt.show()
    #or:
#    plt.figure()
#    plot_3 = sns.swarmplot(x="Mul_sign_group", y="ROI_discovery_Cohen_d", data=merged_rep)
#    fig_3 = plot_3.get_figure()
#    fig_3.savefig(os.path.join(merged_rep_folder, "scatter_replicability.png"))
#    
#    # or
#    plt.figure()
#    plot_4 =sns.barplot(x="Mul_sign_group", y="ROI_discovery_Cohen_d", data=merged_rep)
#    fig_4 = plot_4.get_figure()
#    fig_4.savefig(os.path.join(merged_rep_folder, "CI_bar_replicability.png"))
#    plt.show()
    #or the combnined form:
    plt.figure()
    T = sns.barplot(x="Mul_sign_group", y="ROI_discovery_Cohen_d", data=merged_rep, saturation=0.1);sns.swarmplot(x="Mul_sign_group", y="ROI_discovery_Cohen_d", data=merged_rep)
    fig_6 = T.get_figure()
    fig_6.savefig(os.path.join(merged_rep_folder, "bar_plus_scatter_replicability.png"))
#    plt.show()
    
    plt.figure()
    # The following shows how it will seem if I want to find percent of repeated over not repeated ROIs-
    # This value does not account for the number of 
    plot_1 = merged_rep['Mul_sign_group'].value_counts().plot(kind='bar', color= color)
    plot_1 = plot_1.get_figure()
    plot_1.savefig(os.path.join(merged_rep_folder, "bar_plot_replicability.png"))
#    plt.show()
#    plt.figure()
#    # or 
#    plot_5 =sns.countplot(x="Mul_sign_group", data=merged_rep, palette="Greens_d")
#    fig_5 = plot_5.get_figure()
#    fig_5.savefig(os.path.join(merged_rep_folder, "count_bars_replicability.png"))
    #plt.show()
    plt.figure()
    f, axes = plt.subplots(1,1, figsize=(6,6))
    
    A_colors=["xkcd:pumpkin orange", "xkcd:medium blue"]
    plot_2 = merged_rep['Mul_sign_group'].value_counts().plot(kind='pie', ax= axes, autopct='%.2f',colors=A_colors, labels=None,  title="ROI Replicability %", fontsize=10)
    axes.legend(loc=3, labels=merged_rep['Mul_sign_group'].value_counts().keys())
    plot_2 = plot_2.get_figure()
    plot_2.savefig(os.path.join(merged_rep_folder, "pie_chart_replicability.png"))
#   plt.show()
    
    plt.figure()
    ## Now Pie chart with both sign and P-value 
    merged_rep['replicated_P_value_based'] = 0
    #merged_rep.loc[merged_rep['P_Val_group2']<0.0500000000000000000001, 'replicated_P_value_based'] = 1
    # This is to use a one-sided p-val threshold.as if the p-vals were calculated one-sided (but they are actually calculated two-sided, hence the following...)
    merged_rep.loc[merged_rep['P_Val_group2']<0.100000000000000000001, 'replicated_P_value_based'] = 1

    merged_rep.loc[merged_rep['mean_Mul_Sign'] == -1, 'replicated_P_value_based'] = 0
    
    merged_rep['Bayes_Factor_decision'] = ''
    merged_rep.loc[(merged_rep['one_sided_BF01'] > 3) | (merged_rep['one_sided_BF01']==3), 'Bayes_Factor_decision'] = "Moderate-Strong H0"
    merged_rep.loc[(merged_rep['one_sided_BF01'] < 3) & (merged_rep['one_sided_BF01']>1), 'Bayes_Factor_decision'] = "Anecdotal_H0"
    merged_rep.loc[(merged_rep['one_sided_BF10'] < 3) & (merged_rep['one_sided_BF10']>1), 'Bayes_Factor_decision'] = "Anecdotal_H1"
    merged_rep.loc[(merged_rep['one_sided_BF10'] > 3) | (merged_rep['one_sided_BF10']==3), 'Bayes_Factor_decision'] = "Moderate-Strong H1"
    
    ##### Very interesting:
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    
    
    # Create colors
    K = merged_rep['Mul_sign_group'].value_counts() 
    
    merged_rep["replication_P_sig_label"] = merged_rep.replicated_P_value_based.map({1: "P_sig", 0:"P_not_sig"})
    
    
    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    A_colors=["xkcd:medium blue","xkcd:pumpkin orange"]
    K_ordered = [0,0]
    if len(K[K.index == 'Replicated'].values)>0:
        K_ordered[0] = K[K.index == 'Replicated'].values[0]
    
        
    if len(K[K.index == 'Not_replicated'].values)>0:
        K_ordered[1] = K[K.index == 'Not_replicated'].values[0]
    
        
            
    
    
    mypie,_, fractions= ax.pie(K_ordered, radius=1.1, labels=None, colors=A_colors,startangle=30, autopct=make_autopct(K_ordered), pctdistance=0.8)
    
    for text in fractions:
        text.set_rotation(75)
    
    
    plt.setp( mypie, width=0.33, edgecolor='white')
     
    # Second Ring (Inside)
    Sig = merged_rep['replication_P_sig_label'].value_counts()
    A1 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Replicated")])
    A2 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Replicated")])
    A3= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated")])
    A4= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated")])
    A_labels = ['replicated_P_sig', 'replicated_P_not_sig','not_replicated_P_sig', 'not_replicated_P_not_sig']
    #A_colors = [a(0.4), b(0.4), a(0.5), b(0.5)] 
    A_labels_short= ['sig', '~sig','sig', '~sig']
    A =  [A1, A2, A3, A4]
    modified_color = A_colors + A_colors
    
    for i in range(len(A)):
        if A[i] == 0:
            modified_color[i]='CUT_color'
            A_labels_short[i] = 'delete_label'
    
    
    color_for_donut = [ x for x in modified_color if 'CUT_color' not in x ]
    A_labels_short = [ x for x in A_labels_short if 'delete_label' not in x ]
    A = [x for x in A if x != 0]       
    mypie2, _, fraction_A = ax.pie(A, radius=1.3+0.25, labels=A_labels_short, labeldistance=0.6,rotatelabels = True, autopct= "%0.1f", pctdistance=0.83, colors=color_for_donut, startangle=30)
    
    #for text in fraction_A:
    #    text.set_rotation(75)
    #
    plt.setp(mypie2, width=0.35, edgecolor='white')
    plt.margins(0,0)    
    B11 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H0")])
    B12 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H0")])
    B13 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H1")])
    B14 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H1")])
    B21 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H0")])
    B22 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H0")])
    B23 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H1")])
    B24 = len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H1")])
    B31= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H0")])
    B32= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H0")])
    B33= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H1")])
    B34= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H1")])
    B41= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H0")])
    B42= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H0")])
    B43= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Anecdotal_H1")])
    B44= len(merged_rep[(merged_rep['replication_P_sig_label'] == "P_not_sig") & (merged_rep['Mul_sign_group'] == "Not_replicated") & (merged_rep['Bayes_Factor_decision']== "Moderate-Strong H1")])
    B_labels = ["Moderate-Strong H0", "Anecdotal_H0", "Anecdotal_H1", "Moderate-Strong H1", "Moderate-Strong H0", "Anecdotal_H0", "Anecdotal_H1", "Moderate-Strong H1", "Moderate-Strong H0", "Anecdotal_H0", "Anecdotal_H1", "Moderate-Strong H1", "Moderate-Strong H0", "Anecdotal_H0", "Anecdotal_H1", "Moderate-Strong H1"]
    
    B =  [B11, B12, B13, B14, B21, B22, B23, B24, B31, B32, B33, B34, B41, B42, B43, B44]
    B_colors=["xkcd:deep orange", "xkcd:pumpkin orange","xkcd:robin's egg blue", "xkcd:medium blue"]
    modified_color_B = B_colors + B_colors + B_colors + B_colors
    for i in range(len(B)):
        if B[i] == 0:
            modified_color_B[i]='CUT_color'
            B_labels[i] = "delete_label"
    
    
    color_for_donut_B = [ x for x in modified_color_B if 'CUT_color' not in x ]
    B = [x for x in B if x != 0]
    B_labels = [ x for x in B_labels if 'delete_label' not in x ]
    mypie3,_, fraction_B = ax.pie(B, radius=1.3+0.7, labels=B_labels, labeldistance=1.05, rotatelabels = True, autopct= "%0.1f", pctdistance=0.9, colors=color_for_donut_B, startangle=30)
    #
    
    #for text in fraction_B:
    #    text.set_rotation(75)
        
    plt.setp( mypie3, width=0.38, edgecolor='white')
    plt.margins(0,0)
    plt.suptitle(test_name + '_' + str(replication_sample_size) + '_ntest',y = 1.5)

    ax.legend(labels=['Replicated', 'Not_replicated'], loc='upper right', bbox_to_anchor=(1.5, 1.5))
    plt.savefig(os.path.join(merged_rep_folder, "pie_chart_replicability_withone_sided_P_vals.pdf"),bbox_inches="tight",pad_inches=0.5) 
        
    
    
    
        
## Section7    Spatial maps of the stability of the splits.  I am not really sure about this part: maybe not use for the moment
ROI_lists_file = os.path.join(original_base_folder, 'all_ROI_lists_path_cog_tests.txt')    
Stability_ROI_full_path_file = function_create_and_save_split_stability_on_ROI(ROI_lists_file)
    
#        ##########
#        # Also an example of how to compare percentage of overlaping ROI in different train sample sizes:
#        #!/usr/bin/env python3
#        # -*- coding: utf-8 -*-
#        """
#        Created on Thu Jan 25 13:35:07 2018
#        
#        @author: skharabian
#        """
#I have commented this part too, as it needs manual input of images:      """""Density Plots"""""""
from nilearn.input_data import NiftiMasker
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# How to craete plots of distribution of ROI overlap 
Test_sample_p = [0.7]#[0.3, 0.5, 0.7]
Sides = ["Pos","Neg"]
Sides = ["Neg"]

run_masch = '/data/'
#original_base_folder = run_masch + "BnB2/USER/Shahrzad/eNKI_modular/Handedness_Results_No_outlier_splits_1000_perm"
original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/ADNI/analysis/Immediate_recall_dummy_diagn_dummy_site"
#original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/Age_Results_No_outlier_splits_1000_perm"
original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/IQ_Results_No_outlier_splits_1000_perm"

Tests_list_file_full = os.path.join(original_base_folder, "Cog_list.txt")
try:
    with open(Tests_list_file_full) as L:
        Tests = L.read().splitlines()
except:
    pass
#Tests = ['DKEFSCWI_09', 'DKEFS_WORD_51']
#Tests = ['RAVLT_immediate_bl']
Tests = ['BMI','WASI_Total_IQ', 'WASI_Perceptual', 'WASI_Vocabulary']
Tests = ['BMI']

#Tests = ['Age_current']
for test_variable_name in Tests:
    for side in Sides:
        
        for i in range(len(Test_sample_p)):
            masker = NiftiMasker()
            try:
                
#                locals()["data"+str(i+1)] = masker.fit_transform(os.path.join(original_base_folder, test_variable_name+ "_" + str(Test_sample_p[i]) +"test_sample_1000_perm_100_voxelminClust_SITE_added_with_disease_dummy/" + test_variable_name +"_partial_corr_Run_0/ROI_frequency_maps/Prob_" + side + "_Sig_ROIs_" + test_variable_name+ "_partial_corr.nii"))
#                table = pd.read_csv(os.path.join(original_base_folder,test_variable_name+ "_" + str(Test_sample_p[i]) +"test_sample_1000_perm_100_voxelminClust_SITE_added_with_disease_dummy/"+ test_variable_name+ "_partial_corr_Run_0/Split_0/sample_CSV/grouped_main_sample_"+ test_variable_name+ ".csv"))
                locals()["data"+str(i+1)] = masker.fit_transform(os.path.join(original_base_folder, test_variable_name+ "_" + str(Test_sample_p[i]) +"test_sample_1000_perm_100_voxelminClust/" + test_variable_name +"_partial_corr_Run_0/ROI_frequency_maps/Prob_" + side + "_Sig_ROIs_" + test_variable_name+ "_partial_corr.nii"))
                table = pd.read_csv(os.path.join(original_base_folder,test_variable_name+ "_" + str(Test_sample_p[i]) +"test_sample_1000_perm_100_voxelminClust/"+ test_variable_name+ "_partial_corr_Run_0/Split_0/sample_CSV/grouped_main_sample_"+ test_variable_name+ ".csv"))
                
                count_discovery = len(table[table.Which_sample == 1])
                locals()["df"+str(i+1)] = pd.DataFrame(data=locals()["data"+str(i+1)].T, columns=['percent overlap'])
                locals()["df"+str(i+1)]['Discovery sample'] =  str(count_discovery) +"_" + side#np.ones(data1.T.shape)
            except:
                locals()["df"+str(i+1)]= pd.DataFrame(data=[], columns=['percent overlap'])
                locals()["df"+str(i+1)]['Discovery sample'] =  str(count_discovery) +"_" + side#np.ones(data1.T.shape)
        #Df= pd.concat([df1, df2, df3], axis = 0)
        Df=df1
        if len(Df)>0:
            plt.figure()

            g = sns.violinplot(x= 'Discovery sample', y='percent overlap', data=Df, size=16, scale="count", inner="box",  palette="Set3", cut = 0)
            
            g.set_title(test_variable_name)
            fig = g.get_figure()
            merged_rep_folder = os.path.join(original_base_folder, test_variable_name+ "_" + str(Test_sample_p[i]) +"test_sample_1000_perm_100_voxelminClust_SITE_added_with_disease_dummy/" + test_variable_name +"_partial_corr_Run_0/Overall_split_Replicability_folder")
            #fig.savefig(os.path.join(merged_rep_folder, "n_subjbaased_"+ side+"_X.pdf"))


# How to view them using command in terminal:
    
# eog /data/BnB2/USER/Shahrzad/eNKI_modular/Test_Results_No_outlier_splits_1000_perm/*/*/Overall_split_Replicability_folder_Old/n_subjbaased_*_X.png&

#####

        
#            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#Created on Mon Feb 26 10:57:40 2018
#
#@author: skharabian
#"""
# To test if there is a significance difference between test variable (bad and good performers in the two splits).

import os
import numpy as np
import pandas as pd
import scipy
run_masch = '/data/'
original_base_folder = run_masch + "BnB_USER/Shahrzad/eNKI_modular/ADNI/analysis/Immediate_recall_dummy_diagn_dummy_site"
Test_sample_p = [0.5]#[0.3, 0.5, 0.7]
n_split =100    
Tests_list_file_full = os.path.join(original_base_folder, "Cog_list.txt")
try:
    with open(Tests_list_file_full) as L:
        Tests = L.read().splitlines()
except:
    pass
#Tests = ['BMI','WASI_Total_IQ', 'WASI_Perceptual', 'WASI_Vocabulary']
Tests = ['RAVLT_immediate_bl']

min_ROI_size=100
n_perm =1000
p_vals = pd.DataFrame()
for test_variable_name in Tests:
    
    for percent_test_size in Test_sample_p:
        
        test_sample_size=percent_test_size
        

        p_val= pd.DataFrame()
        Base_dir = os.path.join(original_base_folder, test_variable_name + '_' + str(test_sample_size) + 'test_sample_' + str(n_perm)+ '_perm_' + str(min_ROI_size) + '_voxelminClust_SITE_added_with_disease_dummy')
            
        for i in range(n_split):
            # a : create a caterorical variable from a contineous 
            table_path = os.path.join(Base_dir,test_variable_name + "_partial_corr_Run_0/Split_" + str(i) + "/sample_CSV/grouped_main_sample_" + test_variable_name +".csv")
            print(table_path)
            table = pd.read_csv(table_path)
            group_var = test_variable_name+'_group'
            table[group_var] = pd.cut(table[test_variable_name], bins = 2, labels=False)
            
            
            CT= pd.crosstab(table['Which_sample'], table[group_var])
            print(CT)
            cs1 = scipy.stats.chi2_contingency(CT)
            print(cs1[1])
            p_val= p_val.append({test_variable_name + str(test_sample_size): cs1[1]}, ignore_index=True)
        p_vals = pd.concat([p_vals, p_val], axis=1)
        
DF_FULL_PATH= os.path.join(original_base_folder, "Significance_Difference_between_test_vals_splits.pkl")
p_vals.to_pickle(DF_FULL_PATH)