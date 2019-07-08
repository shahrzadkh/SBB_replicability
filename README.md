# SBB_replicability
_These are codes for running analyses of the following published work:_


The order of running the codes:


# A: 01_GLM_on_many_splits_SLURM_Version.py
  Depending on the Sulrm flag, it either runs the randomise itself or generates Slurm submission script ready with randomise 
  command in it:
  - If Slurm is used, then the generated submission file needs to be submitted to Slurm by: sbatch ${submission file}
  
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
      Now For each behavioral score, for n (#splits) GLMs are fitted using randomise
      
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# B:  02_ROI_create_GLM_on_many_splits_SLURM_Version.py
  identifies significant clusters and saves them as binary ROIS. 
  
# C:  03_ROI_Vol_extraction_With_SLURM_output.py
  This extracts mean/median GMV per ROI from every subject.

# D:  functional_corr_generation_multiple_splits.py or functional_corr_generation_multiple_splits_partII.py
  generates correlation (bivariate/partial, spearman/pearson) between each cluster's mean GMV and the respective behavioral score. 
  
# E:  Many_split_running_code.py 
  Is the last code, which generates the figures and replicability outcomes of the exploratory and confirmatory analysis. 
  It used partly R-codes, for Bayes factors (http://www.josineverhagen.com/?page_id=76) and power estimation (pwr). 
  
  *** If Not using Slurm, one can modify the Many_split_running_code.py to run the analysis entirely. #TO DO
   

# Contact:
shahrzadkhm@gmail.com
