# EEG_XAI_Depression
Implementation of XAI methods in Depression Classification based on EEG and DL 

This resource contains:
1. Generate a synthetic EEG dataset using SEREEGA tools on EEGLab. 

   - Download SEREGA and the documentation at: https://github.com/lrkrol/SEREEGA.
   
   - The generated dataset code follows the design by Ravindran and Contreras-Vidal (2023), https://doi.org/10.1038/s41598-023-43871-8, saved in generate_db folder.

3. Load, train and evaluate synthetic dataset (sereega_db_eval folder)

   Steps to follow :

   - Load synthetic datasets and do pre-processing (load_sereega_db.py)
     
   - Train with DL models (train_sereega_db.py)
     
   - Implement XAI methods on the model and do sensitivity and robustness tests (sensitivity_eval.py and robustness_eval.py)
   
5. Load, train and evaluate real EEG datasets in Depression analysis

   - Get your real EEG public datasets :

     MODMA (https://modma.lzu.edu.cn/data/application/)

     PRED+CT (http://predict.cs.unm.edu/downloads.php)

     MPHC (https://figshare.com/articles/dataset/EEG_Data_New/4244171)

   - Train the dataset using DL models (real_db_eval/train_eeg_hold_out.py)
     
   - Implement XAI methods on the model to get feature importance and topographic maps (real_db_eval/apply_xai_abs.py)

This XAI implementation uses Captum (https://captum.ai/) and Pytorch-GradCam packages (https://github.com/jacobgil/pytorch-grad-cam). 
