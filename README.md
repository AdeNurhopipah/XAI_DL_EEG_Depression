# EEG_XAI_Depression
Implementation of XAI methods in Depression Classification based on EEG and DL 

This resource contains:
1. Generate a synthetic EEG dataset using SEREEGA tools on EEGLab. 

   - Download SEREGA and the documentation at: https://github.com/lrkrol/SEREEGA.
   
   - The generated dataset code follows the design by Ravindran and Contreras-Vidal (2023), https://doi.org/10.1038/s41598-023-43871-8, saved in generate_db folder.

3. Load, train and evaluate synthetic dataset.

   Steps to follow :

   - Load synthetic datasets and do pre-processing 
     
   - Train with DL models
     
   - Implement XAI methods on the model
  
   - Do sensitivity and robustness test
   
5. Load, train and evaluate real EEG datasets in Depression analysis

   - Get your real EEG public datasets :

     MODMA (https://modma.lzu.edu.cn/data/application/)

     PRED+CT (http://predict.cs.unm.edu/downloads.php)

     MPHC (https://figshare.com/articles/dataset/EEG_Data_New/4244171)

   - Train the dataset using DL models
     
   - Implement XAI methods on the model

   - Get feature important and topographic maps
