# How to Leverage Multimodal EHR Data for Better Medical Predictions?
This repository contains the code of the paper: How to Leverage Multimodal EHR Data for Better Medical Predictions? 

## Installation
All the dependencies are in the requirements.txt. You can build the environment with the following command:

`conda create --name <env> --file requirements.txt`

## Data Download
The MIMIC-III data can be downloaded at: https://physionet.org/content/mimiciii/1.4/. This dataset is a restricted-access resource. To access the files, you must be a credentialed user and sign the data use agreement (DUA) for the project.
Because of the DUA, we cannot provide the data directly.

The pre-trained parameters of ClinicalBERT can be downloaded at: https://github.com/kexinhuang12345/clinicalBERT.



## Instructions
To run the code in this folder, please follow the instructions below.

1. Download the MIMIC-III data.
2. Extract the features by the scripts provided at: https://github.com/MLD3/FIDDLE-experiments
3. Set the mimic_dir and data_dir in data_module.py to the path of the above data.
4. Set the task related information and run data_module.py to combine clinical notes with other data.
5. Run the model by: `python run.py --task=task_name --model=model_name`
