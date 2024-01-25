Repository for the paper "Multimorbidity-boosted Patient Stratification Using
Biobank-scale Representations from Modular
Transformers".

# How to use

First, install the requirements from requirements.txt using your preferred package manager.

Replace the appropriate data files, as described in data/raw/readme.md.

Run the preprocessing scripts.
- run presner as explained in data/raw/drug_export/readme.md
- get treatment resistant depression information by following data/raw/trd/readme.md
- run the extract_ukb_data.ipynb to prepare files for the training scripts

Run training scripts.
- main.py trains the model with all modalities
- main_disease_only trains a reference model with only disease information

Run prediction scripts.
- predict.py save the learnt representations into the model folder
- predict_disease.py saves the learnt representations into the model folder

Please note, that most scripts are hard-coded and not written with argparse. Please check the 
contained code before running the scripts.