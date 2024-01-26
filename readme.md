Repository for the paper "Biobank-scale Unified Representations by Modular Transformers: a novel stratification of treatment-resistant depression".

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

---

# Example tokenization of a fictional patient

To further demonstrate the tokenization process, we provide an example tokenization of a fictional patient. All values used in the example are for purely demonstration purposes. Some sequence, which would be required for model/loss are omitted here for brevity. Consider a patient with the following information:

- List of Diseases: [F30, 2000-01-01]
- List of Prescriptions: [CHEMBL865, 2012-03-13], [CHEMBL715, 2019-06-29]
- Available Personal Information: [31-0.0(Sex) Male 2010-10-13], [21002-0.0(Weight) 56.3kg 2010-10-13]
- Available Laboratory Information: [23470-0.0(Glucose) 4.1 mmol/l 2010-10-13]

The initial tokenization step outputs the following sequences, separating information into distinct sequences.

- Disease:
    - tokens: `<F30>`
    - type: `<event>`
    - year: `<2000>`
    - month: `<01>`
    - day: `<01>`
- Prescription:
    - tokens: `<CHEMBL865>`, `<CHEMBL715>`
    - type: `<medication>`
    - year: `<2012>`, `<2019>`
    - month: `<03>`, `<06>`
    - day: `<13>`, `<29>`
- Personal Information:
    - tokens: `<31-0.0>`, `<Male>`, `<21002-0.0>`, `<56.3>`
    - type: `<event>`, `<categorical>`, `<event>`, `<numeric>`
    - year: `<2010>`, `<no-date>`, `<2010>`, `<no-date>`
    - month: `<10>`, `<no-date>`, `<10>`, `<no-date>`
    - day: `<13>`, `<no-date>`, `<13>`, `<no-date>`
- Laboratory Information:
    - tokens: `<23470-0.0>`, `<4.1>`
    - type: `<event>`, `<numeric>`
    - year: `<2010>`, `<no-date>`
    - month: `<10>`, `<no-date>`
    - day: `<13>`, `<no-date>`

Next The different token->id map are used to assign an integer to each token. In the embedding layers these integers are used to get the embedding vectors, with 0 being assigned fixed to the 0 vector. Special tokens indicating th start (`<bos>`) and end of sequences are also added (`<eos>`).

- Disease:
    - token IDs: `[1, 301, 2]`
    - type IDs: `[0, 1, 0]`
    - year IDs: `[0, 67, 0]`
    - month IDs: `[0, 1, 0]`
    - day IDs: `[0, 1, 0]`
- Prescription:
    - Token IDs: `[1, 865, 715, 2]`
    - Type IDs: `[0, 1, 1, 0]`
    - Year IDs: `[0, 69, 77, 0]`
    - Month IDs: `[0, 3, 6, 0]`
    - Day IDs: `[0, 13, 29, 0]`
- Personal Information:
    - Token IDs: `[1, 1341, 2110, 1367, -0.3423, 2]`
    - Type IDs: `[0, 1, 2, 1, 3, 1]`
    - Year IDs: `[0, 2010, 0, 2010, 0, 0]`
    - Month IDs: `[0, 10, 0, 10, 0, 0]`
    - Day IDs: `[0, 13, 0, 13, 0, 0]`
- Laboratory Information:
    - Token IDs: `[1, 23470,  0.34, 2]`
    - Type IDs: `[0, 1, 3, 0]`
    - Year IDs: `[0, 67, 0, 0]`
    - Month IDs: `[0, 10, 0, 0]`
    - Day IDs: `[0, 13, 0, 0]`
