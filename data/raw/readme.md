This directory contains files
- obtained straight from UKB
- obtained from the UKB website
- helper scripts for preprocessing the data

# Files obtained straight from UKB

These files have been replaced with symlinks.

## Prescription data:
drug_export/data/gp_scripts.txt

This file is directly downloadable with access:
https://biobank.ndph.ox.ac.uk/showcase/rectab.cgi?id=1062

## Disease Onset data

disease_onset.csv

This is the aggregate of the 'First Occurrence' data fields from UKB:
https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=1712

## Other UKB Fields

ukb_export/ukb670820.enc_ukb
ukb_export/ukbconv

UKB encypted data file. Can be extracted with ukbconv:
https://biobank.ndph.ox.ac.uk/showcase/download.cgi?id=101&ty=ut

The accessed field can be found in ukb_export/field.txt or seperated by category in the other
field.*.txt files.

# Files obtained from the UKB website

these files are publicly available on the [UKB website](https://biobank.ndph.ox.ac.uk/showcase/index.cgi)
, but we also provide them here for convenience (Date of downloads: 2024.01.20).

## From the official downloads list
https://biobank.ndph.ox.ac.uk/showcase/download.cgi

helper_files/ehierstring.txt: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=12 

## From essential information
https://biobank.ndph.ox.ac.uk/showcase/exinfo.cgi?src=AccessingData

helper_files/Codings.tsv: https://biobank.ndph.ox.ac.uk/~bbdatan/Codings.csv

helper_files/Data_Dictionary_Showcase.tsv: https://biobank.ndph.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.tsv








