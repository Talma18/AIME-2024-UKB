


# Download
downloaded the GP prescription table from the UKB dataportal (login required) -> data/gp_scripts.txt
https://biobank.ndph.ox.ac.uk/showcase/dbportal.cgi


# Prep

`presner_prep.py`

Narrowed down to `drug_name`, `quantity` cols, dropped duplicates and NAs -> `data/presner_input.txt`


# PRESNER
Followed
https://github.com/ccolonruiz/PRESNER

Ran with `run_presner.sh`


# Mapping

See `ukb_map.ipynb` with comments.
