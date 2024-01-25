

input='/home/tonyo/2024_01_11_ukb_prescription_conversion/data/presner_input.txt'
output='/home/tonyo/2024_01_11_ukb_prescription_conversion/out'

wc -l ${input}

cd /home/tonyo/2024_01_11_ukb_prescription_conversion/PRESNER/PRESNER_dir
python PRESNER.py -i ${input} -o ${output} -m cbb chembl -s