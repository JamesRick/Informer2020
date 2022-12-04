python -u main_informer.py \
--model informer \
--data zillow \
--root_path "./data/ga_preprocessed/all_sm_sa_month" \
--data_path "ml_dataset.csv" \
--features "S" \
--freq "m" \
--seq_len 36 \
--label_len 24 \
--pred_len 12 \
--e_layers 2 \
--d_layers 1 \
--attn prob \
--des 'Exp' \
--itr 5 \
--factor 3

# mean: array([188895.30822766])
# std: array([78965.32009149])