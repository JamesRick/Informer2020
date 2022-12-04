python -u main_informer.py \
--model informer \
--data zillow_all_sm_sa_month \
--root_path "./data/ga_preprocessed/all_sm_sa_month" \
--data_path "County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month_ml_dataset.csv" \
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

python -u main_informer.py \
--model informer \
--data zillow_bdrmcnt_2_uc_sfrcondo \
--root_path "./data/ga_preprocessed/bdrmcnt_2_uc_sfrcondo" \
--data_path "County_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month_ml_dataset.csv" \
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

python -u main_informer.py \
--model informer \
--data zillow_condo_sm_sa_month \
--root_path "./data/ga_preprocessed/condo_sm_sa_month" \
--data_path "County_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month_ml_dataset.csv" \
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

python -u main_informer.py \
--model informer \
--data zillow_sfr_sm_sa_month \
--root_path "./data/ga_preprocessed/sfr_sm_sa_month" \
--data_path "County_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month_ml_dataset.csv" \
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

python -u main_informer.py \
--model informer \
--data zillow_zori_sm_sa_month \
--root_path "./data/ga_preprocessed/zori_sm_sa_month" \
--data_path "County_zori_sm_sa_month_ml_dataset.csv" \
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