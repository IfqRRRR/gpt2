decode_strategy="top-p" # "top-p", "top-k"
extra_name="none"
n_layer=3

python main.py --cpu_count 40 --cuda_num 0 --num_epochs 100 --n_layer ${n_layer} --extra_name ${extra_name} --use_pretrain \
--name "${decode_strategy}_${extra_name}_${n_layer}" --decode_strategy ${decode_strategy} --top_p 0.9 --temperature 1 --top_k 40

# --test --choose_layer '1 6 12'
# use_pretrain only_result