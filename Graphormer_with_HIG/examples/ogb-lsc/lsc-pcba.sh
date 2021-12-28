[ -z "${exp_name}" ] && exp_name="pcba_pretrain"
[ -z "${seed}" ] && seed="1"
# [ -z "${arch}" ] && arch="--ffn_dim 768 --hidden_dim 768 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 1024 --dropout_rate 0.1 --attention_dropout_rate 0.3 --n_layers 18 --peak_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="256"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="../../exps/pcba_pretrain/$exp_name/$seed"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)
max_epochs=301

python ../../graphormer/entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name PCQM4Mv2 \
      --gpus $n_gpu --accelerator ddp --precision 16 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir --max_epochs $max_epochs
