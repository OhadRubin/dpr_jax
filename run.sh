export JAX_LOG_COMPILES=1
python3 src/train_dpr.py --output_dir blabla  --train_dir https://huggingface.co/datasets/iohadrubin/nq/resolve/main/data/train-00000-of-00012-aebee16ac9d5ed6f.parquet,https://huggingface.co/datasets/iohadrubin/nq/resolve/main/data/train-00001-of-00012-aebee16ac9d5ed6f.parquet,https://huggingface.co/datasets/iohadrubin/nq/resolve/main/data/train-00002-of-00012-aebee16ac9d5ed6f.parquet
# --dataset_name iohadrubin/nq
sudo kill -9 $(sudo lsof -w /dev/accel0 | awk ‘NR>1{print $2}’ |uniq)