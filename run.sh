
export PYTHONPATH=~/meliad2:$PYTHONPATH
export BUCKET=meliad_eu2
export LOGURU_LEVEL=INFO
export JAX_LOG_COMPILES=1
# /usr/bin/env python3 src/data.py test_stuff

export WANDB_NAME="wd-1e-1_b2-0.95_lr6e-5_v16"
/usr/bin/env python3 src/train_dpr.py --output_dir blabla2  \
    --adam_beta2 0.95 \
    --weight_decay 1e-1 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 32 \
     --dataset_name iohadrubin/nq