
export PYTHONPATH=~/meliad2:$PYTHONPATH
export BUCKET=meliad_eu2
export LOGURU_LEVEL=INFO
# /usr/bin/env python3 src/data.py test_stuff

/usr/bin/env python3 src/train_dpr.py --output_dir blabla2  \
    --adam_beta2 0.95 \
    --weight_decay 1e-1 \
    --learning_rate 1e-5 \
     --dataset_name iohadrubin/nq