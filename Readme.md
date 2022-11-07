# CCBDA hw2

311511035 侯俊宇

## Env

To install dependencies: 

`pip install -r requirements.txt`

The data directory should be arranged as:

```shell
data
├── test
└── unlabeled
```

To train:

```shell
python main.py \
--root_dir PATH_TO_DATA \
--num_epoch 5000 \
--hidden_size 128 \
--batch_size 1024 \
--exp_name exp6 \
--accumulate 5
```

To test and generate embedding:

```shell
python main.py \
--num_epoch 5000 \
--hidden_size 128 \
--batch_size 1024 \
--exp_name exp6 \
--accumulate 5 \
--test \
--weight PATH_TO_BEST_CHECKPOINT
```

The checkpoint weights should be stored in `ckpt/` folder. Just replace PATH_TO_BEST_CHECKPOINT with the desired checkpoint.
