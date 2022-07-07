# Multi-Adaptive Spatiotemporal Graph Neural Network

## Requirements

See `requirements.txt`.

## Prepare Data

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Process raw data 

``` bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py \
    --output_dir=../traffic_data/METR-LA \
    --traffic_df_filename=../traffic_data/metr-la.h5

# PEMS-BAY
python generate_training_data.py \
    --output_dir=../traffic_data/PEMS-BAY \
    --traffic_df_filename=../traffic_data/pems-bay.h5

```

## Train

```
python train.py --device cuda:0 \
                --data ../traffic_data/METR-LA \
                --adjdata ../traffic_data/adj_mx.pkl \
                --save ./garage/metr \
                --adjtype binary \
                --model_name maf-gnn \
                --epochs 100 \
                --batch_size 64 \
                --learning_rate 0.001 \
                --weight_decay 0.0001 \
                --gradient_clip 5

python train.py --device cuda:0 \
                --data ../data/traffic_data/PEMS-BAY \
                --adjdata ../data/traffic_data/adj_mx_bay.pkl \
                --save ./garage/pems \
                --adjtype binary \
                --model_name maf-gnn \
                --epochs 100 \
                --batch_size 64 \
                --learning_rate 0.001 \
                --weight_decay 0.0001 \
                --gradient_clip 5
```

## Test

```
python test.py --device cuda:0 \
                --data ../traffic_data/METR-LA \
                --adjdata ../traffic_data/adj_mx.pkl \
                --adjtype binary \
                --model_name maf-gnn \
                --batch_size 64 \
                --learning_rate 0.001 \
                --weight_decay 0.0001 \
                --gradient_clip 5

python test.py --device cuda:0 \
                --data ../data/traffic_data/PEMS-BAY \
                --adjdata ../data/traffic_data/adj_mx_bay.pkl \
                --adjtype binary \
                --model_name maf-gnn \
                --batch_size 64 \
                --learning_rate 0.001 \
                --weight_decay 0.0001 \
                --gradient_clip 5
```
