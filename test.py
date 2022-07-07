import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train import init_seed
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='../traffic_data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='../traffic_data/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='binary',help='adj type')

parser.add_argument('--model_name',type=str,default='maf-gnn',help='')
parser.add_argument('--checkpoint',type=str,required=True,help='')

parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0005,help='weight decay rate')
parser.add_argument('--gradient_clip',type=float,default=5.,help='gradient clip')
parser.add_argument('--scheduler',type=str,default='step_every_30_epoch_0.5',help='experiment id')
parser.add_argument('--optimizer',type=str,default='adam',help='experiment id')

args = parser.parse_args()


mode_params = {
    'baseline': {
        "blocks": [
            [256, 256, 64, 64],
            [64, 64, 64, 64],
            [64, 64, 64, 64],
            [64, 64, 64, 64],
        ],
        "ndims": [8, 8, 8, 8],
        "K": 2,
        "time_conv_kernel": 3,
        "time_conv_padding": 1,
        "adaptive": True,
        "num_attention_heads": 4,
        "dropout": 0.3,
        "norm": 'bn'
    }
}


def main():
    print(args)

    init_seed(args.seed)
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    engine = trainer(
        scaler=scaler,
        in_dim=args.in_dim,
        seq_length=args.seq_length,
        adjacency=adj_mx[0],
        dropout=args.dropout,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        clip=args.gradient_clip,
        **mode_params[args.model_name])

    state = torch.load(args.checkpoint, map_location='cpu')
    engine.model.load_state_dict(state)
    print(f'checkpoint loaded from {args.checkpoint}')

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 2)[..., 0]

    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 2)
        with torch.no_grad():
            preds = engine.model(testx)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real, 0.0)
        log = 'Horizon {:d} min, Test MAE: {:.8f}, Test MAPE: {:.8f}, Test RMSE: {:.8f}'
        print(log.format((i + 1) * 5, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.8f}, Test MAPE: {:.8f}, Test RMSE: {:.8f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == '__main__':
    main()