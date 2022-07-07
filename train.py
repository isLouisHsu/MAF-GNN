import torch
import random
import numpy as np
import argparse
import time
import util
from pprint import pprint
import matplotlib.pyplot as plt
from engine import trainer
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='../traffic_data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='../traffic_data/adj_mx.pkl',help='adj data path')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--adjtype',type=str,default='binary',help='adj type')

parser.add_argument('--model_name',type=str,default='maf-gnn',help='')
parser.add_argument('--checkpoint',type=str,default=None,help='checkpoint path to resume')

parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=48,help='batch size')
parser.add_argument('--epochs',type=int,default=80,help='')
parser.add_argument('--print_every',type=int,default=200,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--gradient_clip',type=float,default=5.,help='gradient clip')
parser.add_argument('--scheduler',type=str,default='step_every_30_epoch_0.5',help='experiment id')
parser.add_argument('--optimizer',type=str,default='adam',help='experiment id')

args = parser.parse_args()


def init_seed(seed=None):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(optimizer, mode):
    schedulers = {
        'constant': lr_scheduler.LambdaLR(optimizer, lambda x: 1),
        'reduce': lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, min_lr=1e-8),
        'step_every_30_epoch_0.1': lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1),
        'step_every_30_epoch_0.3': lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.3),
        'step_every_30_epoch_0.5': lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.5),
        'step_every_50_epoch_0.5': lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200], gamma=0.5),
        'cosine_t_10': lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        'ploy_0.2': lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / args.epochs) ** 2.0),
        'exp_0.99': lr_scheduler.ExponentialLR(optimizer, gamma=0.97),
    }
    return schedulers[mode]


def get_optimizer(parameters, mode, lr, weight_decay):
    optimizers = {
        'adam': optim.Adam,
    }
    return optimizers[mode](parameters, lr=lr, weight_decay=weight_decay)


def fine_tune(model_to_train, status_pre_train):

    status_to_train = model_to_train.state_dict()
    status_to_train.update(status_pre_train)
    model_to_train.load_state_dict(status_to_train)

    return model_to_train


mode_params = {
    'maf-gnn': {
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

    #set seed
    init_seed(args.seed)

    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    engine = trainer(
        scaler=scaler,
        in_dim=args.in_dim,
        seq_length=args.seq_length,
        adjacency=adj_mx[0],
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        clip=args.gradient_clip,
        **mode_params[args.model_name])

    his_loss =[]
    val_time = []
    train_time = []
    his_test_amae = []

    # -----------------------------------------------------------------------------
    start_epoch = 1
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location='cpu')
        engine.model.load_state_dict(state)
        start_epoch = int(args.checkpoint.split('_')[2]) + 1
        print(f'checkpoint loaded from {args.checkpoint}')

    # -----------------------------------------------------------------------------
    engine.optimizer = get_optimizer(engine.model.parameters(), mode=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay)
    t_total = dataloader['train_loader'].num_batch // 1 * args.epochs
    warmup_steps = int(t_total * 0.1)
    engine.scheduler = get_linear_schedule_with_warmup(
        engine.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # -----------------------------------------------------------------------------
    mvalid_rmse = float('inf')
    for i in range(start_epoch, start_epoch + args.epochs):

        train_loss = []
        train_mae  = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx = trainx.transpose(1, 2)
            trainy = trainy.transpose(1, 2)
            metrics = engine.train(trainx, trainy)
            engine.scheduler.step()
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            if iter % args.print_every == 0 :
                log = 'Epoch: {:03d} Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(i, iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)

        lr = engine.optimizer.param_groups[0]['lr']

        #validation
        valid_loss = []
        valid_mae  = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 2)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Train | Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}\nValid | Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}\nlr={:.8f}'
        print(log.format(mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, lr), flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        print('------------------------------------------------------------')
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 2)[..., 0]

        te2 = time.time()
        print("Training Time: {:.4f}s/epoch >>> ETA {:.4f}h".format(
            te2 - t1, (te2 - t1) * (args.epochs - i) / 3600))
        print('============================================================')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
