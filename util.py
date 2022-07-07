import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        print(f'{xs.shape}, {ys.shape}')

    def __len__(self):

        return self.size // self.batch_size + 1

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std, max, min):
        self.mean = mean
        self.std = std
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return torch.clamp((data * self.std) + self.mean, self.min, self.max)
        # return (data * self.std) + self.mean


class StandardScalerV2():
    """
    Standard the input
    """

    def __init__(self, mean, std, max, min):
        self.mean = mean
        self.std = std
        self.max = max
        self.min = min

    def transform(self, data):
        # 只对正常值作归一化，相当于用均值填充缺失
        return np.where(data == 0., 0., (data - self.mean) / self.std)
        # return (data - self.mean) / self.std

    def inverse_transform(self, data):
        # 限幅
        return torch.clamp((data * self.std) + self.mean, self.min, self.max)
        # return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == 'binary':
        adj = np.where(adj_mx > 0., 1., 0.)
        adj = adj - np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)
        adj = [adj]
    elif adjtype == 'binary_with_self_loop':
        adj = np.where(adj_mx > 0., 1., 0.)
        adj = [adj]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        x, y = cat_data['x'], cat_data['y']
        n1 = x.shape[0]

        # if category == 'train':
        #     x, y = drop_abnormal_data(x, y)
        #     print(f'{category} drop samples: {n1} -> {x.shape[0]}')

        # x, y = drop_abnormal_data(x, y)
        # print(f'{category} drop samples: {n1} -> {x.shape[0]}')

        data['x_' + category] = x
        data['y_' + category] = y

    # 时间信息


    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std(),
        min=data['x_train'][..., 0].min(),
        max=data['x_train'][..., 0].max(),
    )
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def load_dataset_v2(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    """
    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std(),
        min=data['x_train'][..., 0].min(),
        max=data['x_train'][..., 0].max(),
    )
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    """
    # 去除异常值后求取均值方差
    train_data = data['x_train'][..., 0].flatten()
    train_data = train_data[train_data != 0]
    scaler = StandardScalerV2(
        mean=train_data.mean(),
        std=train_data.std(),
        min=train_data.min(),
        max=train_data.max(),
    )
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask

    d = loss.max() / 5.
    loss = torch.where(loss < d, 0.5 * (loss ** 2), d * loss - 0.5 * d ** 2)

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_berhu(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask

    d = loss.max() / 5.
    loss = torch.where(loss < d, loss, (loss ** 2 + d ** 2) / (2 * d))

    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_berhu_ohem(preds, labels, null_val=np.nan, keep_ohem=0.7):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask

    # d = loss.max() / 5.
    d = loss.max() / 1.5
    loss = torch.where(loss < d, loss, (loss ** 2 + d ** 2) / (2 * d))

    # fixme: 下面一句如果没有，会导致验证集loss为nan，因为存在全部是0的验证集样本
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # ohem
    loss = loss.mean(dim=-1)

    loss = loss.flatten()
    keep_num = int(loss.size(0) * keep_ohem)
    if keep_num == 0: return torch.mean(loss)
    loss_sorted, _ = torch.sort(loss, descending=True)
    loss = torch.mean(loss_sorted[: keep_num])

    return loss


def metric(preds, labels, null_val=np.nan):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae,mape,rmse



# ------------------------------------------------------------------


def masked_berhu_ohem_loss_v2(preds, labels, null_val=np.nan, thesh=2.0, keep_ohem=1.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    loss = torch.abs(preds - labels)
    loss = loss.masked_select(mask)

    if loss.size(0) == 0:
        return torch.tensor(0., device=preds.device)

    # berhu
    d = loss.max() / thesh
    loss = torch.where(loss <= d, loss, (loss ** 2 + d ** 2) / (2 * d))

    # ohem
    keep_num = int(loss.size(0) * keep_ohem)
    if keep_num == 0: return torch.mean(loss)
    loss_sorted, _ = torch.sort(loss, descending=True)
    loss = torch.mean(loss_sorted[: keep_num])

    return loss


def masked_mae_loss_v2(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    loss = torch.abs(preds - labels)
    loss = loss.masked_select(mask)

    if loss.size(0) == 0:
        return torch.tensor(0., device=preds.device)

    return loss.mean()


def masked_mae_v2(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    loss = torch.abs(preds - labels)
    loss = loss.masked_select(mask)

    if loss.size(0) == 0:
        return torch.tensor(0., device=preds.device)

    return loss.mean()


def masked_mape_v2(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    loss = torch.abs((preds - labels) / (labels + 1e-8))
    loss = loss.masked_select(mask)

    if loss.size(0) == 0:
        return torch.tensor(0., device=preds.device)

    return loss.mean()


def masked_mse_v2(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    loss = (preds - labels) ** 2
    loss = loss.masked_select(mask)

    if loss.size(0) == 0:
        return torch.tensor(0., device=preds.device)

    return loss.mean()


def masked_rmse_v2(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_v2(preds, labels, null_val))


def metric_v2(preds, labels, null_val=np.nan):
    mae = masked_mae_v2(preds, labels, null_val).item()
    mape = masked_mape_v2(preds, labels, null_val).item()
    rmse = masked_rmse_v2(preds, labels, null_val).item()
    return mae, mape, rmse




from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler


class FindLR(_LRScheduler):
    """exponentially increasing learning rate
    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """

    def __init__(self, optimizer, max_lr=10, total_iters=100, last_epoch=-1):
        self.total_iters = total_iters
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in
                self.base_lrs]


def find_lr(engine, dataloader, device, num_iter=100, min_lr=1e-6, max_lr=1):

    scheduler = FindLR(engine.optimizer, max_lr=max_lr, total_iters=num_iter)

    learning_rate = []
    losses = []

    for param_group, lr in zip(scheduler.optimizer.param_groups, scheduler.get_lr()):
        param_group['lr'] = min_lr

    n = 0
    print("finding best learning rate...")
    epochs = num_iter // len(dataloader['train_loader']) + 1
    for i in range(epochs):
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 2)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 2)
            metrics = engine.train(trainx, trainy)

            losses.append(metrics[0])
            learning_rate.append(scheduler.get_lr()[0])
            scheduler.step()
            n += 1

            if n > num_iter: break
            print(f"[{n}]/[{num_iter}] {learning_rate[-1]} {losses[-1]}", flush=True)

    fig, ax = plt.subplots(1, 1)
    ax.plot(learning_rate, losses)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('losses')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    fig.savefig('lr_finder.png')

    print('Learning rate finding done! Please checkout `result.png`, choose the best learning rate and restart!')

    exit(0)
