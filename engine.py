import torch.optim as optim
import torch.cuda as cuda
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, adjacency, K, ndims, blocks,
                 time_conv_kernel, time_conv_padding, adaptive, num_attention_heads, dropout, norm,
                 lrate, wdecay, device, clip):

        self.model = MAFGNN(adjacency, seq_length, seq_length, in_dim, K, blocks, ndims,
                time_conv_kernel, time_conv_padding, adaptive, num_attention_heads, dropout, norm)
        self.model.to(device)
        self.optimizer = None
        self.scheduler = None

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = clip

        self.horizon = [i for i in range(12)]           # default

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)
        predict = self.scaler.inverse_transform(output) # (B, N, T)

        predict = predict[..., self.horizon]            # (B, N, H)
        real = real_val[..., self.horizon, 0]           # (B, N, H)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    @torch.no_grad()
    def eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)
        predict = self.scaler.inverse_transform(output) # (B, N, T)

        predict = predict[..., self.horizon]            # (B, N, H)
        real = real_val[..., self.horizon, 0]           # (B, N, H)

        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

