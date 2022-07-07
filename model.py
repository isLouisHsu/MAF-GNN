import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from itertools import chain


class TemporalConvLayer(nn.Module):

    def __init__(self, N, in_channels, out_channels, time_conv_kernel, time_conv_padding, dropout):
        super(TemporalConvLayer, self).__init__()

        conv_params = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, time_conv_kernel),
            stride=(1, 1),
            padding=(0, time_conv_padding),
        )
        self.conv_p = nn.Conv2d(**conv_params)
        self.conv_q = nn.Conv2d(**conv_params)
        self.conv_r = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kernel=None):
        """
        :param x: tensor(B, N, Tin, Cin)
        :return:
        """
        # (B, N, Tin, Cin) -> (B, Cin, N, Tin)
        x = x.permute(0, 3, 1, 2)

        # (B, Cin, N, Tin) -> (B, Cout, N, T')
        if self.conv_r:
            r = self.conv_r(x)
        else:
            r = x

        s = self.conv_p(x)
        g = torch.sigmoid(self.conv_q(x))
        x = s * g + r * (1 - g)

        x = x.permute(0, 2, 3, 1)
        x = self.dropout(x)

        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # (B, C, N, T) (N, N)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0, support_len=2, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x: (B, C, N, T)
        # support: [(N, N), ...]
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SpatialGraphConvLayer(nn.Module):

    def __init__(self, N, K, in_channels, out_channels, dropout, support_len=8):
        super(SpatialGraphConvLayer, self).__init__()

        self.conv_r = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.gcn = gcn(in_channels, out_channels, support_len=support_len, order=K)    # support_len = 8
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):

        if self.conv_r:
            res = self.conv_r(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            res = x

        # (B, N, T, C) -> (B, C, N, T)
        x = self.gcn(x.permute(0, 3, 1, 2), supports).permute(0, 2, 3, 1)

        # new: relu
        x = F.relu(self.dropout(x + res))

        return x



class SpatiotemporalFlowModule(nn.Module):

    def __init__(self, N, T, K, ti_channels, si_channels, so_channels, to_channels,
                 time_conv_kernel, time_conv_padding, dropout, norm='bn', **kwargs):
        super(SpatiotemporalFlowModule, self).__init__()

        if si_channels > 0:
            assert to_channels == so_channels

            self.linear_t = nn.Linear(ti_channels, si_channels)
            self.linear_s = nn.Linear(si_channels, to_channels)

        self.temporal = TemporalConvLayer(N, ti_channels, to_channels, time_conv_kernel, time_conv_padding, dropout)
        self.spatial = SpatialGraphConvLayer(N, K, to_channels, so_channels, dropout, **kwargs)

        self.conv_r = nn.Conv2d(ti_channels, so_channels, 1) if ti_channels != so_channels else None

        self.norm = norm
        if norm == 'bn':
            self.bn = nn.BatchNorm2d(N)
        elif norm == 'gn':
            self.gn = UnitedNorm(so_channels)

    def forward(self, xs, xt, kernel, adj=None):
        """
        :param xs: tensor(B, N, T, si_channels)
        :param xt: tensor(B, N, T, ti_channels)
        :param kernel: tensor(K + 1, N, N)
        :param adj: tensor( N, N)
        :return:
        """
        # (B, N, T, si_channels) -> (B, N, T, to_channels)
        to = self.temporal(xs + self.linear_t(xt))

        # (B, N, T, to_channels) -> (B, N, T, so_channels)
        so = self.spatial(to + self.linear_s(xs), kernel)

        if self.conv_r:
            res = self.conv_r(xt.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            res = xt

        # so = F.relu(self.bn(so + res))

        so = so + res
        if self.norm == 'bn':
            so = self.bn(so)
        elif self.norm == 'gn':
            B, N, T, C = so.size()
            so = so.permute(0, 2, 1, 3).contiguous().view(-1, N, C)
            so = self.gn(so, adj)
            so = so.contiguous().view(B, T, N, C).permute(0, 2, 1, 3)
        so = F.relu(so)

        return so, to


class PositionEmbedding(nn.Module):

    def __init__(self, sequence_length, hidden_size, requires_grad=False):
        super(PositionEmbedding, self).__init__()

        position_embedding = []
        pos = torch.arange(sequence_length).float()
        for i in range(hidden_size):
            pe = torch.where(pos % 2 == 0,
                torch.sin(pos / (10000 ** (2 * i / hidden_size))),
                torch.cos(pos / (10000 ** (2 * i / hidden_size))))
            position_embedding.append(pe)
        # (T, C)
        # self.register_buffer('position_embedding', torch.stack(position_embedding, dim=-1))
        self.position_embedding = nn.Parameter(
            torch.stack(position_embedding, dim=-1), requires_grad=requires_grad)

    def forward(self, x):
        # (B, T, C)
        return x + self.position_embedding


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads=1):
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):

        # (B, T, C)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class GlobalAttLSTMLayer(nn.Module):

    def __init__(self, N, Tin, Tout, Cin, Cout, num_attention_heads, dropout):
        super(GlobalAttLSTMLayer, self).__init__()

        # new: position embedding
        self.pe = PositionEmbedding(Tin, Cin)
        self.att = SelfAttention(Cin, num_attention_heads)
        self.dropout = nn.Dropout(dropout)

        hidden_size = 384
        self.lstm = nn.LSTM(
            input_size=Cin, 
            hidden_size=hidden_size // 2, 
            bidirectional=True,
            batch_first=True,
            num_layers=1)
        self.fc = nn.Linear(hidden_size, Cout)

    def forward(self, x, graph_kernel=None):
        """
        :param x: torch.tensor(B, N, tin, Cin)
        :param graph_kernel: torch.tensor(K, N, N)
        :return x: torch.tensor(B, N, tout, Cout)
        """
        B, N, T, Cin = x.size()
        x = x.contiguous().view(-1, T, Cin)

        x = self.pe(x)
        x = self.att(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.fc(x)

        x = x.view(B, N, T, -1).squeeze(-1)

        return x


class AdaptiveAdjacencyMatrix(nn.Module):

    def __init__(self, adjacency, K, ndim, adaptive, dropout):
        super(AdaptiveAdjacencyMatrix, self).__init__()

        if not isinstance(adjacency, torch.Tensor):
            adjacency = torch.tensor(adjacency, dtype=torch.float)

        self.K, N = K, adjacency.size(0)
        # (N, N)
        self.adjacency = nn.Parameter(adjacency, requires_grad=False)

        self.embedding = nn.Parameter(torch.Tensor(N, ndim), requires_grad=adaptive)
        self.W1 = nn.Parameter(torch.Tensor(ndim, ndim // 2), requires_grad=adaptive)
        self.W2 = nn.Parameter(torch.Tensor(ndim, ndim // 2), requires_grad=adaptive)
        nn.init.uniform_(self.embedding)
        nn.init.kaiming_uniform_(self.W1)
        nn.init.kaiming_uniform_(self.W2)

        self.dropout = nn.Dropout(dropout)

        # (K, N, N)
        self.chebKernel = None

    def _adjacency(self):
        """
        :return adjacency: torch.tensor(N, N)，邻接矩阵
        """
        if self.embedding.requires_grad:

            # (N, ndim // 2)
            e1 = self.embedding.mm(self.W1.softmax(dim=0))
            e2 = self.embedding.mm(self.W2.softmax(dim=0))
            e1 = e1 / (e1.norm(dim=1, keepdim=True) + 1e-8)
            e2 = e2 / (e2.norm(dim=1, keepdim=True) + 1e-8)
            weight = e1.matmul(e2.t())

            # for gradient stability
            adjacency = torch.where(self.adjacency > 0.,
                weight + self.adjacency, 9e-15 * torch.ones_like(weight))
        else:
            adjacency = torch.where(self.adjacency > 0.,
                self.adjacency, 9e-15 * torch.ones_like(self.adjacency))

        # new: dropout
        adjacency = self.dropout(adjacency)

        return adjacency

    def _sym(self, adj):

        Dr = torch.diag(1. / adj.sum(dim=1).sqrt())

        return Dr @ adj @ Dr

    def _sym2(self, adj):

        Dr = torch.diag(1. / adj.sum(dim=1))

        return Dr @ adj

    def _kernel(self):
        """
        :return kernel: torch.tensor(K, N, N)，图卷积核多项式展开的基
        """
        adjacency = self._adjacency()

        # laplacian matrix
        I = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
        L = I - self._sym(adjacency)

        # chebyshev ploynomials
        maxeig = 2.
        Ls = 2 / maxeig * L - I

        kernel = [I, Ls]
        for k in range(1, self.K):
            kernel.append(2 * Ls @ kernel[-1] - kernel[-2])
        kernel = torch.stack(kernel, dim=0)

        return kernel

    def kernel(self):
        """
        :return kernel: torch.tensor(K, N, N)，图卷积核多项式展开的基
        """
        if not self.embedding.requires_grad:
            if self.chebKernel is None:
                self.chebKernel = self._kernel()
            return self.chebKernel
        else:
            return self._kernel()

    def forward(self, *input):

        raise NotImplementedError


class AdaptiveAdjacencyMatrices(nn.Module):

    def __init__(self, adjacency, K, ndims, adaptive, dropout):
        super(AdaptiveAdjacencyMatrices, self).__init__()

        if not isinstance(adjacency, torch.Tensor):
            adjacency = torch.tensor(adjacency, dtype=torch.float)

        self.adjacency = nn.Parameter(adjacency, requires_grad=False)
        self.adaptive = adaptive

        embeddings = []
        for ndim in ndims:
            embeddings.append(AdaptiveAdjacencyMatrix(adjacency, K, ndim, adaptive, dropout))
        self.embeddings = nn.ModuleList(embeddings)

        self.K, N = K, adjacency.size(0)
        # (K, N, N)
        self.chebKernel = None

    def reset_parameters(self):
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _adjacency(self):

        adjacency = []
        for embedding in self.embeddings:
            adjacency.append(embedding._adjacency())

        return adjacency

    def _sym(self, adj):

        Dr = torch.diag(1. / adj.sum(dim=1).sqrt())
        return Dr @ adj @ Dr

    def _sym2(self, adj):

        Dr = torch.diag(1. / adj.sum(dim=1))
        return Dr @ adj


class MAFGNN(AdaptiveAdjacencyMatrices):
    """  邻接矩阵形式不同，注意与SpatialGraphConvLayer一起使用 """
    def __init__(self, adjacency, Tin, Tout, in_dim, K, blocks, ndims,
                 time_conv_kernel, time_conv_padding, adaptive,
                 num_attention_heads, dropout, norm):
        super(MAFGNN, self) \
            .__init__(adjacency, K, ndims, adaptive, dropout)

        inlayer_hidden_size = blocks[0][0]
        self.inlayer = nn.Sequential(
            nn.Linear(in_dim, inlayer_hidden_size),
            nn.ReLU(inplace=True),
        )

        head_hidden_size = 256
        self.n_blocks = len(blocks)
        self.st_blocks = nn.ModuleList()
        self.st_heads  = nn.ModuleList()
        for Csi, Cti, Cso, Cto in blocks:
            self.st_blocks.append(SpatiotemporalFlowModule(
                self.adjacency.size(0), Tin, K, Csi, Cti, Cso, Cto,
                time_conv_kernel, time_conv_padding, dropout, norm, 
                support_len=len(ndims) * 2))
            self.st_heads.append(nn.Linear(Cso, head_hidden_size))

        self.outlayer = GlobalAttLSTMLayer(
            adjacency.shape[0], Tin, Tout, head_hidden_size, 1,
            num_attention_heads, dropout)

    def forward(self, x, scaler=None):
        """
        :param X: (B, N, T, C)
        """
        xt = self.inlayer(x)
        xs = xt

        # connect all
        y = 0
        adjacency = self._adjacency()
        supports = list(chain(*[[self._sym2(a), self._sym2(a.t())] for a in adjacency]))
        for i in range(self.n_blocks):
            xs, xt = self.st_blocks[i](xs, xt, supports, self.adjacency)
            y = y + self.st_heads[i](xs)

        # output
        x = self.outlayer(y)
        x = x.squeeze(-1)

        if scaler:
            x = scaler.inverse_transform(x)

        return x