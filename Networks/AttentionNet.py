import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax


class AttentionConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)


        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        # self.edge_conv_lin = torch.nn.Sequential(
        #     torch.nn.Linear(out_channels + in_channels[0], 4*out_channels),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(4*out_channels, out_channels),
        # )

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        # self.edge_conv_lin.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, u: Tensor =None,
                edge_attr: OptTensor = None, return_attention_weights=None):
        ## type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value, u=u,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                u_i: Tensor, u_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr
        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dim_head=None,
                 dim_linear_block=1024, dim_u=None, dim_edge=None, dropout=0.1, activation=nn.GELU):
        super().__init__()
        self.dim_head = (int(out_channels / heads)) if dim_head is None else dim_head
        self.mhsa = AttentionConv(in_channels=in_channels, out_channels=self.dim_head, heads=heads, edge_dim=dim_edge)

        # Feed foward layer
        self.norm1 = torch_geometric.nn.BatchNorm(out_channels)
        self.norm2 = torch_geometric.nn.BatchNorm(out_channels)
        self.linear = nn.Sequential(
            nn.Linear(out_channels, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, out_channels),
            nn.Dropout(dropout)
        )

        # edge process & shortcut
        self.edge_process = nn.Sequential(
            torch.nn.Linear(dim_edge + 2*out_channels, dim_edge),
            activation(),
            torch.nn.Linear(dim_edge, dim_edge),
            nn.Dropout(dropout),
        )
        self.norm_edge = torch_geometric.nn.BatchNorm(dim_edge)

        self.act = activation()

    def forward(self, node_attr, edge_attr, edge_index, batch):
        # edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False)
        node_out = self.norm1(self.mhsa(x=node_attr, edge_attr=edge_attr, edge_index=edge_index))
        # node_out = self.norm1(self.mhsa(x=node_attr, edge_attr=edge_attr, edge_index=edge_index, u=u[batch]))
        node_out = (self.norm2(self.linear(node_out) + node_out))

        row, col = edge_index
        node_x, node_y = node_out[row], node_out[col]
        edge_out = torch.cat([edge_attr, node_x-node_y, node_x+node_y], dim=-1)
        edge_out = self.norm_edge(self.edge_process(edge_out))
        
        return self.act(node_out), self.act(edge_out)


class AttentionNet(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        dim_embedding = settings['dim_embedding']
         
        self.input_embedding = nn.Sequential(
            torch_geometric.nn.BatchNorm(settings['input_features']),
            nn.Linear(settings['input_features'], dim_embedding),
            torch_geometric.nn.BatchNorm(dim_embedding),
            nn.ReLU(),  
        )

        self.glob_embedding = nn.Sequential(
            torch_geometric.nn.BatchNorm(settings['global_features']),
            nn.Linear(settings['global_features'], 4*dim_embedding),
            nn.ReLU(),  
            nn.Linear(4*dim_embedding, dim_embedding),
            torch_geometric.nn.BatchNorm(dim_embedding),
            nn.ReLU(), 
        )

        self.edge_embedding = nn.Sequential(
            torch_geometric.nn.BatchNorm(settings['edge_features']),
            nn.Linear(settings['edge_features'], 4*dim_embedding),
            nn.ReLU(),  
            nn.Linear(4*dim_embedding, dim_embedding),
            nn.ReLU(), 
        )

        previous_output_shape = dim_embedding
        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['block_params']):
            out_channels, heads, dim_linear_block, dropout= layer_param
            self.conv_process.append(AttentionBlock(in_channels=previous_output_shape,
                                        out_channels=out_channels, heads=heads, dim_edge=dim_embedding,
                                        dim_linear_block=dim_linear_block, dropout=dropout,))
            previous_output_shape = out_channels


        self.fc_process = torch.nn.ModuleList()
        previous_output_shape = dim_embedding + previous_output_shape
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            )
            self.fc_process.append(seq)
            previous_output_shape = units

        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        node_attr = self.input_embedding(batch.x)
        u = self.glob_embedding(batch.u)
        edge_attr = self.edge_embedding(batch.edge_attr)

        for idx, layer in enumerate(self.conv_process):
            node_attr, edge_attr = layer(node_attr, edge_attr, batch.edge_index, batch.batch)

        x = torch_geometric.nn.global_mean_pool(node_attr, batch.batch)
        x = torch.cat([u, x], dim=-1)
        # x = fts

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        return x
