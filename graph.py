import torch
from torch_geometric.nn import GATConv, HeteroConv
import torch.nn.functional as F


# 辅助函数来创建全连接的边索引
def create_full_connect_edge_index(num_nodes):
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    return torch.stack([row, col], dim=0)


# 辅助函数来创建单向连接的边索引
def create_single_connect_edge_index(num_nodes):
    company_nodes = torch.arange(num_nodes)  # 0, 1, 2, ..., num_nodes-1
    news_nodes = torch.arange(num_nodes)  # 同上，假设新闻节点有相同的索引

    # 创建一个从每个公司节点到对应新闻节点的边索引
    return torch.stack([news_nodes, company_nodes], dim=0)


class Graph(torch.nn.Module):
    def __init__(self, company_features_dimension, news_features_dimension, num_heads, dropout):
        super(Graph, self).__init__()

        # 定义HAN的5层
        self.layers = torch.nn.ModuleList([
            HeteroConv({
                ('news', 'impacts', 'company'): GATConv((company_features_dimension, company_features_dimension),
                                                        company_features_dimension, heads=3, concat=False,
                                                        dropout=dropout, add_self_loops=False),
                ('company', 'influences', 'company'): GATConv((company_features_dimension, company_features_dimension),
                                                              company_features_dimension, heads=3, concat=False,
                                                              dropout=dropout, add_self_loops=False),
                ('company', 'influenced_by', 'company'): GATConv(
                    (company_features_dimension, company_features_dimension),
                    company_features_dimension, heads=3, concat=False,
                    dropout=dropout, add_self_loops=False),
            }, aggr='mean') for _ in range(3)
        ])

    def forward(self, company_features, daily_news_features):

        # 打印company_features和news_features_list的形状和内容
        #print("company_features shape:", company_features.shape)
        #print("company_features content:", company_features)
        #print("news_features_list shape:", [nf.shape for nf in news_features_list])
        #print("news_features_list content:", news_features_list)

        for i, layer in enumerate(self.layers):
            if i < len(daily_news_features):
                single_day_news_features = daily_news_features[i].unsqueeze(1).repeat(1,1)
            node_features_dict = {
                'company': company_features,
                'news': single_day_news_features
            }

            edge_index_dict = {
                ('news', 'impacts', 'company'): create_single_connect_edge_index(15),
                ('company', 'influences', 'company'): create_full_connect_edge_index(15),
                ('company', 'influenced_by', 'company'): create_full_connect_edge_index(15)
            }

            node_features_dict = layer(node_features_dict, edge_index_dict)
            company_features = F.sigmoid(node_features_dict['company'])

        return company_features

