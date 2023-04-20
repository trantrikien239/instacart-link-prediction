import torch
from torch_geometric.data import Data, HeteroData



def prep_dataset(df_edges, df_labels, user_features, prod_features):
    edge_index = torch.tensor(df_edges[["enc_user_id", "product_id"]].values, 
                                dtype=torch.long).t().contiguous()
    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(user_features.shape[0])
    data["prod"].node_id = torch.arange(prod_features.shape[0])


    # Add the node features and edge indices:
    data["user"].x = torch.tensor(user_features, dtype=torch.float32)
    data["prod"].x = torch.tensor(prod_features, dtype=torch.float32)
    data["user", "buy", "prod"].edge_index = edge_index

    data = T.ToUndirected()(data)
    data["user", "buy", "prod"].edge_label_index = torch.tensor(
        df_labels[["enc_user_id", "product_id"]].values, dtype=torch.long).t().contiguous()
    data["user", "buy", "prod"].edge_label = torch.tensor(
        df_labels["label"].values, dtype=torch.float32).t().contiguous()
    return data