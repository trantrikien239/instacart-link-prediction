import pandas as pd
import numpy as np

def gen_neg_sample(test_edges, test_labels):
    hist_purchases = pd.concat([test_edges[["user_id", "product_id"]], test_labels[["user_id", "product_id"]]])
    print("test_edges.shape: ", test_edges.shape)
    print("test_labels.shape: ", test_labels.shape)
    print("hist_purchases.shape: ", hist_purchases.shape, " - Is total: ", test_edges.shape[0] + test_labels.shape[0] == hist_purchases.shape[0])
    # Extract all products
    all_products = hist_purchases["product_id"].unique()
    n_set_purchased = len(all_products)
    # Establish the skeleton
    df_gen = test_labels.groupby("user_id")["product_id"].count().to_frame()
    df_gen["existing"] = hist_purchases.groupby("user_id")["product_id"].apply(set)
    # Define function to apply to df_gen
    def gen_neg(df_gen):
        num_samp = df_gen["product_id"]
        existing = df_gen["existing"]
        i = num_samp
        existed_samp_idx = []
        neg_samp = []
        while i > 0:
            samp_idx = np.random.randint(0, n_set_purchased)
            if all_products[samp_idx] in existing or samp_idx in existed_samp_idx:
                continue
            else:
                existed_samp_idx.append(samp_idx)
                neg_samp.append(all_products[samp_idx])
                i -= 1
        return neg_samp
    # Generate neg_samples
    df_gen["neg_samp"] = df_gen.apply(gen_neg, axis=1)
    # post process
    df_neg_samp = df_gen[["neg_samp"]].reset_index().explode("neg_samp").reset_index(drop=True)
    df_neg_samp.columns = ["user_id", "product_id"]
    df_neg_samp["label"] = 0
    # Merge with test_labels
    output = pd.concat([test_labels, df_neg_samp]).sort_values("user_id").reset_index(drop=True)

    return output