import pandas as pd
import numpy as np
import os



experiments_dir = "experiments"
exps = [exp for exp in os.listdir(experiments_dir) if exp != "autoencoder"]
exps = sorted(exps)
df = pd.DataFrame()
for exp in exps:
    cur_df = os.path.join(experiments_dir, exp, "metrics.csv")
    if os.path.exists(cur_df):
        cur_df = pd.read_csv(cur_df)
        cols = list(cur_df.columns)
        cols.remove("Fold")
        cols.remove("Set")

        cur_df = cur_df[cur_df["Set"] == "Val"]
        cur_df = cur_df[cols]
        cur_df = cur_df.describe().loc[["mean", "std"], :]

        new_df = {}
        for col in cols:
            new_df.update({col+" Mean" : [cur_df.loc["mean", col]]})
            new_df.update({col+" Std" : [cur_df.loc["std", col]]})
        new_df = pd.DataFrame(new_df)

        exp_vals = exp.split('_')
        identifier = exp_vals[0]
        if (identifier == "eucl") or (identifier == "cos"):
            exp_datatype = "Transfer"
            exp_dist_type = "Euclidean" if (identifier == "eucl") else "Cosine"
            exp_extract_layer = exp_vals[1].split(':')[1]
            exp_comb_func = exp_vals[2].split(':')[1]
            exp_n_neur = exp_vals[3].split(':')[1]
            exp_n_lays = exp_vals[4].split(':')[1]
        elif (identifier == "autoencoder"):
            exp_datatype = "Auto-Encoder"
            exp_dist_type = "Euclidean" if (exp_vals[1] == "eucl") else "Cosine"
            exp_extract_layer = "-1"
            exp_comb_func = exp_vals[2].split(':')[1]
            exp_n_neur = exp_vals[3].split(':')[1]
            exp_n_lays = exp_vals[4].split(':')[1]
        else:
            exp_datatype = "Basic Transfer"
            exp_dist_type = "Cosine"
            exp_extract_layer = exp_vals[1].split(':')[1]
            exp_comb_func = "4"
            exp_n_neur = "-1"
            exp_n_lays = "-1"

        new_df.insert(0, "Extraction Type", [exp_datatype])
        new_df.insert(1, "Extraction Layer", [exp_extract_layer])
        new_df.insert(2, "Distance Function Type", [exp_dist_type])
        new_df.insert(3, "Combination Function Level", [exp_comb_func])
        new_df.insert(4, "# of Neurons per Layer", [exp_n_neur])
        new_df.insert(5, "# of Layers", [exp_n_lays])
        
        df = pd.concat([df, new_df], ignore_index=True)

df.to_csv("all_exps_metrics.csv", index=False)