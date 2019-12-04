from bokeh.plotting import figure, output_file, show
import pandas as pd



df = pd.read_csv("all_exps_metrics.csv")
for col in df.columns:
    df.loc[df[col] == -1, col] ="Not Applicable"
    if (col == "# of Neurons per Layer"):
        df.loc[df[col] == 0, col] ="Not Applicable"

colors = ['blue', 'red', 'purple', 'orange', 'yellow', 'green']
tooltips = [
    ("Extraction Type", "@{Extraction Type}"),
    ("Extraction Layer", "@{Extraction Layer}"),
    ("Distance Function Type", "@{Distance Function Type}"),
    ("Combination Function Level", "@{Combination Function Level}"),
    ("# of Neurons per Layer", "@{# of Neurons per Layer}"),
    ("# of Layers", "@{# of Layers}"),
    ("Binary Accuracy Mean", "@{Binary Accuracy Mean}"),
    ("Binary Accuracy Std", "@{Binary Accuracy Std}")]

col = "Distance Function Type"
col_vals = sorted(df[col].unique())
col_vals = {col_vals[i] : colors[i] for i in range(len(col_vals))}
df["Colors"] = [col_vals[cur_val] for cur_val in df[col]]
output_file("by_extraction.html")
p = figure(title="Experiments", x_axis_label="Accuracy Mean", y_axis_label="Accuracy Std", tooltips=tooltips)
p.circle(x="Binary Accuracy Mean", y="Binary Accuracy Std", color="Colors", legend_field=col, source=df,
        size=10, alpha=0.5)
show(p)