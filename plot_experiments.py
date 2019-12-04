from bokeh.plotting import figure, output_file, show
import pandas as pd



def plot_columns(df, cols, out_filename="plot.html", tooltips=None,
            colors=['blue', 'red', 'purple', 'black', 'yellow', 'green']):
    df = df.copy()
    col_vals = [tuple((str(df.iloc[i][col]) for col in cols)) for i in range(len(df))]
    aux_splits = [str(col_val).split(',') for col_val in col_vals]
    aux_splits = [[word.strip("()', ") for word in pair if word.strip("()', ")] for pair in aux_splits]
    df["Legend"] = [", ".join(split) for split in aux_splits]

    col_vals = sorted(list(set(col_vals)))
    col_vals = {col_vals[i] : colors[i] for i in range(len(col_vals))}
    df["Colors"] = [col_vals[tuple((str(df.iloc[i][col]) for col in cols))] for i in range(len(df))]

    output_file(out_filename)
    p = figure(title="Experiments", x_axis_label="Accuracy Mean", y_axis_label="Accuracy Std", tooltips=tooltips)
    p.circle(x="Binary Accuracy Mean", y="Binary Accuracy Std", color="Colors", legend_field="Legend", source=df,
            size=10, alpha=0.5)
    show(p)



df = pd.read_csv("all_exps_metrics.csv")
for col in df.columns:
    df.loc[df[col] == -1, col] = "Not Applicable"
    if (col == "# of Neurons per Layer"):
        df.loc[df[col] == 0, col] = "Not Applicable"

tooltips = [
    ("Extraction Type", "@{Extraction Type}"),
    ("Extraction Layer", "@{Extraction Layer}"),
    ("Distance Function Type", "@{Distance Function Type}"),
    ("Combination Function Level", "@{Combination Function Level}"),
    ("# of Neurons per Layer", "@{# of Neurons per Layer}"),
    ("# of Layers", "@{# of Layers}"),
    ("Binary Accuracy Mean", "@{Binary Accuracy Mean}"),
    ("Binary Accuracy Std", "@{Binary Accuracy Std}")]

#Transfer melhor
plot_columns(df, ["Extraction Type"], tooltips=tooltips, out_filename="extraction_type.html")

#Aparenta bem distribuido
df = df[df["Extraction Type"] == "Transfer"]
plot_columns(df, ["Distance Function Type"], tooltips=tooltips, out_filename="eucl_vs_cos.html")

#3 e 2 quase tudo alto, os outros distribuidos. 3 e 2 melhor
plot_columns(df[df["Extraction Type"] == "Transfer"], ["Combination Function Level"],
tooltips=tooltips, out_filename="levels.html")

#Todos bons, mas cosseno um pouco melhor
df = df[df["Combination Function Level"] >= 3]
plot_columns(df, ["Extraction Type", "Extraction Layer"], tooltips=tooltips, out_filename="layers.html")

#Melhores
df = df[df["Distance Function Type"] == "Cosine"]
plot_columns(df, ["Distance Function Type"], tooltips=tooltips, out_filename="tops.html")