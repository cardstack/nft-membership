import os
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.ml.clustering import BisectingKMeans, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col


def feature_plot(df, bins=70):
    ncol = 2
    nrow = len(df.columns)
    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(10, 10),
    )
    fig.tight_layout()  # improves spacing between graphs
    for i, col_name in enumerate(df):
        sns.boxplot(ax=axs[i][0], x=col_name, data=df)
        sns.histplot(
            ax=axs[i][1], x=col_name, data=df, bins=bins
        )  # set bins =20 otherwise, will be very slow


def view_summary(df):
    print(df.describe())
    feature_plot(df)


standardize = lambda df: (df - df.mean()) / df.std()


def remove_outliers(df, lower_q=0.01, upper_q=0.99):
    for col in df:
        q_low = df[col].quantile(lower_q)
        q_hi = df[col].quantile(upper_q)
        df_without_outliers = df[(df[col] < q_hi) & (df[col] > q_low)]
        print(f"Remove outliers {lower_q}-{upper_q }")
        print(f"Before removal: {len(df)} rows")
        print(f"After removal: {len(df_without_outliers)} rows")
        return df_without_outliers


def log_transform(df):
    new_df = df.copy()
    non_negative_index = (df >= 0).apply(
        lambda x: all(x)
    )  # col_name <> boolean (if col has all > 0 values)
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for c in [
        c
        for c in new_df.columns
        if new_df[c].dtype in numerics and non_negative_index[c]
    ]:
        new_df[c] = np.log10(new_df[c])
    return new_df


def summarize_pairs(df):
    print(df.corr())
    g = sns.PairGrid(df)
    g.map_diag(sns.histplot, bins=20)
    g.map_offdiag(sns.scatterplot)

def build_path(parent,key,config):
    prefix = f"{parent}/model={config['model']}/k={str(config['k'])}/" #parent directories
    Path(prefix).mkdir(parents=True, exist_ok=True) #create directory if no exists
    path =  prefix + '-'.join(config['feature_names']) + key
    return path

def summary_plots(df, config, parent_path=None):
    new_df = df.copy()
    new_df = new_df[[*config['feature_names'], 'prediction']] 

    if parent_path is None:
        g = sns.PairGrid(new_df, hue='prediction',palette="husl") 
        g.map(sns.scatterplot)
        plt.legend(title='Clusters')
        path = build_path("plots", "_pairplot.png", config)
        g.figure.savefig(path)
        plt.clf()
        g2 = sns.countplot(data=new_df, x='prediction', palette='husl')
        plt.legend(title='Clusters')
        path = build_path("plots", "_countplot.png", config)
        g2.figure.savefig(path)
        plt.clf()
    else:
        g = sns.PairGrid(new_df, hue='prediction', palette="husl") 
        g.map(sns.scatterplot)
        plt.legend(title='Clusters')
        path = build_path("plots", "_pairplot.png", config)
        g.figure.savefig(parent_path + '/pairplot.png')
        plt.clf()
        g2 = sns.countplot(data=new_df, x='prediction', palette='husl')
        plt.legend(title='Clusters')
        g2.figure.savefig(parent_path+ '/count_plot.png')
        plt.clf()



    # plotting the cluster center (this is not very useful)
    # if model is not None:
    #     centers = model.clusterCenters()
    #     d = pd.DataFrame(centers, columns = feature_names) 
    #     d['prediction'] = 0 # needs prediction to be in index
    #     g.data = d
    #     g.map(sns.scatterplot, zorder=10, marker="s", s=4) 


def write_spark_file(dfs, config, path=None):
    if(path is None):
        path = build_path("model_data", ".parquet", config) # make sure the .parquet is not something.parquet
    dfs.write.parquet(path, mode="overwrite") #when we build data avoid overwrite



def run_models(dfs, configs =[]):
    run_results = []
    for config in configs:
        result = run_model(dfs,config)
        run_results.append(result)
    return run_results


def run_model(dfs, config, path=None):
    assemble=VectorAssembler(inputCols=config['feature_names'], outputCol='features')
    assembled_data=assemble.transform(dfs)

    #Standardisation
    # scaler = StandardScaler(inputCol='features', outputCol='scaled_features') #feel free to use std scalar
    scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features') #use min max scaler to normalise 0-1
    scaler_model = scaler.fit(assembled_data)
    data = scaler_model.transform(assembled_data)

    if config['model'] == "kmeans":
        Model = KMeans(featuresCol='scaled_features', k=config["k"],maxIter=20, initMode='k-means||', seed = 1)
    elif config['model'] == 'bisecting_kmeans':
        Model = BisectingKMeans(featuresCol='scaled_features', k=config["k"],maxIter=20, seed = 1)

    # Apply Gaussian Mixture Model clustering
    # K=3
    # gmm = GaussianMixture(k=K, featuresCol="scaled_features")
    # model = gmm.fit(data)
    # output = model.transform(data)
    else:
        raise Exception("model not recognised")

    model = Model.fit(data)
    output = model.transform(data)
    evaluator = ClusteringEvaluator()
    evaluator.setPredictionCol("prediction") #silhoutte score
    score = evaluator.evaluate(output)
    write_spark_file(output, config, path)
    return {
        "score": score,
        "config": config
    }

def get_model_folders(search_path):
    SEARCH_PATH = search_path 

    all_files = []

    for root, dirs, files in os.walk(SEARCH_PATH):
        for file in files:
            relativePath = os.path.relpath(root, SEARCH_PATH)
            if relativePath == ".":
                relativePath = ""
            all_files.append(
                (relativePath.count(os.path.sep),
                relativePath,
                file
                )
            )

    all_files.sort(reverse=True)

    out = []
    for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
        out.append(SEARCH_PATH+'/'+folder)
    return out


def to_config(file_path):
    path_components = os.path.normpath(file_path).split(os.path.sep)

    _, model_path, k_path, feature_path = path_components

    return {
        "model": model_path.split("=")[1],
        "k": int(k_path.split("=")[1]),
        "feature_names": feature_path.split(".")[0].split("-")
    }

def to_scaled_df(dfs_output, config):
    """
    takes scaled_features column and makes it out into a spark dataframe with columns of all the features
    """
    sc_df = dfs_output.withColumn("xs", vector_to_array("scaled_features")).select(['prediction']+[col("xs")[i].alias(x) for i, x in enumerate(config["feature_names"])])
    return sc_df


def print_types(dfs):
    for col in dfs.dtypes:
        print(col[0]+" , "+col[1])

    


def feature_plots(dfs_output, config):
    scaled_df = to_scaled_df(dfs_output, config)
    out_dfp = scaled_df.toPandas()
    out_dfp['prediction'] = pd.Categorical(out_dfp.prediction)
    p = out_dfp.melt(id_vars='prediction', var_name="features")
    g=sns.catplot(data=p, x='value', y='prediction', estimator="mean", hue='prediction',col='features', col_wrap=1, kind="bar")
    path = build_path("plots", "_featuremean.png", config)
    g.figure.savefig(path)
    plt.clf()