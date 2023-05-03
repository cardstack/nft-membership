import pyspark
from pyspark.sql import SparkSession

conf = pyspark.SparkConf()
spark = SparkSession.builder.getOrCreate()

## spark data
dfs = spark.read.parquet("../data") # data Yashavi sent
dfs = dfs.drop(*['Score1', 'Score2', 'Score3', 'Score4', 'Score5', 'Score6']) # let's not be distracted by Score
# cols_to_drop = ["num_nftcontract", 'num_currency']
# dfs = dfs.drop(*cols_to_drop)

## pandas data
dfp = dfs.drop('buyer').select("*").toPandas() #'p' for pandas


feature_names = [
 'avg_spent',
 'number_txns',
#  'num_nftcontract',
#  'num_currency',
 'avg_duration',
 'frequency'
]