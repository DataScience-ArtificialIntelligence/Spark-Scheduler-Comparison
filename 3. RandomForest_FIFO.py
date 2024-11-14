import time
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, DoubleType

# Initialize SparkSession with cluster path
spark = SparkSession.builder \
    .appName("RandomForestRegressorExample") \
    .master("spark://master:7077") \
    .getOrCreate()

# Hardcoded data (10 records)
data_schema = StructType([
    StructField("label", DoubleType(), True),
    StructField("feature1", DoubleType(), True),
    StructField("feature2", DoubleType(), True)
])

data_values = [(1.0, 0.1, 0.8),
               (2.0, 0.2, 0.7),
               (3.0, 0.3, 0.6),
               (4.0, 0.4, 0.5),
               (5.0, 0.5, 0.4),
               (6.0, 0.6, 0.3),
               (7.0, 0.7, 0.2),
               (8.0, 0.8, 0.1),
               (9.0, 0.9, 0.2),
               (10.0, 1.0, 0.3)]

data = spark.createDataFrame(data_values, schema=data_schema)

# Assemble features
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
data = assembler.transform(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (70% training, 30% testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Start timing
start_time = time.time()

# Train model. This also runs the indexer.
model = pipeline.fit(trainingData)

# End timing
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("-----------------------------------------------------------------")
print("Time taken for execution: {:.2f} seconds".format(execution_time))
print("-----------------------------------------------------------------")

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
print("-----------------------------------------------------------------")
predictions.select("prediction", "label", "features").show(5)
print("-----------------------------------------------------------------")

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("-----------------------------------------------------------------")
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
print("-----------------------------------------------------------------")

# Calculate R-squared value
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print("-----------------------------------------------------------------")
print("R-squared value on test data = %g" % r2)
print("-----------------------------------------------------------------")

# Get the trained RandomForest model
rfModel = model.stages[1]
print(rfModel)  # summary only

# Stop SparkSession
spark.stop()

