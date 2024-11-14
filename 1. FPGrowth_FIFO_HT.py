from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
import time 
import socket
import os
import random

# Set the Spark master URL
spark_master_url = "spark://master:7077"

# Create a SparkSession
start = time.time()
spark = SparkSession.builder \
    .appName("PFP Growth Example") \
    .master(spark_master_url) \
    .getOrCreate()

# Load your dataset as a DataFrame
data = spark.createDataFrame([
    (0, ["bread", "jelly", "peanut_butter"]),
    (1, ["bread", "peanut_butter"]),
    (2, ["bread", "milk", "peanut_butter"]),
    (3, ["beer", "bread"]),
    (4, ["beer", "milk"])
], ["id", "items"])

# Define the range for hyperparameters
min_support_range = [i / 10.0 for i in range(1, 10)]  # Range from 0.1 to 0.9
min_confidence_range = [i / 10.0 for i in range(1, 10)]  # Range from 0.1 to 0.9

# Define the number of random trials
num_trials = 5

# Initialize variables to store the best hyperparameters
best_min_support = None
best_min_confidence = None
best_accuracy = float('-inf')

# Perform randomized search
for _ in range(num_trials):
    # Randomly select hyperparameters
    random_min_support = random.choice(min_support_range)
    random_min_confidence = random.choice(min_confidence_range)

    # Configure and train the FP-growth model with the randomly selected hyperparameters
    fpGrowth = FPGrowth(itemsCol="items", minSupport=random_min_support, minConfidence=random_min_confidence)
    model = fpGrowth.fit(data)

    # Get accuracy (considering the number of frequent itemsets as a measure)
    accuracy = model.freqItemsets.count()

    # Check if the current model has better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_min_support = random_min_support
        best_min_confidence = random_min_confidence

# Configure and train the FP-growth model with the best hyperparameters
fpGrowth_best = FPGrowth(itemsCol="items", minSupport=best_min_support, minConfidence=best_min_confidence)
model_best = fpGrowth_best.fit(data)

# Display frequent itemsets
print("Frequent itemsets:")
model_best.freqItemsets.show()

# Display association rules
print("Association rules:")
model_best.associationRules.show()

# Transform the original dataset with the best model to generate predictions
print("Transformed:")
model_best.transform(data).show()

stop = time.time()

print("-------------------------------------------------")
print("Best hyperparameters:")
print("MinSupport:", best_min_support)
print("MinConfidence:", best_min_confidence)

print("-------------------------------------------------")
print("Total Time Taken (in Mins) : ", (stop-start)/60)
print("-------------------------------------------------")

# Get the IP address of the current node
ip_address = socket.gethostbyname(socket.gethostname())

# Define the file path
file_path = os.path.join("/home/saikartheek", "FP_Growth_Cluster_Node_Details.txt")

# Write the node information to the text file
with open(file_path, "a") as file:
    file.write(f"Node Information:\nIP Address: {ip_address}\nHyperparameters:\nMinSupport: {best_min_support}\nMinConfidence: {best_min_confidence}\n")

print(f"Node information saved to: {file_path}")

# Stop the SparkSession
spark.stop()

