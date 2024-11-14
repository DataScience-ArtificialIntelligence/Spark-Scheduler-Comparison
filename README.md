# Spark Scheduler Comparison and Algorithm Performance Analysis

This repository contains an analysis of different Spark schedulers and the performance of various algorithms implemented in Spark MLlib. We conducted experiments using different Spark schedulers including FIFO and Fair, and measured the execution time of algorithms such as FP Growth and Random Forest. Additionally, we incorporated hyperparameter tuning for these algorithms using grid search cross-validation.

## Overview

In distributed computing frameworks like Spark, efficient job scheduling is crucial for optimizing resource utilization and improving overall performance. Spark provides different schedulers, each with its own scheduling policies and strategies. Understanding how these schedulers impact job execution time and resource allocation can provide valuable insights for optimizing Spark applications.

Moreover, the choice of algorithm and its parameter settings can significantly affect the performance of machine learning tasks. In this analysis, we focused on popular algorithms available in Spark MLlib and evaluated their performance under different scheduling configurations.

## Project Architecture

![Architecture](https://github.com/Sai-Kartheek-Reddy/Spark-Scheduler-Comparison/blob/main/Arch.png)

## Experiments

### Spark Schedulers

1. **FIFO Scheduler**: The First-In-First-Out (FIFO) scheduler is the default scheduler in Spark. It schedules jobs in the order they are submitted to the cluster, without considering resource availability or job priority.
   
2. **Fair Scheduler**: The Fair Scheduler aims to provide fair distribution of resources among users and applications. It dynamically adjusts resource allocation based on the demand from different jobs or users.

### Algorithms

1. **FP Growth**: FP Growth is a frequent pattern mining algorithm used for mining frequent itemsets from transaction data efficiently. We evaluated its performance under different scheduling configurations.

2. **Random Forest**: Random Forest is an ensemble learning method used for classification and regression tasks. We measured its execution time and accuracy under various scheduling settings.

### Hyperparameter Tuning

Hyperparameter tuning is essential for optimizing the performance of machine learning models. We employed grid search cross-validation to find the best combination of hyperparameters for FP Growth and Random Forest algorithms.

### Conclusion and Future Work
This project compares Spark's FIFO and FAIR schedulers in hyperparameter tuning tasks for FP Growth and Random Forest algorithms. The FAIR scheduler consistently exhibits reduced execution times across different algorithms and cluster sizes, particularly as the number of worker nodes increases, underscoring its scalability and efficiency. Further study is needed to analyze the fair scheduler on a larger number of nodes in the Spark cluster. Additionally, efforts should be made to mitigate network overhead to obtain accurate results.
