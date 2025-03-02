// Databricks notebook source
// MAGIC %md
// MAGIC ####Data Analytics Project Code
// MAGIC
// MAGIC ######Authors: Mounika Gullapalli
// MAGIC ######Rajasekhar Gullapalli
// MAGIC
// MAGIC ####Objective: 
// MAGIC
// MAGIC In this project, we've developed a predictive model for USA house prices, employing regression techniques on a Kaggle-sourced dataset capturing real estate records over several months[1]. Using Scala and Apache Spark, we extract insights from this rich tapestry of data, forecasting prices based on key features. Our aim is to provide actionable insights for informed decision-making in the dynamic realm of real estate.

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession 
import org.apache.spark.sql.Column
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.evaluation.RegressionMetrics

val spark = SparkSession.builder.appName("House Price Prediction").getOrCreate()

// COMMAND ----------

//load the dataset
val df = spark.read.option("header", "true")
       .option("inferSchema","true")
       .csv("/FileStore/tables/realtor_data.csv") 
       
 df.show()

// COMMAND ----------

// Get the names of numerical attributes
val numericalAttributes = df.dtypes.filter(_._2 == "DoubleType").map(_._1)
println("Numerical Attributes:")
numericalAttributes.foreach(println)

// Get the names of categorical attributes
val categoricalAttributes = df.dtypes.filter(_._2 == "StringType").map(_._1)
println("\nCategorical Attributes:")
categoricalAttributes.foreach(println)

// COMMAND ----------

// defining a function to calculate the null values in each column of the dataset
def countCols(columns: Array[String]): Array[Column] = {
  columns.map(c => {
    count(when(col(c).isNull, c)).alias(c)
  })
  
}

// COMMAND ----------

//calculating the number of null values before the data preprocessing
df.select(countCols(df.columns): _*).show()

// COMMAND ----------

df.describe("price").show()

// COMMAND ----------

// Create a temporary view for the DataFrame
df.createOrReplaceTempView("df")

// COMMAND ----------

// MAGIC %python
// MAGIC
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC from pyspark.sql import SparkSession
// MAGIC
// MAGIC df = spark.sql("SELECT * FROM df")
// MAGIC
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame
// MAGIC pandas_df = df.select('price').toPandas()
// MAGIC
// MAGIC # Create a boxplot using seaborn and matplotlib
// MAGIC sns.boxplot(x='price', data=pandas_df)
// MAGIC plt.show()

// COMMAND ----------

// Drop rows with null values in the "price" column
val df_removed_price_null_values = df.na.drop("any", Array("price"))

// Calculate the outlier cutoff (quantile 0.9)
val outlierCutoff = df_removed_price_null_values.stat.approxQuantile("price", Array(0.9), 0.01)(0)

// Filter data for prices below the outlier cutoff
val df1 = df_removed_price_null_values.filter(col("price") < outlierCutoff)

// Show the summary statistics of the filtered data
df1.describe("price").show()

// COMMAND ----------

// Create a temporary view for the DataFrame
df1.createOrReplaceTempView("df1")

// COMMAND ----------

// MAGIC %python
// MAGIC
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC from pyspark.sql import SparkSession
// MAGIC
// MAGIC df1 = spark.sql("SELECT * FROM df1")
// MAGIC
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame
// MAGIC pandas_df = df1.select('price').toPandas()
// MAGIC
// MAGIC # Create a boxplot using seaborn and matplotlib
// MAGIC sns.boxplot(x='price', data=pandas_df)
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql.functions import col, concat_ws
// MAGIC
// MAGIC df = df1.groupBy('state').agg({'price': 'median'}).withColumnRenamed('median(price)', 'median_price')
// MAGIC
// MAGIC # Order by median price in descending order
// MAGIC df = df.orderBy(col('median_price').desc())
// MAGIC
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame for plotting
// MAGIC pandas_df = df.toPandas()
// MAGIC
// MAGIC # Create a barplot using seaborn
// MAGIC sns.barplot(data=pandas_df, x='median_price', y='state')
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql.functions import col, concat_ws
// MAGIC
// MAGIC # Group by 'zip_code' and 'state', calculate the median price, and sort in descending order
// MAGIC df = (
// MAGIC     df1.groupBy('zip_code', 'state')
// MAGIC     .agg({'price': 'median'})
// MAGIC     .withColumnRenamed('median(price)', 'median_price')
// MAGIC     .orderBy(col('median_price').desc())
// MAGIC     .limit(10)
// MAGIC     .withColumn('zip_code-state', concat_ws('-', col('zip_code').cast('int').cast('string'), col('state')))
// MAGIC     .select('zip_code-state', 'median_price')
// MAGIC )
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame for plotting
// MAGIC pandas_df = df.toPandas()
// MAGIC
// MAGIC # Create a barplot using seaborn
// MAGIC sns.barplot(data=pandas_df, x='median_price', y='zip_code-state')
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.functions import col,concat_ws
// MAGIC
// MAGIC # Group by 'city' and 'state', calculate the median price, and sort in descending order
// MAGIC df = (df1.groupBy('city', 'state')
// MAGIC     .agg({'price': 'median'})
// MAGIC     .withColumnRenamed('median(price)', 'median_price')
// MAGIC     .orderBy(col('median_price').desc())
// MAGIC     .limit(10)
// MAGIC     .withColumn('city-state', concat_ws('-', col('city'), col('state')))
// MAGIC     .select('city-state', 'median_price')
// MAGIC )
// MAGIC
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame for plotting
// MAGIC pandas_df = df.toPandas()
// MAGIC
// MAGIC # Create a barplot using seaborn
// MAGIC sns.barplot(data=pandas_df, x='median_price', y='city-state')
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql.functions import col, when
// MAGIC import seaborn as sns
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC # Create a new DataFrame with 'bed' values capped at 10
// MAGIC df = df1.withColumn('bed', col('bed').cast('int')).withColumn('bed', when(col('bed') >= 10, 10).otherwise(col('bed')))
// MAGIC
// MAGIC # Group by 'bed' and calculate the median price
// MAGIC df_price_bed = df.groupBy('bed').agg({'price': 'median'}).withColumnRenamed('median(price)', 'median_price')
// MAGIC
// MAGIC # Convert PySpark DataFrame to Pandas DataFrame for plotting
// MAGIC pandas_df_price_bed = df_price_bed.toPandas()
// MAGIC
// MAGIC # Create a barplot using seaborn
// MAGIC sns.barplot(data=pandas_df_price_bed, x='bed', y='median_price')
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql import SparkSession
// MAGIC from pyspark.sql import functions as F
// MAGIC import seaborn as sns
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC # Select relevant numerical columns for correlation calculation
// MAGIC numerical_columns = ['bed', 'bath', 'acre_lot', 'zip_code', 'house_size']  # Replace with your actual column names
// MAGIC selected_columns =  numerical_columns + ['price'] 
// MAGIC df_selected = df1.select(selected_columns)
// MAGIC
// MAGIC # Calculate the correlation matrix
// MAGIC corr_df = df_selected.toPandas().corr()
// MAGIC
// MAGIC # Plot the heatmap using Seaborn
// MAGIC plt.figure(figsize=(10, 8))
// MAGIC sns.heatmap(corr_df, annot=True, cmap='coolwarm')
// MAGIC plt.title("Correlation Matrix Heatmap")
// MAGIC plt.show()
// MAGIC

// COMMAND ----------

// Calculate mean and median
val bedMedian = df1.selectExpr("percentile_approx(bed, 0.5, 10000) as bed_median").first().getDouble(0)
val bathMedian = df1.selectExpr("percentile_approx(bath, 0.5, 10000) as bath_median").first().getDouble(0)
val acreLotMean = df1.selectExpr("round(avg(acre_lot)) as acre_lot_mean").first().getDouble(0)
val houseSizeMean = df1.selectExpr("round(avg(house_size)) as house_size_mean").first().getDouble(0)
val priceMean = df1.selectExpr("round(avg(price)) as price_mean").first().getDouble(0)

// Find the most frequent city
val mostFrequentCity = df1.groupBy("city").agg(count("*").alias("count"))
  .orderBy(col("count").desc)
  .select("city")
  .first()
  .getString(0)

val mostFrequentZipCode = df1.groupBy("zip_code")
  .agg(count("*").alias("count"))
  .orderBy(col("count").desc)
  .select("zip_code")
  .first()
  .getDouble(0)

//Fill the null values
val dfFilled = df1
  .na.fill(bedMedian, Seq("bed"))
  .na.fill(bathMedian, Seq("bath"))
  .na.fill(acreLotMean, Seq("acre_lot"))
  .na.fill(houseSizeMean, Seq("house_size"))
  .na.fill(priceMean, Seq("price"))
  .na.fill(mostFrequentCity, Seq("city"))
  .na.fill(mostFrequentZipCode,Seq("zip_code"))
  .drop("prev_sold_date")

// COMMAND ----------

dfFilled.select(countCols(dfFilled.columns): _*).show()

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

// Step 1: Index the categorical column
val indexer = new StringIndexer()
  .setInputCols(Array("status","city", "state"))
  .setOutputCols(Array("status_index","city_index", "state_index"))
  .setHandleInvalid("keep") 

val indexedDf = indexer.fit(dfFilled).transform(dfFilled)

// Step 2: One-hot encode the indexed columns
val oneHotEncoder = new OneHotEncoder()
  .setInputCols(Array("status_index","city_index", "state_index"))
  .setOutputCols(Array("status_vec","city_vec", "state_vec"))

val oneHotEncoderModel = oneHotEncoder.fit(indexedDf)
val encoded = oneHotEncoderModel.transform(indexedDf)

val finalDf = encoded.drop("status", "city", "state","status_index","city_index","state_index")

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.RegressionMetrics 
import org.apache.spark.sql.DataFrame

// choose the features
val featureCols = Array("bed", "bath","house_size","zip_code","status_vec", "city_vec", "state_vec")
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
val assembledDf = assembler.transform(finalDf.withColumnRenamed("price", "label"))
val assembleDf2 = assembledDf.select("features","label")

//Split the dataset into train and test data
val Array(trainData, testData) = assembledDf.randomSplit(Array(0.8,0.2), seed = 123)

// COMMAND ----------

//Inital linear regression
// Make predictions
val linearRegression = new LinearRegression()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Train models
val linearRegressionModel = linearRegression.fit(trainData)

// Make predictions
val predictionsLR = linearRegressionModel.transform(testData)

// Evaluate models
val evaluatorLR = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")

val mseLR2 = evaluatorLR.evaluate(predictionsLR)

// Calculate additional metrics using RegressionMetrics
val metricsLR = new RegressionMetrics(predictionsLR.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1))))
val rmseLR = metricsLR.rootMeanSquaredError
val maeLR = metricsLR.meanAbsoluteError
val r2LR = metricsLR.r2

println(s"Linear Regression RMSE: $rmseLR, MAE: $maeLR, R²: $r2LR")

// COMMAND ----------

// To Obtain the best model for the linear regression
// Define your Linear Regression model
val linearRegression = new LinearRegression()
  .setLabelCol("label")
  .setFeaturesCol("features")

// Define a grid of hyperparameters to search over
val paramGrid = new ParamGridBuilder()
  .addGrid(linearRegression.regParam, Array(0.01, 0.1, 1.0))  // Regularization parameter
  .addGrid(linearRegression.elasticNetParam, Array(0.0, 0.5, 1.0))  // Elastic Net mixing parameter
  .build()

// Define an evaluator
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mse")

// Define a CrossValidator
val crossValidator = new CrossValidator()
  .setEstimator(linearRegression)
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(evaluator)
  .setNumFolds(5)  // Number of folds in cross-validation

// Fit the model with the best hyperparameters
val bestModel = crossValidator.fit(trainData)

// Make predictions on the test set using the best model
val predictions = bestModel.transform(testData)

// Evaluate models
val evaluator2 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")

val mseLR2 = evaluator2.evaluate(predictions)

// Calculate additional metrics using RegressionMetrics
val metricsLR = new RegressionMetrics(predictions.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1))))
val rmseLR = metricsLR.rootMeanSquaredError
val maeLR = metricsLR.meanAbsoluteError
val r2LR = metricsLR.r2

println(s"Best model Linear Regression RMSE: $rmseLR, MAE: $maeLR, R²: $r2LR")

// COMMAND ----------

// Initialize random forest model with maxdepth 25 and NumTrees 15
val randomForestRegressor = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setMaxDepth(25).setNumTrees(15)

// Train models
val randomForestModel = randomForestRegressor.fit(trainData)

// Make predictions
val predictionsRF = randomForestModel.transform(testData)

// Evaluate models
val evaluatorRF = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")
val mseRF = evaluatorRF.evaluate(predictionsRF)

// Similar calculations for Random Forest and Gradient Boosting
val metricsRF = new RegressionMetrics(predictionsRF.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1))))
val rmseRF = metricsRF.rootMeanSquaredError
val maeRF = metricsRF.meanAbsoluteError
val r2RF = metricsRF.r2

println(s"Random Forest train- RMSE: $rmseRF, MAE: $maeRF, R²: $r2RF")

// COMMAND ----------

// Initialize models
val gradientBoostingRegressor = new GBTRegressor().setLabelCol("label").setFeaturesCol("features")

// Train models
val gradientBoostingModel = gradientBoostingRegressor.fit(trainData)

// Make predictions
val predictionsGB = gradientBoostingModel.transform(testData)

// Evaluate models
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("mse")

val mseGB = evaluator.evaluate(predictionsGB)

// Similar calculations for Random Forest and Gradient Boosting
val metricsGB = new RegressionMetrics(predictionsGB.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1))))
val rmseGB = metricsGB.rootMeanSquaredError
val maeGB = metricsGB.meanAbsoluteError
val r2GB = metricsGB.r2

println(s"Gradient Boosting RMSE: $rmseGB, MAE: $maeGB, R²: $r2GB")
