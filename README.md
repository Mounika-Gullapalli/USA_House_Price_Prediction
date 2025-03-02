USA House Price Prediction Using Regression
📌 Project Overview

This project develops a predictive model for USA house prices using regression techniques on a Kaggle-sourced dataset (Dataset Link). By leveraging Scala and Apache Spark, we analyze real estate data to forecast prices based on key features, providing valuable insights for informed decision-making.


📊 Dataset Description
The dataset contains real estate records across various locations in the USA, with attributes such as:

Numerical Features: Number of bedrooms, bathrooms, lot size, house size, zip code, and price.
Categorical Features: Property status, city, and state.
⚙️ Technologies Used
Scala for data processing and model development
Apache Spark for distributed data handling and analysis
Machine Learning: Implemented models include:
Linear Regression
Random Forest
Gradient Boosting
🏗️ How to Run the Code
Clone the repository:
sh
Copy
Edit
git clone https://github.com/your-username/USA_House_Price_Prediction.git
cd USA_House_Price_Prediction
Set up Apache Spark and Scala on your system.
Load the dataset (realtor_data.csv) into Spark.
Run the Scala scripts (HousePricePrediction.scala) in your Spark environment.
📈 Results & Insights
Random Forest performed best with an R² score of 0.86, outperforming Linear Regression and Gradient Boosting.
Feature importance analysis highlighted square footage and location as major price determinants.
Future work includes enhancing feature engineering and experimenting with ensemble learning techniques.
