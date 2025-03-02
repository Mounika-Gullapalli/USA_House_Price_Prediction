##USA House Price Prediction Using Regression

This project develops a predictive model for USA house prices using regression techniques on a Kaggle-sourced dataset. We leveraged Scala and Apache Spark to analyze real estate data and forecast prices based on key features, providing valuable insights for informed decision-making.

**Files in This Repository**

1.HousePricePrediction.scala - Contains the full implementation of the regression models.

2.Project_Report.pdf - A detailed report explaining the project, dataset, methodology, results, and insights.

3.Project_Presentation.pptx - A presentation summarizing the key findings and methodology.


Dataset: The dataset consists of real estate records across various locations in the USA, containing attributes such as:
Numerical Features: Number of bedrooms, bathrooms, lot size, house size, zip code, and price.
Categorical Features: Property status, city, and state.
Dataset Link: USA Real Estate Dataset

**Technologies Used**
1.Scala - For data processing and model development

2.Apache Spark - For distributed data handling and analysis

3.Machine Learning Models: Linear Regression, Random Forest, Gradient Boosting

**Results & Insights**

Random Forest performed best with an R² score of 0.86, outperforming Linear Regression and Gradient Boosting.

Feature importance analysis highlighted square footage and location as the major price determinants.

Future Work: Enhancing feature engineering and experimenting with ensemble learning techniques.
