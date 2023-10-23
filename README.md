# Project Title: Customer Churn Prediction for the Customer Retention Department at Syriatel: A Data-Driven Analysis.

## Overview

## Project Objective
The primary objective of this project is to develop a predictive model to help the Customer Retention Department at SyriaTel reduce customer churn. By analyzing historical data and customer behaviors, we aim to build a model that can identify customers at risk of leaving the company. This model will enable the department to take proactive measures to retain high-risk customers, ultimately improving customer satisfaction and reducing revenue loss due to churn.

### Specific Goals

1. **Churn Prediction**: Build a classification model to predict customer churn based on historical data and customer behavior patterns.

2. **Model Performance**: Assess the performance of the predictive model using appropriate classification metrics. Understand how well the model can identify customers likely to churn.

3. **Feature Importance**: Determine which features (e.g., call patterns, customer service calls, international plan) are most influential in predicting churn. This information can guide the department in formulating retention strategies.

4. **Model Interpretation**: Provide insights into why the model makes certain predictions. Understand the factors contributing to churn risk and communicate these insights to the Customer Retention Department.

5. **Recommendations**: Offer actionable recommendations to the Customer Retention Department based on the model's findings. Suggest strategies for retaining at-risk customers and reducing churn.



## Installation and Setup

To set up this project on your local machine, you'll need to ensure you have the following software, libraries, and dependencies installed. We recommend using a virtual environment to manage your project-specific dependencies.

### Software Requirements

- Python (3.7+ recommended)

### Python Packages Used

You can install the required Python packages using `pip`. Open your terminal or command prompt and run the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Codes and Resources Used

- **Editor Used:** Jupyter Notebook.
- **Python Version:** Python 3.8.5.

### Python Packages Used

List the Python packages and libraries used for the project, categorizing them based on their purpose:

- **General Purpose:** Warnings.
- **Data Manipulation:** Packages for data handling, e.g., pandas, numpy.
- **Data Visualization:** Include packages for data visualization like seaborn, matplotlib.
- **Machine Learning:** List the machine learning packages used, such as scikit-learn, TensorFlow, etc.

## Data

### Source Data

- **Data Source**: The dataset used for this project is the "Churn in Telecom's dataset" sourced from Kaggle. You can find the dataset at [this Kaggle link](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). This dataset contains information related to customer interactions and behavior at SyriaTel.

### Dataset Size

- The dataset consists of 3333 rows and 21 columns.

### Data Fields

The dataset contains a total of 21 features, including both numerical and categorical variables. Some of the key fields include:

- `state` (object): The state in which the customer resides.
- `account length` (int64): The length or duration of the customer's account with SyriaTel.
- `area code` (int64): The area code associated with the customer's phone number.
- `phone number` (object): The customer's phone number (likely for identification purposes).
- `international plan` (object): Whether the customer has an international calling plan (categorical: "yes" or "no").
- `voice mail plan` (object): Whether the customer has a voicemail plan (categorical: "yes" or "no").
- `number vmail messages` (int64): The number of voicemail messages received by the customer.
- `total day minutes` (float64): Total minutes of usage during the day.
- `total day calls` (int64): Total number of calls made during the day.
- `total day charge` (float64): Total charges incurred during the day.
- `total eve minutes` (float64): Total minutes of usage during the evening.
- `total eve calls` (int64): Total number of calls made during the evening.
- `total eve charge` (float64): Total charges incurred during the evening.
- `total night minutes` (float64): Total minutes of usage during the night.
- `total night calls` (int64): Total number of calls made during the night.
- `total night charge` (float64): Total charges incurred during the night.
- `total intl minutes` (float64): Total minutes of international usage.
- `total intl calls` (int64): Total number of international calls.
- `total intl charge` (float64): Total charges incurred for international usage.
- `customer service calls` (int64): The number of customer service calls made by the customer.
- `churn` (bool): A binary flag indicating whether the customer churned ("True" or "False").

### Target Variable

The target variable for this project is "churn," which is a binary variable indicating whether a customer has churned (1) or not (0).


### Data Preprocessing

The following data preprocessing steps were performed on the "Churn in Syriatel's dataset":

1. **Data Cleaning:** An initial check for missing values in the dataset shows that there are no missing values. The dataset is complete, and there are no null entries.

2. **Label Encoding:** Categorical variables such as "International plan" and "Voice mail plan" were encoded into numerical values for machine learning model compatibility.

3. **Feature Scaling:** Numerical features were standardized using the StandardScaler from scikit-learn to ensure that they were on the same scale.

4. **Train-Test Split:** The dataset was split into training and testing sets with a 70-30 split ratio to facilitate model evaluation.

5. **Handling Imbalanced Data:** Given the imbalance between churned and non-churned customers, class weighting techniques were implemented to address this issue. This approach was chosen due to the relatively small dataset, as oversampling or undersampling might have further reduced the dataset size.

6. **Feature Selection:** Feature importance techniques were applied to determine which features had the most impact on predicting customer churn. Columns like phone number were dropped from the dataset as they offered no predictive value.

These preprocessing steps were crucial in preparing the dataset for model development and analysis.


## Code Structure

The project repository is organized with the following structure:

- **data**: This directory contains the dataset used for the project. You can find the dataset files here.
  - `syriatel.csv`: The main dataset used for analysis.

- `index.ipynb`: The main notebook containing data preparation, exploratory analysis, model training, and evaluation.

- **Images/**: Stores various images and visualizations generated during the project.

    - **base_model_matrix.png**: Confusion matrix for the base logistic regression model.
    - **churn_distribution.png**: Churn distribution (bar plot).
    - **correlation_heatmap.png**: A correlation heatmap of all features.
    - **feature_importance.png**: A barplot showing feature importance across various models.
    - **important_feat_corr.png**: Plots showing the relationship between important features and the target variable.
    - **roc_curve.png**: ROC curve showing AUC for various class_weights.
    - **multiple_model_bar.png**: A bar chart showing crucial performance metrics of Decision Tree, Random Forest, Gradient Boosting, and SVM.
    - **numeric_features_histogram.png**: Histogram showing the distribution of numerical features.
  
- **Project_Presentation.pdf**: Access the comprehensive project presentation that provides an in-depth overview of our work, including model analysis, insights, and recommendations.

- **README.md**: The main README file containing an overview of the project, installation instructions, data sources, code structure, and more.

- **.gitignore**: The Git version control system configuration file to specify which files and directories should be ignored in the repository.




## Results and Evaluation

### Base Model Evaluation

#### Logistic Regression Model

- **Training Accuracy**: 0.8723
- **Testing Accuracy**: 0.862
- **Recall (Churned) on Testing Data**: 0.22
- **Precision (Churned) on Testing Data**: 0.54

##### Insights:

- While the accuracy of the Logistic Regression model is reasonable, its ability to recall churned customers is limited.
- Precision is moderate but comes at the cost of missing actual churn cases.

##### Recommendation:

- Suggests exploring other models and fine-tuning hyperparameters to enhance recall.

### Handling Class Imbalance

#### Balanced Class Weights

- Class weighting is introduced to handle class imbalance effectively.
- Different class weight settings were tested, with 'balanced' class weights achieving the best recall.

#### Logistic Regression Model with Balanced Class Weights

- **Accuracy**: 0.65
- **Recall**: 0.87
- **Precision**: 0.28

##### Insights:

- The Logistic Regression model with balanced class weights significantly improves recall.
- However, it results in lower precision, indicating a trade-off between precision and recall.

##### Recommendation:

- Consider the cost of false positives and negatives, aiming to proactively identify at-risk customers while maintaining reasonable precision.

### Advanced Model Evaluation and Insights

#### Decision Tree Model

- **Accuracy**: 94%
- **Recall**: 82%
- **Precision**: 79%

#### Random Forest Model

- **Accuracy**: 93%
- **Recall**: 73%
- **Precision**: 78%

#### Gradient Boosting Model

- **Accuracy**: 95%
- **Recall**: 73%
- **Precision**: 93%

#### Support Vector Machine (SVM) Model

- **Accuracy**: 75%
- **Recall**: 77%
- **Precision**: 34%

##### Insights:

- Gradient Boosting outperforms other models, achieving high accuracy and balance between precision and recall.
- Decision Tree offers high recall but with more false positives.
- Random Forest maintains a reasonable balance.
- SVM has high recall but lacks precision.

##### Recommendation:

- Gradient Boosting is the preferred choice for churn prediction. Further optimization is possible.

### Decision Tree vs. Gradient Boosting: A Trade-off Perspective

- Decision Tree offers high recall but increases false positives.
- Gradient Boosting balances precision and recall, minimizing false alarms.

#### Recommendation:

- The choice should align with the department's priorities and resource constraints.
- Consider a cost-benefit analysis to determine the best model for Syriatel's objectives and operational capacity.



## Future Work

### Model Tuning and Improvement

#### Logistic Regression Model

- Continue fine-tuning hyperparameters to achieve a better balance between precision and recall.
- Explore feature engineering and selection to enhance model performance.
- Investigate the impact of other regularization techniques on model performance.

#### Gradient Boosting Model

- Further optimize hyperparameters to improve overall model performance.
- Explore feature importance analysis to understand the key drivers of churn.
- Investigate ensemble techniques like XGBoost or LightGBM for potential enhancements.

### Exploring K-Nearest Neighbors (KNN)

- Consider incorporating K-Nearest Neighbors (KNN) as an alternative classification model.
- Evaluate KNN's performance in identifying at-risk customers.
- Tune KNN hyperparameters to achieve optimal results.

### Data Enhancement

- Gather more data if available to improve model generalization.
- Explore the incorporation of additional features that might provide valuable insights.
- Investigate external data sources to supplement the analysis and predictions.

### Advanced Model Interpretability

- Implement advanced model interpretation techniques to explain the predictions.
- Utilize SHAP (SHapley Additive exPlanations) values or LIME (Local Interpretable Model-Agnostic Explanations) to provide transparent insights into model predictions.

### Real-Time Deployment

- Develop a real-time prediction system for immediate churn risk assessment.
- Create a user-friendly interface for Syriatel's customer retention team to access and utilize the model.

### Cost-Benefit Analysis

- Conduct a thorough cost-benefit analysis to quantify the impact of churn prevention efforts.
- Assess the economic implications of implementing the model's recommendations and compare them to the cost of churn.

By focusing on these areas of future work, Syriatel's Customer Retention Department can further refine its customer churn prediction strategies and create more proactive and efficient retention plans.



## Acknowledgments/References

I would like to acknowledge the following references for their valuable insights and contributions to this project:

- "Machine learning pipelines and workflows" (Read the article [here](https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html)) - This article provided a foundational understanding of managing machine learning workflows and pipelines.

- Scikit-learn Documentation (Read the documentation [here](https://scikit-learn.org/stable/documentation.html)) - The official documentation of scikit-learn was an invaluable resource for understanding the algorithms used in this project.

- "The Elements of Statistical Learning: Data Mining, Inference, and Prediction" - A comprehensive reference material that offered deeper insights into statistical learning and data mining concepts.

I extend our gratitude to the authors and contributors of these references for sharing their knowledge and expertise, which greatly contributed to the success of this project.


## License

This project is released into the public domain and is available for use by anyone without any specific licensing restrictions. Feel free to use, modify, and distribute the code as needed.

