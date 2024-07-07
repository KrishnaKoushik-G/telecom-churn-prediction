## **Abstract**

Churn prediction is the process of identifying customers who are likely to discontinue a service. This is crucial in the telecom industry as it helps in retaining customers, thereby reducing revenue loss. In this project, we undertook a comprehensive analysis to predict customer churn in the telecom sector. We used data science techniques to obtain findings from the churn data. We began with a completely raw dataset from the telecom industry, which included over 100 numeric attributes indicating various customer characteristics and behaviors. Our first step was to preprocess the dataset by addressing issues such as missing values, duplicate entries, and outliers. We also performed dimensionality reduction techniques to streamline the dataset, ultimately obtaining a cleaner dataset with a reduced number of attributes. With the preprocessed data, we initially built a Decision Tree model to predict whether a customer would churn. To enhance the model's performance and prevent overfitting, we conducted hyperparameter tuning. Following this, we developed Logistic Regression and Random Forest models, and also applied hyperparameter tuning procedures to optimize their performance. After building and fine-tuning these models, we evaluated their predictive accuracies and compared them and documented the findings. The findings from this project highlight the importance of preprocessing in handling real-world data and the impact of model selection and tuning on predictive accuracy.

## **1. Introduction**

Churn refers to the phenomenon where customers discontinue their subscription to a service. In the telecom industry, churn prediction involves identifying customers who are likely to leave, enabling companies to take proactive measures to retain them. This is particularly important because acquiring new customers is often more costly than retaining existing ones. Churn prediction works by analysing historical customer data to identify patterns and factors that signal potential churn of the customer. This process is crucial for the telecom sector as it helps in maintaining a stable customer base and optimizing marketing strategies. By effectively predicting churn, telecom companies can implement targeted interventions to enhance customer satisfaction and loyalty, thereby retaining their revenue streams and competitive edge. Here’s where the predictive/classification models of Machine Learning come into picture.  

### **1.1 Machine Learning in Churn Prediction**

Machine learning plays an important role in telecom churn prediction by utilizing advanced algorithms to analyse vast datasets and understand patterns indicating churn behavior. Techniques such as decision trees, logistic regression, random forests, support vector machine, Naive Bayes etc., are employed to build predictive models that accurately forecast customer churn. These models are trained on historical data and make predictions basing on the same. Additionally, machine learning enables continuous learning and adaptation, allowing telecom companies to update their models with new data, thereby maintaining their effectiveness in dynamic market conditions. This approach not only helps in retaining customers but also provides insights for improving overall customer experience and service offerings.  

### **The Data Science Process**


Figure 1: Data Science Process 

The process of collecting raw data from the real world, cleaning and preprocessing the data, analysing the data, building machine learning models on the preprocessed data and reporting or utilizing the findings is called data science. In this project on telecom churn prediction, we began by gathering datasets from various sources, checked, analysed them and finalised the dataset. The final dataset comprised 25,000 rows and 111 columns, with all attributes being numeric. Our initial focus was on preprocessing this large dataset to facilitate effective analysis and model building. The preprocessing phase started with an in-depth examination of the dataset to understand the dependent variable (target variable), which is a binary variable indicating whether a customer churns or not. We explored the data to get some insights about it. Then we moved on to checking for null values, duplicates, zero variance variables, unique value variables and handling them properly to ensure data quality. Handling outliers was a crucial step to mitigate their impact on the model's performance. We also addressed highly correlated variables and multicollinearity issues to enhance the robustness of our models. These steps resulted in a clean and well-structured dataset ready for analysis. 

With the preprocessed data, we proceeded to build three different models: a decision tree, logistic regression, and a random forest. The dataset was split into training and testing subsets, allowing us to train the models on one part and evaluate them on the other. Each model was trained using the training data, and we obtained various metrics such as accuracy, precision, recall, and F1 score to evaluate their performance on both the training and test sets. The decision tree model was particularly useful for its interpretability, while the logistic regression provided a probabilistic framework for churn prediction. The random forest model, known for its ensemble learning technique, offered high accuracy and robustness. Analysing the metrics helped us understand the strengths and weaknesses of each model. To further improve their performance, we performed hyperparameter tuning, which involved adjusting the models' parameters to find the optimal settings that minimize overfitting and enhance predictive accuracy. This process of tuning and evaluation ensures that we obtain optimized models. We can, in future, develop a data product like recommendation system that recommends the telecom service provider, to take necessary measures to retain the customers using these predictive models. 

The entire process is structured into four milestones. In Milestone 1, we focused on gathering and understanding the data, as well as acquiring the necessary tools for our analysis. We dedicated Milestone 2 to data preprocessing, where we cleaned and prepared the dataset for modelling. In Milestone 3, we built our initial decision tree model, evaluated, analysed and improved by hyperparameter tuning. Finally, Milestone 4 involved building additional models, specifically logistic regression and random forest, and comparing their performances. Each of these milestones will be discussed in detail in the subsequent sections of this document, providing an understanding of our methodology and findings. 

 
## **2. Milestone 1**

In this milestone we began the project with efforts towards gathering the data that is relevant to churn prediction from telecom industry, then, acquiring and understanding the tools required for the churn prediction on the data. We acquired a raw dataset from the telecom industry and finalised that the dataset will be used for modelling. 

### **2.1  Dataset Description**

The dataset is a Telecom churn dataset containing the churn information of customers along with the information that would determine the churn. The dataset consists of a total 25000 samples (rows) and 111 feature (columns). All the features of the dataset are numeric, i.e., either integer or float data types. 

Variable to be predicted: ‘target’, it is a binary variable containing [0,1] values where 0 represents that the customer churns (discontinues using the company’s service) and 1 represents that the customer doesn’t churns (continues using the company’s service). 

Other Attributes: There are 110 other features of the dataset out of which 80 are of float type and 30 are of integer type data. Features from this set of features will be used to predict ‘target’ as 0 or 1.

### **2.2  Tools and Technologies**

The following tools and technologies were installed and/or used in the project: 

Anaconda: We first installed Anaconda navigated and the required packages into our machine. It provides all the necessary tools and libraries for Python and R, making it ideal for large-scale data analysis and machine learning projects. 

Jupyter Notebook: Jupyter Notebook is used for writing the python code. It is an interactive computing environment that allows for the creation and sharing of documents containing live code, equations, visualizations, and narrative text. It is particularly useful for data cleaning, transformation, and visualization in Python. 

Python: Python is a versatile programming language widely used in data science and machine learning for its simplicity and extensive libraries. Its readability and comprehensive support make it ideal for developing complex machine learning models and data analysis workflows. 

Pandas: Pandas is a powerful data manipulation and analysis library for Python, providing data structures like DataFrames to handle large datasets efficiently. It is essential for preprocessing tasks, including data cleaning, merging, and aggregation. 

Scikit-Learn: Scikit-Learn is a robust machine learning library in Python that offers simple and efficient tools for data mining and data analysis. It supports various supervised and unsupervised learning algorithms, making it critical for building and evaluating churn prediction models. 

Matplotlib and Seaborn: Matplotlib and Seaborn are visualization libraries in Python. Matplotlib provides extensive 2D plotting capabilities, while Seaborn builds on Matplotlib to offer enhanced statistical visualizations. Both are invaluable for exploratory data analysis and presenting results visually. 

### **2.3  Exploratory Data Analysis**

Exploratory Data Analysis (EDA) serves as a fundamental step in the data analysis process, allowing researchers to gain insights and identify patterns. We import the necessary libraries and proceed to read, explore and understand the data. The data is read using pandas and the column name, head of the data are observed. There is not much information obtained till now, as already known all the variables are numeric, the shape of the data is (25000, 111), we try to get the statistics describing the data, but the data is large, and it is difficult to interpret easily. We then tried to understand the ‘target’ variable by considering it as a categorical variable and describing it. It is illustrated below in Table 1. However, we converted it back to integer type as required for building models. 

Table 1: summary statistics of ‘target’ 
|       | target  |
|-------|---------|
| count | 25000   |
| unique| 2       |
| top   | 0       |
| freq  | 17083   |


Figure 2: Histogram plots of two variables 

We then tried to get some insight by checking the variables that have the same value for more than 75% of the samples, found two such and plotted their frequency distribution using matplotlib’s histogram, illustrated in Figure 1. By this we concluded the first milestone proceeding to the data preprocessing in milestone 2. 

## **3. Milestone 2**

In milestone 2 we performed various steps of data preprocessing to curate a clean and consistent dataset that is ready for further analysis. Data preprocessing is a vital step in preparing raw data for analysis, involving various techniques to enhance data quality and usability. This phase typically includes tasks such as handling misclassified variables, handling missing values, removing duplicates, handling zero  variance variables, unique value variables, handling outliers, reducing the dimensionality by handling highly correlated variables and multicollinearity and transforming data into a suitable format for analysis.   

We began by checking for missing (Null) values and duplicates, we found none. Then proceeded to check for the variables with all the values different from each other, i.e., unique value variables, that do not provide any information to the machine learning model. There are no unique valu variables. We then checked for zero variance variables, which are typically the variables with more than 90% of values the same, and found none. 

### **3.1  Handling Outliers**

Outliers, or data points that significantly deviate from the rest of the dataset. These anomalies may arise due to various factors, including measurement errors, data entry mistakes, or genuine observations that do not conform to the overall pattern of the data. They can have a notable impact on statistical analysis and machine learning models if left unaddressed. Identifying and treating outliers is thus a critical aspect of data preprocessing. Various statistical methods and visualization techniques can aid in detecting outliers. One common approach is to use summary statistics such as mean, median, and standard deviation to identify observations that fall beyond a certain threshold from the mean. Box plots, histograms, and scatter plots are graphical tools that provide visual insights into the distribution of data and help in spotting anomalies. Boxplots of all the columns are analysed, two of those are shown in Figure 3 below. 

 

  

Figure 3: Boxplots of two columns 

Once outliers are detected, several strategies like removal, transformation, imputation, etc., can be employed to address them, depending on the nature of the data and the specific analytical goals. Sometimes removal introduces unduly bias the analysis or compromise the representativeness of the dataset. So, we consider the following methods to handle outliers: 

*Boxplot:* A common method for identifying outliers is using a boxplot, which visually represents the distribution of data. The interquartile range (IQR) is the range between the first quartile (Q1) and the third quartile (Q3).  

𝐼𝑄𝑅 = 𝑄3 − 𝑄1	(1) 

𝐿𝑜𝑤𝑒𝑟 𝑏𝑜𝑢𝑛𝑑 = 𝑄1 − (1.5 × 𝐼𝑄𝑅)	(2) 

𝑈𝑝𝑝𝑒𝑟 𝑏𝑜𝑢𝑛𝑑 = 𝑄3 + (1.5 × 𝐼𝑄𝑅)	(3) 

Data points outside these bounds are considered outliers and may be further investigated or treated. This method helps in identifying extreme values that might distort statistical analyses. 

*Standardization:* +/- 3 Sigma Approach: Standardization transforms data to have a mean of zero and a standard deviation of one, making it easier to identify outliers. In the +/- 3 Sigma approach, data points that lie beyond three standard deviations from the mean are considered outliers. The standardized value is calculated as: Z=X−μσ	(4) Where X is the data point, μ refers to the mean of variable X and σ is the standard deviation. Any data point with a standardized value below -3 or above +3 is considered an outlier. This method assumes a normal distribution and is useful for detecting extreme deviations from the mean. 

*Capping & Flooring:* Capping and flooring involves limiting the extreme values in a dataset to reduce the influence of outliers. This method sets a maximum (cap) and a minimum (floor) threshold, beyond which data points are adjusted. For example, if the cap is set at the 95th percentile and the floor at the 5th percentile, any data point above the 95th percentile is set to the 95th percentile value, and any data point below the 5th percentile is set to the 5th percentile value. This method helps in retaining most of the data while reducing the impact of extreme values, ensuring that the dataset remains representative without significant distortion from outliers. 

### **3.2  Highly Correlated variables and Multicollinearity**

Highly correlated variables are pairs of variables that exhibit a strong linear relationship, often indicated by a correlation coefficient (r) close to 1 or -1. In data analysis, retaining highly correlated variables can lead to redundancy and inefficiency, as they provide overlapping information. To address this, we calculate the correlation matrix and remove one of each pair of variables with a correlation above a certain threshold, typically set at |r| > 0.8 or 0.9.

Multicollinearity occurs when independent variables in a regression model are highly correlated, making it difficult to isolate the individual effect of each variable. This is quantified using the Variance Inflation Factor (VIF), which measures how much the variance of a regression coefficient is inflated due to multicollinearity. A VIF values 5, 10, 20 are commonly used as a threshold to indicate significant multicollinearity. Removing variables with high VIF values helps to improve the model's stability and interpretability, ensuring that the estimates of the coefficients are reliable and that the model's predictive power is not compromised by multicollinearity.

We trained a basic decision tree model by using each of the three outlier handling approaches and setting different thresholds for correlation and VIF, and determined the accuracy, checked for the number of columns left. The idea behind this is, retaining lesser number of columns and getting higher accuracy, for understanding the patterns more effectively and easy modelling. From those observations, we chose the 3-sigma approach and the thresholds for r and VIF as 0.8 and 20 respectively, as it gives consistent result with more accuracy retaining lesser number of columns. We used these thresholds and reduced the dimensionality of the dataset by removing columns with lesser importance, i.e., the columns with VIF>20 and one column from the columns with correlation(r)>0.8, as they both provide the same information to the model. After removing the highly correlated variables, we are left with 41 columns in the data and 10 columns get removed on handling multicollinearity and we are left with 30 float datatype independent columns and the target variable.
We then export the remaining columns and rows a csv file of the clean telecom dataset and proceed to further analysis by building models on the preprocessed dataset in the milestone 3.

## **4. Milestone 3**

We split the data into training and testing data, 80% and 20% respectively as X_train, X_test, y_train, y_test, with X denoting independent variables and y the ‘target’ variable and trained the Decision tree classifier on the training data. 

### **4.1 Decision Tree**

The decision tree algorithm is a supervised learning algorithm commonly used for classification tasks. In supervised learning, the algorithm is trained on a labeled dataset, meaning each data point is associated with a known outcome. The decision tree algorithm builds a tree-like structure where each internal node represents a feature, each branch represents a decision based on that feature, and each leaf node represents a class label. During training, the algorithm recursively splits the dataset into subsets based on the feature that best separates the data into distinct classes. Decision trees are popular due to their simplicity, interpretability, and ability to handle both numerical and categorical data. They can also handle non-linear relationships between features and the target variable, making them versatile for various classification tasks. 

*Algorithm:* 

1. If all examples in data have the same class: Return a leaf node with that class label
2. If no features left to split on or data is empty: Return a leaf node with the majority class label in the parent data
3. Else:
       (a) Select the best feature to split on using Entropy or Gini Index
       (b) Create a decision tree node based on the selected feature
       (c) For each value of the selected feature:
           i. Split the data into subsets based on the feature value
           ii. Recursively call DecisionTree on each subset
           iii. Attach the resulting subtree to the decision tree node
       (d) Return the decision tree node  

### **4.2  Splitting Criteria**

To build an effective decision tree, the algorithm needs a criterion to decide how to split the data at each node. Two commonly used criteria are entropy and the Gini index. 
*Entropy:* Entropy is a measure of impurity or disorder used in information theory. In the context of decision trees, entropy quantifies the randomness in the dataset. When a dataset is perfectly pure (i.e., all instances belong to one class), entropy is 0. When the dataset is evenly distributed among all classes, entropy is maximized. 
*Gini Index:* The Gini index, or Gini impurity, is another metric to measure the impurity of a dataset. It calculates the probability of incorrectly classifying a randomly chosen element if it was labeled according to the distribution of labels in the dataset. A Gini index of 0 indicates a pure dataset, while higher values indicate greater impurity. 
*Splitting the Data:* The decision tree algorithm uses these criteria to choose the best feature to split the data at each node. For a feature X that splits the datasetD into subsets D1, D2, ...., Dm, the algorithm computes the weighted sum of the impurity of the subsets. The feature that maximizes the information gain or minimizes the Gini index is selected for splitting. 

On training the decision tree classifier using gini index criterion (default) we got an accuracy of 100% on the training data (i.e., the data already known to the model) and a 72.2% accuracy on the testing data (i.e., the data not already seen by the model). A 100% train accuracy and a similar test accuracy of 71.5% is obtained using Entropy as criterion. The significant difference between train and test accuracies and a 100% accurate prediction of the training data indicates overfitting of the models. Hence, to reduce the overfitting and improve the models, we perform hyperparameter tuning. 

### **4.3  Hyperparameter Tuning**

Decision trees are intuitive and easy to interpret, as the decision process is explicitly shown in the tree structure. They can handle different types of data and capture complex relationships. However, they are prone to overfitting. Overfitting is when a model learns the training data too well, capturing unnecessary patterns, resulting in poor generalization to new, unseen data. Our model is one such overfitted model. So, we apply hyperparameter tuning using grid search library of scikit learn, setting a grid of parameters, i.e., different sets of values for: Maximum depth of the tree (max_depth), Minimum number of samples required to split an internal node (min_samples_split), Minimum number of samples required to be at a leaf node (min_samples_leaf), using which the decision tree is repeatedly trained using different combinations of the parameters. It systematically explores all possible parameter combinations to identify the optimal settings for the model. The parameters’ values combination which gives the best accuracy is obtained are used for the final model. The best parameters we obtained for both the entropy and gini index criteria are: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2} 

### **4.4  Evaluation**

For the evaluation of the trained models, the following metrics are used: 

True Positive (TP): The number of correctly predicted positive cases.  

True Negative (TN): The number of correctly predicted negative cases. 

False Positive (FP): The number of incorrectly predicted positive cases. 

False Negative (FN): The number of incorrectly predicted negative cases. 

Accuracy: Ratio of correct predictions to total predictions. 
Accuracy = (TP+TN)/(TP+FP+TN+FN)

Precision (P): Ratio of correctly predicted positive cases to all predicted positive cases. 
P = TP/(TP+FP)

Recall (R): Ratio of correctly predicted positive cases to all actual positive cases. 
R = TP/(TP+FN)

F1 Score: Harmonic mean of precision and recall. 
F1 score = (2×P×R)/(P+R)

The confusion matrices (with TP, FP, FN, TN) are visualized as a heatmap in Figure 4 below. 


Figure 4: Heatmaps – Decision Tree models 

The observation from these results is that there is only a very slight difference between the decision tree models using gini index and entropy as criteria. 

 
## **5. Milestone 4**

In this milestone we proceeded towards building other models of Logistic Regression and Random Forest, improve and evaluate those models.  

### **5.1  Logistic Regression**

Logistic regression is a widely used supervised learning algorithm ideal for binary classification tasks. Unlike linear regression which predicts continuous outcomes, logistic regression predicts the probability of a categorical outcome, specifically whether an event will occur or not. It is particularly useful in this scenario where the goal is to determine whether a customer will leave the service. The algorithm works by applying the logistic function, or sigmoid function, to a linear combination of the input features. The logistic function ensures that the output is a probability value between 0 and 1. The model is trained by finding the best-fitting parameters (coefficients) that minimize the difference between the predicted probabilities and the actual outcomes, often using a method called maximum likelihood estimation. 

*Algorithm:* 

1. Start by initializing with random weights.
2. Compute the linear combination of input features and weights, z = β0 + β1X1 + β2X2 +…+ βnXn
3. Apply sigmoid function to z to get the probability, P(y=1|X) =1/(1+e^−z)
4. Compute the log-loss
5. Update the weights to minimize the log-loss using gradient descent or other optimization algorithms. 

We trained the algorithm on the preprocessed dataset, by splitting it into train and test data, using python. On training the logistic regression model, we achieved the train and test accuracies of 78.21% and 78.96% respectively. To further improve the model, we perform hyperparameter tuning using grid search. We define a parameter grid with different sets of values for: Regularization term to prevent overfitting by penalizing large coefficients (penality), Inverse of regularization strength (C), Algorithm to use in the optimization problem (solver), Maximum number of iterations the solver will run to converge to a solution (max_iter) and run the grid search algorithm. We get the best parameters as:  

{'C': 10, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'saga'} 

The model is trained using these best parameters, evaluated using different metrics and the confusion matrix (heatmap) is shown in the Figure 5. 


Figure 5: Heatmap – Logistic Regression 

We observe that logistic regression also gives similar results to decision tree model. It is predicting the target = 0 slightly more correctly and target = 1 slightly less accurately than decision tree. Also, it is performing slightly less accurately on training data and slightly more accurately on test data compared to decision tree. 

### **5.2  Random Forest**

Random Forest is a robust and versatile supervised learning algorithm used for both classification and regression tasks. It operates by constructing multiple decision trees during training and outputting the mode of the classes for classification or mean prediction for regression of the individual trees. This ensemble method enhances the model's accuracy and robustness, making it ideal for complex datasets like the one in our telecom churn prediction.

*Algorithm:*
1. Randomly sample subsets of the training data with replacement. Simply make a subset by taking n random records and m features from the data set having k number of records.
2. For each subset, build a decision tree model using a random selection of features at each node. This helps in reducing correlation among trees.
3. Nodes are split based on criteria such as Gini impurity or entropy for classification, aiming to maximize information gain.
4. Combine the predictions from all trees to form the final output. For classification, use majority voting; for regression, compute the average prediction. 

We trained the algorithm on the preprocessed dataset, by splitting it into train and test data (80%-20%), using python. On training the random forest model, we achieved the train and test accuracies of 100% and 80% respectively. The significant difference between train and test accuracies and a 100% accurate prediction of the training data signals overfitting of the model. To further improve the model by reducing overfitting, we perform hyperparameter tuning using grid search.  

We define a parameter grid with different sets of values for: number of trees used for prediction (n_estimators), depth of trees (max_depth), and number of features considered for splitting (min_samples_split) and minimum samples at leaf node (min_samples_leaf). Proper tuning ensures the model generalizes well to unseen data, balancing bias and variance. The best parameters obtained are: {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 150} 

The model is trained using these best parameters, the confusion matrix (heatmap) is illustrated in the Figure 6. 



 Figure 6: Heatmap – Random Forest 

We observe that the random forest model outperforms other models in terms of accuracies. Compared to the decision tree model, it performs better in predicting target = 0 while, performing slightly less in predicting target = 1 values. And, compared to logistic regression model, it performs slightly less in predicting target=0 values and performs better in predicting target = 1 values. 

### **5.3  Comparision of models**

A comparison of these three models: Decision Tree (DT), Logistic Regression (LR), Random Forest (RF) is made in the table 8 below in terms of metrics. 

Table 2: Comparison of models results 
| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----|-----------|---|---|--------|---|---|--------|---|---|
|        | Train    | Test| 0         | 1 |   | 0      | 1 |   | 0      | 1 |   |
| DT     | 0.7975   |0.7988| 0.84      | 0.70 | | 0.88  | 0.63 | | 0.86  | 0.66 | |
| LR     | 0.7937   |0.8026| 0.82      | 0.74 | | 0.91  | 0.57 | | 0.86  | 0.64 | |
| RF     | 0.9414   |0.8128| 0.84      | 0.74 | | 0.90  | 0.61 | | 0.87  | 0.67 | |

Observations from the table: The Decision Tree (DT), shows strong precision and recall for class 0 (not churn), indicating its strength in predicting non-churners, but it struggles with lower recall for class 1 (churn). Logistic Regression (LR) exhibits moderate performance with test accuracy (80.26%) and highlights its strength in high recall for class 0 (0.91), suggesting it is efficient at identifying non-churners, though it has lower precision for class 1 (0.74), indicating more false negatives. Random Forest (RF) model built with many decision trees together excels in overall performance, showing the highest accuracy on the test set (81.28%) and robust precision, recall, and F1 scores for both classes, making it particularly effective in balancing predictions between churn and non-churn cases. Each model has its own strengths, Decision Tree is strong in non-churn prediction but weaker in churn, Logistic Regression excels in recall for non-churn but has a trade-off in precision for churn predictions and Random Forest is well-rounded compared to remaining two models. 
