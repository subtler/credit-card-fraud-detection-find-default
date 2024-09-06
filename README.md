# FindDefault (Prediction of Credit Card Fraud) - Capstone Project

## Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Although credit cards can be a convenient way to manage finances, they can also be risky. Credit card fraud is the unauthorized use of someone else’s credit card or credit card information to make purchases or withdraw cash. It is crucial that credit card companies can recognize fraudulent credit card transactions so that customers are not charged for items they did not purchase.

---

## About Credit Card Fraud Detection:
- **What is credit card fraud detection?**  
  Credit card fraud detection refers to the tools, methodologies, and techniques used by financial institutions to detect and prevent unauthorized transactions. With the increasing volume of payment card transactions, fraud detection systems have evolved into largely digitized and automated processes. These systems leverage **machine learning (ML)** and **artificial intelligence (AI)** to manage data analysis, predictive modeling, fraud alerts, and remediation when fraudulent transactions are detected.

- **Anomaly detection**  
  Anomaly detection identifies deviations from a normal pattern of behavior by analyzing massive amounts of data. This includes:
  - Purchase history
  - Location
  - Device ID
  - IP address
  - Payment amount
  - Transaction details
  
  Any transaction that falls outside the established “normal” behavior is flagged for review. If the anomaly detection tool uses **machine learning**, it continuously learns from new data, improving the model’s precision in fraud detection.

- **What can be an anomaly?**
  - A sudden increase in spending
  - Large ticket item purchases
  - Rapid series of transactions
  - Transactions with the same merchant
  - Transactions from unusual locations or foreign countries
  - Transactions at unusual times

---

## Project Introduction:
The dataset contains transactions made by European credit cardholders in September 2013. The dataset includes 284,807 transactions, of which only 492 are fraudulent (approximately 0.172%). The primary challenge of this project was the high imbalance in the data, as the positive class (fraud) represents only a small fraction of all transactions. The objective of this project was to build a machine learning model capable of accurately predicting whether a transaction is fraudulent.

---

## Project Outline:
- **Exploratory Data Analysis (EDA):** Analyze and understand the data to identify patterns and trends using descriptive statistics and visualizations.
- **Data Cleaning:** Check for data quality and handle any missing values or outliers.
- **Dealing with Imbalanced Data:** The dataset is highly imbalanced, so resampling techniques like **NearMiss Undersampling** and **SMOTETomek** were applied to balance the dataset.
- **Feature Engineering:** Transform features for improved model performance.
- **Model Training:** Split the data into training and test sets, and estimate model parameters on the training data.
- **Model Validation:** Evaluate the models on the test data to ensure they generalize well to new, unseen data.
- **Model Selection:** Select the most appropriate model based on performance metrics.
- **Model Deployment:** Deploy the best-performing model for production use.

---

## Project Work Overview:
Our dataset exhibited significant class imbalance, with the vast majority of transactions being non-fraudulent (99.82%). This presented a challenge for predictive modeling, as models tend to struggle with highly imbalanced data. To address this issue, we employed both **undersampling** and **oversampling** techniques to balance the dataset.

1. **Undersampling:**  
   Initially, we used the **NearMiss** technique to reduce the number of non-fraudulent transactions to match the number of fraudulent ones. This approach, however, did not yield satisfactory results as it led to a loss of valuable data. The drastic reduction of majority class instances resulted in poor model performance due to the limited dataset.

2. **Oversampling with SMOTETomek:**  
   To further address the imbalance, we applied the **SMOTETomek** method, which combines **SMOTE** to oversample the minority class and **Tomek links** to remove noisy data from the majority class. This approach led to a better-balanced dataset and allowed models to learn more effectively from the fraudulent transactions.

3. **Machine Learning Models:**  
   After preprocessing and balancing the dataset, we trained the following machine learning models:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **AdaBoost Classifier**
   - **XGBoost Classifier**

4. **Evaluation Metrics:**  
   We evaluated each model’s performance using various metrics, including **accuracy**, **precision**, **recall**, and **F1-score**. Additionally, hyperparameter tuning was applied to optimize the models’ performance, focusing particularly on the **Random Forest** model.

5. **Model Selection:**  
   Although **XGBoost** achieved the best performance across most metrics, including **ROC-AUC**, **Random Forest** was chosen for hyperparameter tuning due to its strong performance and the simplicity of the tuning process. **Random Forest** provided a robust balance between precision, recall, and accuracy, making it the best fit for this project.

---

## Evaluation Metrics and Results:

- **Logistic Regression:**  
  - **Accuracy:** 94.84%  
  - **ROC-AUC:** 0.9895  
  - **Precision (Fraudulent):** 97%, **Recall (Fraudulent):** 92%

- **Random Forest (Hyperparameter Tuned):**  
  - **Accuracy:** 95.28%  
  - **ROC-AUC:** 0.9942  
  - **Precision (Fraudulent):** 91%, **Recall (Fraudulent):** 91%

- **AdaBoost:**  
  - **Accuracy:** 94.28%  
  - **ROC-AUC:** 0.9909  
  - **Precision (Fraudulent):** 98%, **Recall (Fraudulent):** 90%

- **XGBoost:**  
  - **Accuracy:** 99.51%  
  - **ROC-AUC:** 0.9998  
  - **Precision (Fraudulent):** 99%, **Recall (Fraudulent):** 100%

---

## Conclusion:
To address the challenge of class imbalance, we applied both **undersampling** and **SMOTETomek** oversampling techniques. While **XGBoost** emerged as the top-performing model across multiple evaluation metrics, we selected **Random Forest** for hyperparameter tuning due to its robust performance and ease of tuning. The **Random Forest** model demonstrated high accuracy and a balanced approach to fraud detection, making it the most practical choice for this project.

---

## Future Work:
Future work could focus on using **anomaly detection techniques** like **Isolation Forests** and **Autoencoders** to further enhance the detection of fraudulent transactions. Additionally, exploring **deep learning models** like **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** could improve the detection of fraudulent patterns in large, sequential datasets. Techniques like **transfer learning** and **unsupervised learning** can also be employed to refine fraud detection models and improve real-time detection capabilities.
