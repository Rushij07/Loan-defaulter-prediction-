Loan Defaulter Prediction System
Project Overview
This project aims to develop a machine learning model to predict loan defaulters, assisting financial institutions in minimizing risk and maximizing profits. The model identifies potential loan defaulters based on customer data, enabling proactive risk management.

Business Context
Problem Statement
A substantial proportion of customers are defaulting on loans, leading to significant financial risk.

Objectives
Minimize the number of loan defaulters.
Balance risk reduction with profit maximization.
Constraints
Achieve business, machine learning, and economic success criteria without compromising overall profitability.
Success Criteria
Business Success: Reduce loan defaulters by at least 10%.
ML Success: Attain a model accuracy of over 70%.
Economic Success: Save the bank more than $1.2 million due to a reduction in loan defaults.
Data
Source: Bank's internal database.
Sample Size: Data of 1000 customers.
Variables: Total of 17 (16 inputs like income, credit score; 1 output - default status).
Implementation
1. Data Collection
Data is collected from the bank's database, focusing on 17 key variables to predict loan default likelihood.

2. Data Preprocessing
Includes data cleaning and exploratory data analysis (EDA) to prepare the dataset for modeling.

3. Model Building
Experimentation with various models and hyperparameter tuning to find the optimal solution.

4. Evaluation
Evaluation based on accuracy, business impact, and economic criteria.

5. Deployment
The model is deployed using Flask for real-time predictions.

6. Monitoring & Maintenance
Continuous monitoring and periodic updates for model relevance and accuracy.

Local Setup & Running the Project
To run this project locally, follow these steps:

Prerequisites
Ensure you have Python installed on your system. Python libraries like Flask, Pandas, NumPy, and Scikit-learn are required.

Step 1: Clone the Repository
Clone the repository to your local machine:

bash
Copy code
git clone [URL of the repository]
Step 2: Install Dependencies
Navigate to the project directory and install the required Python libraries:

bash
Copy code
cd [project-directory]
pip install -r requirements.txt
Step 3: Run the Flask Application
Execute the following command to start the Flask server:

Copy code
python DT_Flask_app.py
Step 4: Access the Application
Open your web browser and go to http://127.0.0.1:5000/ to interact with the application.

Contributions
Feel free to fork the project, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
