import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, request, jsonify

def load_datasets():
    dataset1 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
    dataset2 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
    return dataset1, dataset2

# Call the function and unpack the returned values into separate variables
dataset1, dataset2 = load_datasets()

# Merge the two datasets based on common columns
data = pd.merge(dataset1, dataset2)

# Display the first 10 rows of the merged DataFrame
print(data.head(10))

# Print the column names in the merged DataFrame
print(data.columns)

# Check the summary statistics of the merged DataFrame to see if 'Mental Fitness' contains only NaNs
print(data.describe())

# Check the count of missing values in each column
print(data.isnull().sum())

dataset1.head()

dataset2.head()

data.isnull().sum()

data.head(10)

data.size,data.shape

data.set_axis(['Country','Code', 'Year', 'Schizophrenia', 'Bipolar_Disorder', 'Eating_Disorder',
               'Anxiety', 'Drug_Usage', 'Depression', 'Alcohol', 'Mental_Fitness'],
              axis='columns', inplace=True)

data.head(10)

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.plot()

sns.pairplot(data, corner=True)
plt.show()

print(data.columns)

mean = data['Mental_Fitness'].mean()
print(mean)

fig = px.pie(data, values='Mental_Fitness', names='Year')
fig.show()

fig = px.line(data, x="Year", y="Mental_Fitness", color='Country', markers=True, color_discrete_sequence=['red', 'blue'], template='plotly_dark')
fig.show()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in data.columns:
    if data[i].dtype=='object':
        data[i]=l.fit_transform(data[i])

data.shape

data.info()

from sklearn.model_selection import train_test_split
x = data.drop('Mental_Fitness',axis=1)
y = data['Mental_Fitness']  # Assigning the target variable
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2)

print("xtrain:", xtrain.shape)
print("xtest:", xtest.shape)
print("\nytrain:", ytrain.shape)
print("ytest:", ytest.shape)

def preprocess_data(dataset1, dataset2):
    # Your data preprocessing code here
    # Clean the data, handle missing values, merge the datasets, etc.

    # Step 1: Check for missing values and handle them (e.g., fill with mean, median, or drop rows/columns)
    dataset1 = dataset1.fillna(dataset1.mean())
    dataset2 = dataset2.fillna(dataset2.mean())

    # Step 2: Merge the datasets based on the common columns 'Entity', 'Code', and 'Year'
    common_columns = ['Entity', 'Code', 'Year']
    merged_data = pd.merge(dataset1, dataset2, on=common_columns, how='inner')

    # Step 3: Feature engineering (create new features if needed)
    merged_data['new_feature'] = merged_data['Entity'] + merged_data['Entity']

    # Step 4: Encode categorical variables (if any) into numerical form
    label_encoder = LabelEncoder()
    for col in merged_data.columns:
        if merged_data[col].dtype == 'object':
            merged_data[col] = label_encoder.fit_transform(merged_data[col])

    # Step 5: Exclude unnecessary columns
    X = merged_data.drop(columns=['new_feature'])  # Use 'new_feature' instead of 'Mental_Fitness'
    y = merged_data['new_feature']  # Use 'new_feature' as the target variable

    # Step 6: Standardize/normalize features (if needed) using MinMaxScaler or StandardScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Finally, return the preprocessed data
    preprocessed_data = X, y
    return preprocessed_data

print("Columns in dataset1:", dataset1.columns)
print("Columns in dataset2:", dataset2.columns)

def select_features(preprocessed_data):
    # Your feature selection/extraction code here
    X, y = preprocessed_data  # Unpack the preprocessed_data into X and y

    # For demonstration purposes, you can use all features in this example
    selected_features = X
    target_variable = y

    return selected_features, target_variable

def train_regression_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(),
        'Support Vector Regression': SVR(),
        'Gradient Boosting Regression': GradientBoostingRegressor(),
        'Neural Network Regression': MLPRegressor(max_iter=1000)
    }

    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models

def evaluate_regression_models(trained_models, X_test, y_test):
    results = {}
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {'MSE': mse, 'MAE': mae, 'R-squared': r2}
    return results

def plot_model_comparison(results):
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    mse_values = [result['MSE'] for result in results.values()]
    plt.bar(models, mse_values, color='skyblue')
    plt.xlabel('Regression Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of Regression Models')
    plt.xticks(rotation=45)
    plt.show()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_mental_fitness():
    data = request.get_json()
    # Preprocess the input data if necessary
    input_features = np.array([data['feature1'], data['feature2'], ..., data['featureN']]).reshape(1, -1)
    prediction = best_model.predict(input_features)[0]
    return jsonify({'mental_fitness_prediction': prediction})

if __name__ == '__main__':
    dataset1, dataset2 = load_datasets()
    preprocessed_data = preprocess_data(dataset1, dataset2)
    selected_features, target_variable = select_features(preprocessed_data)
    X = selected_features
    y = target_variable

X= data.drop("Mental_Fitness",axis=1)
y= data["Mental_Fitness"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

trained_models = train_regression_models(X_train, y_train)

results = evaluate_regression_models(trained_models, X_test, y_test)

best_model_name = min(results, key=lambda x: results[x]['MSE'])
best_model = trained_models[best_model_name]
plot_model_comparison(results)


# ... (Previous code)

app = Flask(__name__)

# Load and preprocess the datasets
dataset1, dataset2 = load_datasets()
preprocessed_data = preprocess_data(dataset1, dataset2)
selected_features, target_variable = select_features(preprocessed_data)
X = selected_features
y = target_variable

# Train regression models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
trained_models = train_regression_models(X_train, y_train)

# Evaluate and select the best model
results = evaluate_regression_models(trained_models, X_test, y_test)
best_model_name = min(results, key=lambda x: results[x]['MSE'])
best_model = trained_models[best_model_name]

# Plot model comparison (optional)
plot_model_comparison(results)

@app.route('/predict', methods=['POST'])
def predict_mental_fitness():
    data = request.get_json()
    # Preprocess the input data if necessary
    input_features = np.array([data['feature1'], data['feature2'], ..., data['featureN']]).reshape(1, -1)
    prediction = best_model.predict(input_features)[0]
    return jsonify({'mental_fitness_prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
