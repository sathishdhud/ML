from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

app = Flask(__name__)

# Global variables for models and datasets
knn_model = None
linear_model = None
knn_data = None
linear_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    global knn_model, knn_data
    accuracy = None
    try:
        if request.method == 'POST':
            file = request.files['dataset']
            knn_data = pd.read_csv(file)
            # Assuming the last column is the target
            X = knn_data.iloc[:, :-1]
            y = knn_data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_model = KNeighborsClassifier(n_neighbors=3)
            knn_model.fit(X_train, y_train)
            predictions = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_matrix = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions
            })
            return render_template('knn.html', accuracy=accuracy, accuracy_matrix=accuracy_matrix.to_html(classes='table table-striped'))
        return render_template('knn.html', accuracy=accuracy)
    except Exception as e:
        app.logger.error("Error in /knn route: %s", e)
        return str(e), 500

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    global knn_model
    if knn_model is not None:
        input_data = request.form.to_dict()
        input_data = pd.DataFrame([input_data])
        prediction = knn_model.predict(input_data)
        return render_template('knn.html', prediction=prediction[0], accuracy_matrix=None)
    return redirect(url_for('knn'))

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    global linear_model, linear_data
    r2_score = None
    try:
        if request.method == 'POST':
            file = request.files['dataset']
            linear_data = pd.read_csv(file)
            # Assuming the last column is the target
            X = linear_data.iloc[:, :-1]
            y = linear_data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            predictions = linear_model.predict(X_test)
            r2_score = linear_model.score(X_test, y_test)
            mse = mean_squared_error(y_test, predictions)
            accuracy_matrix = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions
            })
            return render_template('linear_regression.html', r2_score=r2_score, mse=mse, accuracy_matrix=accuracy_matrix.to_html(classes='table table-striped'))
        return render_template('linear_regression.html', r2_score=r2_score)
    except Exception as e:
        app.logger.error("Error in /linear_regression route: %s", e)
        return str(e), 500

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    global linear_model
    if linear_model is not None:
        input_data = request.form.to_dict()
        input_data = pd.DataFrame([input_data], dtype=float)
        prediction = linear_model.predict(input_data)
        return render_template('linear_regression.html', prediction=prediction[0], accuracy_matrix=None)
    return redirect(url_for('linear_regression'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
