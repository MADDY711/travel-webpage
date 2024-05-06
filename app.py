from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from waitress import serve

app = Flask(__name__)

# Load the trained model
with open('ml.pkl', 'rb') as model_file:
    lr_model = pickle.load(model_file)



@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    mse = None
    predictions = None
    graph_path = None  # Define graph_path initially
    if request.method == 'POST':
        # Get user input
        n = int(request.form['month'])

        # Reading data from CSV file
        df = pd.read_csv('C:\\final project\Mock.csv ', usecols=['Sr no', 'total'])

        # Splitting the data into features (X) and target variable (y)
        X = df.drop('Sr no', axis=1)
        y = df['total']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n / 100, random_state=n)

        # Making predictions
        predictions = lr_model.predict(X_test)

        # Calculating Mean Squared Error
        mse = mean_squared_error(y_test, predictions)
        a=list(range(1,n+1))

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(a[-5:], predictions[-5:], marker='o', color='b', linestyle='-', linewidth=1, markersize=6)        
        plt.title("Crowd Comparison")
        plt.xlabel("Month")
        plt.ylabel("No. of Tourists")
        plt.grid(True)
        graph_path = 'static/graph.png'  # Save the graph as a PNG file in the static folder
        plt.savefig(graph_path)

        output = True

    return render_template('index.html', output=output, mse=mse, predictions=predictions, graph_path=graph_path)




if __name__ == '__main__':
   app.run(debug=True)
