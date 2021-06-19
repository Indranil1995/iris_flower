from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Sepal_Length']
    data2 = request.form['Sepal_Width']
    data3 = request.form['Petal_Length']
    data4 = request.form['Petal_Width']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

