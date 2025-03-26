#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('load_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]
    except ValueError:
        # Handle invalid input
        return render_template('result.html', prediction="Invalid input. Please enter numeric values.")
    
    # Predict the class
    prediction = model.predict([features])[0]

    # Map prediction to class names
    class_names = ['setosa', 'versicolor', 'virginica']
    result = class_names[prediction]

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




