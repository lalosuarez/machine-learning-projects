# -*- coding: utf-8 -*-

# Loading the model
from keras.models import load_model
new_classifier = load_model('users_leave_bank_optimized_model.h5')

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction_2 = new_classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction_2 = (new_prediction > 0.5)