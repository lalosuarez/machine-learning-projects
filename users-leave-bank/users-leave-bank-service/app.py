from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# initalize our flask app
app = Flask(__name__)
# initialize these variables
global model, graph
sc = StandardScaler()

@app.route('/api/user/predict', methods=['POST','GET'])
def predict():
	print('Predicting if user will leave the bank...')
	# json = request.get_json(force=True)
	# name = request.json['name']
	# in our computation graph
	with graph.as_default():
		x = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
		# Feature Scaling
		x = sc.fit_transform(x)
		y_pred = model.predict(x)
		y_pred = (y_pred > 0.5)
		y_pred = bool(y_pred[0][0])
		print('y_pred = ' + str(y_pred))
		return jsonify({ 'will_leave_bank': y_pred }), 200

if __name__ == '__main__':
	print('Loading model...')
	model = load_model('users_leave_bank_optimized_model.h5')
	graph = tf.get_default_graph()
	print('Model loaded from disk')
	# decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	# run the app locally on the givn port
	app.run(host='0.0.0.0', port=port, debug=True)