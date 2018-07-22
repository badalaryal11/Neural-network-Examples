from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

#import / load datasets
#Download the dataset and place it in your local working directory, the same as your python file. Save it with the file name:
#pima-indians-diabetes.csv
#You can now load the file directly using the NumPy function loadtxt(). 
#There are eight input variables and one output variable (the last column). 
#Once loaded we can split the dataset into input variables (X) and the output class variable (Y).
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
#2. Define model

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. Compile model

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#4. Fit models
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

#5. Evaluate model
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#6. Tie it all together
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

output :
#...
Epoch 145/150
768/768 [==============================] - 0s - loss: 0.5105 - acc: 0.7396
Epoch 146/150
768/768 [==============================] - 0s - loss: 0.4900 - acc: 0.7591
Epoch 147/150
768/768 [==============================] - 0s - loss: 0.4939 - acc: 0.7565
Epoch 148/150
768/768 [==============================] - 0s - loss: 0.4766 - acc: 0.7773
Epoch 149/150
768/768 [==============================] - 0s - loss: 0.4883 - acc: 0.7591
Epoch 150/150
768/768 [==============================] - 0s - loss: 0.4827 - acc: 0.7656
 32/768 [>.............................] - ETA: 0s
acc: 78.26%*//

7.Make prediction :

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

#Running this modified example now prints the predictions for each input pattern.
 #We could use these predictions directly in our application if needed
 #output of predictions
  #predicted value in the form of array


