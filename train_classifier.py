import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_dict = pickle.load(open("./data.pickle", 'rb'))

# Pad sequences to a fixed length
max_length = max(len(seq) for seq in data_dict['data'])
padded_data = pad_sequences(data_dict['data'], maxlen=max_length, padding='post', truncating='post', dtype='float32')

# Convert padded sequences to NumPy array
data = np.array(padded_data)
labels = np.array(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.40, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()