import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#write the column names since the data doesnt have it
columnNames = [
    "Sample code Number",
    "Clump Thickness", 
    "Uniformity of Cell Size", 
    "Uniformity of Cell Shape", 
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class"              
    ]

#load the data
data = pd.read_csv("breastCancerWistconsinData.data", names = columnNames)

# Replace "?" in the data with NaN
data.replace("?", np.nan, inplace=True)

# Convert 'Bare Nuclei' column to numeric, forcing errors to NaN
data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'], errors='coerce')

# Fill missing values (NaNs) in the dataset with the median of the respective columns
data.fillna(data.median(), inplace=True)

#deleate un-needed data
data = data.drop("Sample code Number", axis = 1)

#split data into values and target
XData = data.iloc[:, :-1].values
yData = data.iloc[:, -1].values

#standardize the Xdata
scaler = StandardScaler()
XData = scaler.fit_transform(XData)

label_encoder = LabelEncoder()
yData = label_encoder.fit_transform(yData)
yData = yData.reshape(-1, 1)

#split data into Training and testing data
XTrain, XTest, yTrain, yTest = train_test_split(XData, yData, test_size = 0.2, random_state = 40)

#create the model
model = Sequential([
    Input(shape = (XTrain.shape[1],)),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(64, activation = "relu"),
    Dropout(0.5),
    Dense(32, activation = "relu"),
    Dense(1, activation = "sigmoid")
])

#compile the model
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

#create early stoping callback
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#train the model
history = model.fit(XTrain, yTrain, epochs = 50, batch_size = 32, validation_data = (XTest, yTest), callbacks=[earlyStopping])

#evaluate the data
loss, accuracy = model.evaluate(XTest, yTest)
print(f"accuracy: {accuracy}\nloss: {loss}")

# Plotting Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
