# Breast Cancer Classification Using TensorFlow and Keras (Python)

## Description

This project builds a neural network to classify breast cancer tumors as benign or malignant using the Wisconsin Breast Cancer dataset. The model is trained using many cell-related data features to make a diagnosis.

## Dataset

The dataset consists of multiple cell characteristics, such as clump thickness, uniformity of cell size, mitoses, and so on. It includes missing values formatted as "?", which are handled by replacing them with the median of the respective columns.

The classified breast cancer tumors (target values) originally had two numerical labels:
- **Benign tumors (2)**
- **Malignant tumors (4)**

To allow the use of binary classification, **Label Encoding** was applied, converting the labels to:
- **0 for benign**
- **1 for malignant**

## Libraries Used

The following libraries were used in the development of this project:
- **Pandas**: For data process and loading
- **Numpy**: For handling data arrays
- **Matplotlib**: For plotting graphs
- **Scikit-learn**: For data preprocessing and model evaluation
- **TensorFlow/Keras**: For building and training the model

## Model Architecture

The model follows a fully connected neural network (MLP) approach with the following layers:
- **Input Layer**: Accepts data
- **Dense (128 neurons, ReLU activation)**: Learns patterns from the input
- **Dropout (0.5)**: Reduces overfitting by randomly dropping neurons
- **Dense (64 neurons, ReLU activation)**: Learns deeper patterns
- **Dropout (0.5)**: Reduces overfitting by randomly dropping neurons
- **Dense (32 neurons, ReLU activation)**: Learns deeper patterns
- **Dense (1 neuron, Sigmoid activation)**: Outputs a probability for binary classification

## Code Walkthrough

### Data Loading and Preprocessing
```python
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
```

- The dataset is loaded and assigned proper column names
- Missing values (represented by "?") are replaced with the column median
- Features are standardized using **StandardScaler()** to improve model performance
- **The target labels are encoded into binary values (0 and 1) for classification**
- Data is split into **80% training** and **20% testing**

### Create the Model
```python
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
```
The model is built using the **Sequential API** in TensorFlow, consisting of dense layers with **ReLU** activation and **Dropout** layers to prevent overfitting. The final layer uses a **sigmoid** activation for binary classification.

### Compile and Train
```python
#compile the model
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

#create early stoping callback
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#train the model
history = model.fit(XTrain, yTrain, epochs = 50, batch_size = 32, validation_data = (XTest, yTest), callbacks=[earlyStopping])
```

- The model is compiled with **binary cross-entropy loss** and the **Adam optimizer**
- **Early stopping** is used to stop training if validation loss does not improve for 10 epochs
- Training is conducted over **50 epochs** with a batch size of **32**

### Evaluation and Plot Graphs
```python
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
```

The model's performance is evaluated on the test dataset, measuring **accuracy** and **loss**. Use plt to plot an accuracy and a loss graph.

### Accuracy Graph
![Accuracy Graph](https://github.com/rhb140/Breast-Cancer-Classification-NN/blob/main/breastCancerDiagnosisImage5.jpg?raw=true)

A plot of training and validation accuracy over epochs.

### Loss Graph
![Loss Graph](https://github.com/rhb140/Breast-Cancer-Classification-NN/blob/main/breastCancerDiagnosisImage6.jpg?raw=true)

A plot of training and validation loss over epochs.

## Conclusion

The model effectively classifies breast cancer tumors with a **96%** accuracy. **Binary encoding of the target labels** allows the model to properly perform classification, and early stopping helps prevent overfitting.

### Author
Created by **Rory Howe-Borges**
[rhb140](https://github.com/rhb140)

## Citation
Dataset:
Badole, S. (2020). *Breast Cancer Wisconsin (State) Data Set*. Kaggle. Retrieved from [https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state](https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state)

