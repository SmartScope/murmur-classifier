import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# Replace this line with location of JSON data file
DATA_PATH = "/Users/manthanshah/Desktop/data.json"

def load_data(data_path):
    """
    Loads the data to be fed into the CNN.

    Args:
        data_path (string): the path at which the JSON data file is stored
    Returns:
        X (ndarray): the data values
        y (ndarray): the labels for the data
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["values"])
    y = np.array(data["labels"])

    return X, y

def prepare_datasets(test_size, validation_size, should_add_axis=False):
    """
    Loads data and splits it into train, validation and test sets.

    Args:
        test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
        validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
        should_add_axis (bool): If the data has depth one, set to True
    Returns:
        X_train (ndarray): Input training set
        X_validation (ndarray): Input validation set
        X_test (ndarray): Input test set
        y_train (ndarray): Target training set
        y_validation (ndarray): Target validation set
        y_test (ndarray): Target test set
    """

    # Load data
    X, y = load_data(DATA_PATH)

    # Create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    if should_add_axis:
        # add an axis to input sets
        X_train = X_train[..., np.newaxis] # 4-D array
        X_validation = X_validation[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    """
    Generates CNN model

    Args:
        input_shape (tuple): Shape of input set
    Returns:
        model: CNN model
    """

    # Build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(4, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(keras.layers.Dropout(0.25))

    # Flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(20, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    # Output layer
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    return model

# Get train, validation, test splits
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

# Create network
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the CNN
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=100)

# Evaluate the CNN on the test set
test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy on the test set is: {}".format(test_accuracy))