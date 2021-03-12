import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from classification.cnn_preprocess import CNNPreprocess
import matplotlib.pyplot as plt

class CNN:
    def prepare_datasets(self, X, y, test_size, validation_size, should_add_axis=False):
        """
        Splits data into train, validation and test sets.

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

        # Create train, validation and test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        if should_add_axis:
            # add an axis to input sets
            X_train = X_train[..., np.newaxis] # 4-D array
            X_validation = X_validation[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def plot_history(self, history):
        """Plots accuracy/loss for training/validation set as a function of the epochs
            :param history: Training history of model
            :return:
        """

        fig, axs = plt.subplots(2)

        # create accuracy sublpot
        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy eval")

        # create error sublpot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error eval")

        plt.show()

    def build_model(self, input_shape):
        """
        Generates CNN model.

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

    def train_model(self, X, y, model_filename = "./cnn_model"):
        """
        Trains and saves a CNN model.

        Args:
            X (ndarray): Input data
            y (ndarray): Target value for input data
            model_filename (string): path and name of CNN model to be saved
        Returns:
            test_error: error of the the testing data split
            test_accuracy: accuracy of the testing data split
        """

        # Get train, validation, test splits
        X_train, X_validation, X_test, y_train, y_validation, y_test = self.prepare_datasets(X, y, 0.2, 0.25)

        # Create network
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        model = self.build_model(input_shape)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

        # Plot accuracy/error for training and validation
        self.plot_history(history)

        # Evaluate the model on the test set
        test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

        # Save the model
        model.save(model_filename)

        return test_error, test_accuracy

    def predict(self, filename, ensemble = False, model_location = "./cnn_model"):
        """
        Makes a prediction.

        Args:
            filename (string): path and name of file for prediction
        Returns:
            predicted_index: prediction (0 for normal, 1 for abnormal)
        """

        # Preprocess the file
        cnn_preprocess = CNNPreprocess()
        data = cnn_preprocess.preprocess_file(filename)
        X = np.array(data["values"])

        # Retrieve the model
        reconstructed_model = keras.models.load_model(model_location)

        # Perform prediction
        prediction = reconstructed_model.predict(X)
        
        if ensemble:
            return prediction

        # Get index with max value
        predicted_index = np.argmax(prediction, axis=1)

        return predicted_index[-1]