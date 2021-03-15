import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from classification.cnn_preprocess import CNNPreprocess
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

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
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        # 2nd conv layer
        model.add(keras.layers.Conv2D(4, (5, 5), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(keras.layers.Dropout(0.25))

        # Flatten output and feed it into dense layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(20, activation='relu', kernel_regularizer='l2'))
        model.add(keras.layers.Dropout(0.5))

        # Output layer
        model.add(keras.layers.Dense(2, activation='sigmoid'))

        return model
    
    def train_model(self, X, y, plot_history=False):
        # Create network
        input_shape = (X.shape[1], X.shape[2], X.shape[3])
        model = self.build_model(input_shape)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.0007)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Add early stopping callback to prevent overfitting
        # stop training if validation loss doesn't improve after 15 epochs 
        callback_es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                                    patience=20, restore_best_weights=False)

        # Train the model
        # Weigh positives (abnormal examples) more when training 
        history = model.fit(X, y, batch_size=32, epochs=200, validation_split=0.1, 
                            callbacks=[callback_es], class_weight={0: 1, 1: 5})
        
        if plot_history:
            # Plot accuracy/error for training and validation
            self.plot_history(history)
        
        return model

    def validate_model_kfold(self, X, y):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(7))
        reports = []
        for train, test in kfold.split(X, y):
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            
            model = self.train_model(X_train, y_train)

            # Generate report
            y_pred = np.argmax(model.predict(X_test), axis=1)
            report = classification_report(y_test, y_pred)
            reports.append(report)

        return reports
    
    
    def validate_model_train_test(self, X, y):
        # perform 80-20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = self.train_model(X_train, y_train)

        # Generate report
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred)

        return report


    def save_model(self, X, y, model_filename = "./cnn_model"):
        """
        Trains CNN model on entire dataset and saves it

        Args:
            X (ndarray): Input data
            y (ndarray): Target value for input data
            model_filename (string): path and name of CNN model to be saved
        Returns:
            test_error: error of the the testing data split
            test_accuracy: accuracy of the testing data split
        """
        
        model = self.train_model(X, y)
        # Save the model
        model.save(model_filename)

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