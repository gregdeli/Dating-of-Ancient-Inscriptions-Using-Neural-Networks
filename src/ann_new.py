import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np

import preprocessing
import argparse


def build_model(
    hidden_layer_no, hidden_layer_neurons, learning_rate, momentum, r_in, r_h
):
    model = Sequential()
    if r_in != 0:
        model.add(Dropout(r_in, input_shape=(1000,)))  # Input Dropout layer
    # First hidden layer
    model.add(Dense(hidden_layer_neurons, input_dim=1000, activation="relu"))
    if r_h != 0:
        model.add(Dropout(r_h))  # Hidden Dropout layer
    if hidden_layer_no >= 2:
        # Second hidden layer
        model.add(Dense(int(hidden_layer_neurons * 2), activation="relu"))
    if hidden_layer_no == 3:
        # Third hidden layer
        model.add(Dense(int(hidden_layer_neurons * 4), activation="relu"))
    model.add(Dense(1, activation="linear"))  # Output layer
    model.compile(
        loss=my_loss_function,
        optimizer=Adam(learning_rate=learning_rate, beta_1=momentum),
    )
    model.summary()
    return model


def my_loss_function(y_true, y_pred):
    # every y value is of the form: y = [date_min, date_max]
    y_date_min, y_date_max = tf.unstack(y_true, axis=-1)
    # Αν το y_pred είναι μέσα στο διάστημα [y_date_min, y_date_max] τότε η απώλεια είναι 0,
    # αλλιως η απώλεια είναι η απόσταση απο το κοντινότερο άκρο του έυρους
    # Create a tensor with the same shape as y_pred, filled with zeros
    zero_loss = tf.zeros_like(y_pred)
    abs_distance_min = tf.abs(y_date_min - y_pred)
    abs_distance_max = tf.abs(y_date_max - y_pred)
    # Keep the smallest distance
    min_distance = tf.minimum(abs_distance_min, abs_distance_max)
    loss = tf.where(
        (y_pred >= y_date_min) & (y_pred <= y_date_max), zero_loss, min_distance
    )
    return loss


def plot_results(history, fold_no, hidden_layer_no):
    # rmse = history.history["RootMeanSquaredError"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "y", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(f"Training and Validation Loss, {hidden_layer_no} hidden layers")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"Plots/train_val_loss_fold_{fold_no}.png")


if __name__ == "__main__":
    # Δημιουργούμε έναν parser για να εισάγουμε μέσω του cli τον αριθμό των νευρώνων του κρυφού επιπέδου,
    # τον ρυθμό εκπαίδευσης και την σταθερά ορμής του μοντέλου
    parser = argparse.ArgumentParser(description="Train the neural network")
    parser.add_argument("--hidden_layer_no", type=int, default=1)
    parser.add_argument("--neurons", type=int, required=True)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("--r_in", type=float, default=0)
    parser.add_argument("--r_h", type=float, default=0)

    args = parser.parse_args()

    X, Y = preprocessing.preprocess_data("Dataset/iphi2802.csv", max_features=1000)

    # Cross-validation object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_no = 1
    rmse_per_fold = []  # Store the average RMSE for each fold
    for train, test in cv.split(X, Y):
        print(f"\nTraining for fold {fold_no} ...")
        train_X = X[train]
        test_X = X[test]
        train_Y = Y[train]
        test_Y_mean = np.mean(Y[test], axis=1)  # Μέσος όρος του κάθε έυρους ημερομηνιών
        test_Y_min_max = Y[test]

        # Create a model for each fold
        model = build_model(
            hidden_layer_no=args.hidden_layer_no,
            hidden_layer_neurons=args.neurons,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            r_in=args.r_in,
            r_h=args.r_h,
        )
        # Optional Early stopping
        if args.early_stopping:
            monitor = EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=1,
                restore_best_weights=True,
            )
            # Train the model
            history = model.fit(
                train_X,
                train_Y,
                verbose=1,
                epochs=100,
                validation_data=(test_X, test_Y_min_max),
                callbacks=[monitor],
            )
        else:
            # Train the model
            history = model.fit(
                train_X,
                train_Y,
                verbose=1,
                epochs=40,
                validation_data=(test_X, test_Y_min_max),
            )

        # RMSE ap to history
        # loss, rmse = model.evaluate(test_X, test_Y, verbose=1)
        # rmse = history.history["val_RootMeanSquaredError"]
        # average_val_rmse = sum(rmse) / len(rmse)
        # print(f"Average Validation RMSE for fold {fold_no}: {average_val_rmse}")
        # rmse_per_fold.append(average_val_rmse)

        rmse_metric = RootMeanSquaredError()
        rmse = rmse_metric(test_Y_mean, model.predict(test_X))
        rmse_per_fold.append(rmse)

        plot_results(history, fold_no, hidden_layer_no=args.hidden_layer_no)

        fold_no += 1

    # Calculate and print the average RMSE
    print(f"\nAverage Validation RMSE: {sum(rmse_per_fold) / len(rmse_per_fold)}")
