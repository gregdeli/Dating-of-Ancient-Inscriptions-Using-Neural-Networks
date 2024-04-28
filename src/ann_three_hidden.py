import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

import preprocessing
import argparse


def build_model(hidden_layer_neurons):
    model = Sequential()
    # First Hidden layer
    model.add(Dense(hidden_layer_neurons, input_dim=1000, activation="relu"))
    # Second Hidden layer
    model.add(Dense((hidden_layer_neurons * 2), activation="relu"))
    # Third Hidden layer
    model.add(Dense((hidden_layer_neurons * 4), activation="relu"))
    model.add(Dense(1, activation="linear"))  # Output layer
    model.compile(
        loss=my_loss_function,
        optimizer=Adam(learning_rate=0.001),
        metrics=["RootMeanSquaredError"],
    )
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


def plot_results(history, fold_no):
    rmse = history.history["RootMeanSquaredError"]
    val_rmse = history.history["val_RootMeanSquaredError"]
    epochs = range(1, len(rmse) + 1)
    plt.figure()
    plt.plot(epochs, rmse, "y", label="Training RMSE")
    plt.plot(epochs, val_rmse, "r", label="Validation RMSE")
    plt.title("Training and validation RMSE, Three hidden layers")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"Plots/Three Hidden Layers/train_val_rmse_{fold_no}.png")


if __name__ == "__main__":
    # Δημιουργούμε ένα parser για να εισάγουμε τον αριθμό των νευρώνων του κρυφού επιπέδου στο cli
    parser = argparse.ArgumentParser(description="Neurons in the hidden layer")
    parser.add_argument("--neurons", type=int, required=True)
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
        test_Y = Y[test]

        # Create a model for each fold
        model = build_model(hidden_layer_neurons=args.neurons)
        history = model.fit(
            train_X, train_Y, verbose=1, epochs=10, validation_data=(test_X, test_Y)
        )

        loss, rmse = model.evaluate(test_X, test_Y, verbose=1)
        rmse_per_fold.append(rmse)

        plot_results(history, fold_no)

        fold_no += 1

    # Calculate and print the average RMSE
    print(f"\nAverage Validation RMSE: {sum(rmse_per_fold) / len(rmse_per_fold)}")
