import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

import preprocessing


def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=1000, activation="relu"))  # Hidden layer
    model.add(Dense(1, activation="linear"))  # Output layer
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=0.001),
        metrics=["RootMeanSquaredError"],
    )
    return model


def plot_results(history):
    # Plot the training and validation rmse and loss at each epoch
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "y", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    rmse = history.history["root_mean_squared_error"]
    val_rmse = history.history["val_root_mean_squared_error"]
    plt.plot(epochs, rmse, "y", label="Training RMSE")
    plt.plot(epochs, val_rmse, "r", label="Validation RMSE")
    plt.title("Training and validation RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = preprocessing.preprocess_data(
        "Dataset/iphi2802.csv", max_features=1000
    )
    model = build_model()
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        verbose=1,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
    )
    _, rmse = model.evaluate(X_val, y_val)
    # print(f"Root Mean Squared Error: {rmse}")
    plot_results(history)
