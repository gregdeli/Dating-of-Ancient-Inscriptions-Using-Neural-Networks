import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold


def preprocess_data(path_to_dataset, max_features):
    # Open the dataset
    df = pd.read_csv("Dataset/iphi2802.csv", delimiter="\t")

    # Create dataframes with the input (text col) and output data (date_min, date_max cols)
    texts = df["text"]
    dates = df[["date_min", "date_max"]]

    # Fit and transform the text data
    # Δημιουργία sparse matrix με τα vectorized κείμενα χρησιμοποιώντας την μέθοδο tf-idf
    # Περιορίζω τον αριθμό των tokens σε max_features
    tfidf = TfidfVectorizer(max_features=max_features)
    text_features = tfidf.fit_transform(texts)

    # Κανονικοποίηση των ημερομηνιών στο πεδίο [0,1] με τη μέθοδο min-max
    minimum_date = dates.min().min()
    maximum_date = dates.max().max()
    for index, row in dates.iterrows():
        for col in dates.columns:
            dates.at[index, col] = (dates.at[index, col] - minimum_date) / (
                maximum_date - minimum_date
            )

    # Split data into training and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        text_features, dates, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = preprocess_data(
        "Dataset/iphi2802.csv", max_features=1000
    )
