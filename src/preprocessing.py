from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("Dataset/iphi2802.csv", error_bad_lines=False)

# Extract the text from the "text" column into a new dataframe
texts_df = pd.DataFrame(df["text"])

# Display the first few rows of the new dataframe
print(texts_df.head())
