from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Open the dataset
df = pd.read_csv("Dataset/iphi2802.csv", delimiter="\t")

# Create a dataframe with with the column named "text"
df_text = df["text"]

print("stop")
