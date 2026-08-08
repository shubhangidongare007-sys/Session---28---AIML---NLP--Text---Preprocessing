# ============================================================
# SESSION 28 (AIML) - NLP TEXT PREPROCESSING
# Q1 TO Q10 COMPLETE ASSIGNMENT
# ============================================================

import pandas as pd
import string
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ============================================================
# STOPWORDS
# ============================================================

try:
    from nltk.corpus import stopwords
    import nltk

    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

except:
    stop_words = {
        "a", "an", "the", "is", "am", "are", "was", "were",
        "i", "me", "my", "we", "our", "you", "your", "he",
        "she", "it", "they", "them", "this", "that", "to",
        "of", "in", "on", "for", "and", "or", "but", "with",
        "very", "so", "as", "at", "be", "been", "from"
    }


# ============================================================
# Q1. LOADING THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("Q1. LOADING THE DATASET")
print("=" * 60)

try:
    df = pd.read_csv(
        "train.txt",
        sep=";",
        header=None,
        names=["text", "emotions"],
        encoding="utf-8"
    )

    # Remove empty rows
    df = df.dropna(subset=["text", "emotions"])

    # Remove completely blank text
    df["text"] = df["text"].astype(str).str.strip()
    df["emotions"] = df["emotions"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["emotions"] != "")
    ]

    df = df.reset_index(drop=True)

except Exception as e:
    print("train.txt could not be loaded.")
    df = pd.DataFrame(columns=["text", "emotions"])


# ------------------------------------------------------------
# If dataset is empty, use sample data
# ------------------------------------------------------------

if len(df) == 0:

    print("\nNo usable data found in train.txt.")
    print("Using sample emotion data so the assignment can run.\n")

    sample_data = [
        ["I feel very happy today", "joy"],
        ["I am feeling sad today", "sadness"],
        ["I am very angry right now", "anger"],
        ["I am scared of the dark", "fear"],
        ["I love my family very much", "love"],
        ["This is a very big surprise", "surprise"],
        ["Today I feel wonderful", "joy"],
        ["I feel lonely and sad", "sadness"],
        ["He made me very angry", "anger"],
        ["I am afraid of the results", "fear"],
        ["I love spending time with my friends", "love"],
        ["I did not expect this surprise", "surprise"],
        ["I am happy to see my friends", "joy"],
        ["I miss my friends", "sadness"],
        ["This situation makes me angry", "anger"],
        ["I am frightened by the noise", "fear"],
        ["I love beautiful flowers", "love"],
        ["I was shocked by the news", "surprise"],
        ["I am enjoying my day", "joy"],
        ["I feel unhappy today", "sadness"]
    ]

    df = pd.DataFrame(
        sample_data,
        columns=["text", "emotions"]
    )


print("Dataset loaded successfully!")

print("\nFirst 10 rows:")
print(df.head(10))

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# Q2. EXPLORING TARGET LABELS
# ============================================================

print("\n" + "=" * 60)
print("Q2. EXPLORING TARGET LABELS")
print("=" * 60)

unique_emotions = df["emotions"].unique()

print("\nUnique emotion labels:")
print(unique_emotions)

# Label Encoding
label_encoder = LabelEncoder()

df["emotion_encoded"] = label_encoder.fit_transform(
    df["emotions"]
)

# Mapping
emotion_mapping = dict(
    zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )
)

print("\nEmotion Mapping:")
print(emotion_mapping)

print("\nEncoded Data:")
print(
    df[["text", "emotions", "emotion_encoded"]].head(10)
)


# ============================================================
# Q3. LOWERCASE
# ============================================================

print("\n" + "=" * 60)
print("Q3. LOWERCASE")
print("=" * 60)

print("\nBefore Lowercasing:")

for i in range(min(5, len(df))):
    print(df["text"].iloc[i])

# Lowercase
df["text_lower"] = (
    df["text"]
    .astype(str)
    .str.lower()
)

print("\nAfter Lowercasing:")

for i in range(min(5, len(df))):
    print(df["text_lower"].iloc[i])

print("\nWhy lowercase is important in NLP:")

print(
    "Lowercasing converts all text into a common format. "
    "It reduces duplicate words caused by different capitalization "
    "and helps NLP models process text consistently."
)


# ============================================================
# Q4. REMOVING PUNCTUATION
# ============================================================

print("\n" + "=" * 60)
print("Q4. REMOVING PUNCTUATION")
print("=" * 60)


def remove_punctuation(text):
    return str(text).translate(
        str.maketrans("", "", string.punctuation)
    )


df["text_no_punctuation"] = (
    df["text_lower"]
    .apply(remove_punctuation)
)

print("\nBefore and After Punctuation Removal:\n")

for i in range(min(5, len(df))):

    print("Before:", df["text_lower"].iloc[i])

    print(
        "After :",
        df["text_no_punctuation"].iloc[i]
    )

    print()


# ============================================================
# Q5. REMOVING NUMBERS
# ============================================================

print("\n" + "=" * 60)
print("Q5. REMOVING NUMBERS")
print("=" * 60)


def remove_numbers(text):
    return re.sub(r"\d+", "", str(text))


df["text_no_numbers"] = (
    df["text_no_punctuation"]
    .apply(remove_numbers)
)

print("\nBefore and After Number Removal:\n")

for i in range(min(5, len(df))):

    print(
        "Before:",
        df["text_no_punctuation"].iloc[i]
    )

    print(
        "After :",
        df["text_no_numbers"].iloc[i]
    )

    print()


# ============================================================
# Q6. REMOVING EMOJIS & SPECIAL CHARACTERS
# ============================================================

print("\n" + "=" * 60)
print("Q6. REMOVING EMOJIS & SPECIAL CHARACTERS")
print("=" * 60)


def keep_ascii(text):

    return (
        str(text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


df["text_ascii"] = (
    df["text_no_numbers"]
    .apply(keep_ascii)
)

print("\nCleaned Text Samples:\n")

for i in range(min(5, len(df))):

    print(
        df["text_ascii"].iloc[i]
    )


# ============================================================
# Q7. REMOVING STOPWORDS
# ============================================================

print("\n" + "=" * 60)
print("Q7. REMOVING STOPWORDS")
print("=" * 60)


def remove_stopwords(text):

    words = str(text).split()

    cleaned_words = []

    for word in words:

        if word.lower() not in stop_words:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


df["text_no_stopwords"] = (
    df["text_ascii"]
    .apply(remove_stopwords)
)

print("\nBefore and After Stopword Removal:\n")

for i in range(min(5, len(df))):

    print(
        "Before:",
        df["text_ascii"].iloc[i]
    )

    print(
        "After :",
        df["text_no_stopwords"].iloc[i]
    )

    print()


# ============================================================
# Q8. COMPLETE TEXT CLEANING PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("Q8. COMPLETE TEXT CLEANING PIPELINE")
print("=" * 60)


def clean_text(text):

    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove punctuation
    text = remove_punctuation(text)

    # 3. Remove numbers
    text = remove_numbers(text)

    # 4. Remove emojis and special characters
    text = keep_ascii(text)

    # 5. Remove stopwords
    text = remove_stopwords(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_text"] = (
    df["text"]
    .apply(clean_text)
)

print("\nOriginal Text and Cleaned Text:\n")

for i in range(min(10, len(df))):

    print("Original:")
    print(df["text"].iloc[i])

    print("Cleaned:")
    print(df["cleaned_text"].iloc[i])

    print("-" * 60)


# ============================================================
# Q9. TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("Q9. TEXT LENGTH ANALYSIS")
print("=" * 60)


# Count words
df["text_length"] = (
    df["cleaned_text"]
    .apply(
        lambda x:
        len(str(x).strip().split())
        if str(x).strip()
        else 0
    )
)

print("\nText lengths:")

print(
    df[
        ["cleaned_text", "text_length"]
    ].head(10)
)


# Statistics

average_length = df["text_length"].mean()

minimum_length = df["text_length"].min()

maximum_length = df["text_length"].max()


print("\nAverage text length:",
      round(average_length, 2))

print("Minimum text length:",
      minimum_length)

print("Maximum text length:",
      maximum_length)


# ------------------------------------------------------------
# HISTOGRAM
# ------------------------------------------------------------

print("\nCreating Histogram...")

# Make sure there is data for graph
if len(df) > 0:

    min_value = int(df["text_length"].min())
    max_value = int(df["text_length"].max())

    # If all values are same
    if min_value == max_value:

        min_value = max(0, min_value - 1)
        max_value = max_value + 1

    bins = list(
        range(
            min_value,
            max_value + 2
        )
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        df["text_length"],
        bins=bins,
        edgecolor="black"
    )

    plt.xlabel(
        "Number of Words"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Histogram of Cleaned Text Lengths"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

else:

    print("No data available for histogram.")


print("\nObservation:")

print(
    "The histogram shows the distribution of "
    "the number of words present in the cleaned text. "
    "The cleaned dataset contains texts of different lengths."
)


# ============================================================
# Q10. MINI PROJECT - FULL PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("Q10. MINI PROJECT - FULL PREPROCESSING PIPELINE")
print("=" * 60)


# Save cleaned dataset

df.to_csv(
    "cleaned_emotions.csv",
    index=False
)

print(
    "\nCleaned dataset saved as:"
)

print(
    "cleaned_emotions.csv"
)


# Value counts

print(
    "\nValue Counts of Each Emotion:"
)

print(
    df["emotions"].value_counts()
)


print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED SUCCESSFULLY")
print("=" * 60)