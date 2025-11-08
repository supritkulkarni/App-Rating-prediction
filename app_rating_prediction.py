# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# LOAD CSV DATA USING PANDAS
df = pd.read_csv("googleplaystore.csv")

# CHECK FOR NULL VALUES
print("Null values per column:\n", df.isnull().sum())

# DROP ROWS WITH NULL VALUES
df.dropna(inplace=True)

# CORRECT FORMATTING AND TYPE ISSUES

# REMOVE NON-NUMERICAL DATA FROM SIZE COLUMN
df["Size"] = df["Size"].replace("Varies with device", np.nan)
df.dropna(subset=["Size"], inplace=True)

# FUNCTION TO CONVERT Mb INTO Kb
def convert_size(size):
    if "M" in size:
        return float(size.replace("M", "")) * 1000
    elif "k" in size:
        return float(size.replace("k", ""))
    else:
        return np.nan

df["Size"] = df["Size"].apply(convert_size)

# Reviews
df["Reviews"] = df["Reviews"].astype(int)

# Installs
df["Installs"] = df["Installs"].str.replace("[+,]", "", regex=True).astype(int)

# Price
df["Price"] = df["Price"].str.replace("$", "").astype(float)

# PERFORM SANITY CHECKS

# RATING BETWEEN 1 AND 5
df = df[df["Rating"].between(1, 5)]

# Reviews ≤ Installs
df = df[df["Reviews"] <= df["Installs"]]

# FREE APPS SHOULD HAVE Price = 0
df = df[~((df["Type"] == "Free") & (df["Price"] > 0))]

# UNIVARIATE ANALYSIS

plt.figure(num=1, figsize=(8, 4))
sns.boxplot(x=df["Price"])
plt.title("Figure 1: Boxplot of Price")
plt.xlabel("Price")
plt.show()

plt.figure(num=2, figsize=(8, 4))
sns.boxplot(x=df["Reviews"])
plt.title("Figure 2: Boxplot of Reviews")
plt.xlabel("Reviews")
plt.show()

plt.figure(num=3, figsize=(8, 4))
sns.histplot(df["Rating"], bins=20, kde=True)
plt.title("Figure 3: Histogram of Rating")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

plt.figure(num=4, figsize=(8, 4))
sns.histplot(df["Size"], bins=20, kde=True)
plt.title("Figure 4: Histogram of Size")
plt.xlabel("Size (KB)")
plt.ylabel("Count")
plt.show()

# OUTLIER TREATMENT

# DROP SAMPLES FOR Price > $200
df = df[df["Price"] <= 200]

# DROP REVIEWS GREATER THAN 2 MILLION
df = df[df["Reviews"] <= 2_000_000]

# INSTALLS : DROP ABOVE 99TH PERCENTILE
install_cutoff = df["Installs"].quantile(0.99)
df = df[df["Installs"] <= install_cutoff]

# BIVARIATE ANALYSIS

plt.figure(num=5, figsize=(8, 4))
sns.scatterplot(x="Price", y="Rating", data=df)
plt.title("Figure 5: Rating vs Price")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.show()

plt.figure(num=6, figsize=(8, 4))
sns.scatterplot(x="Size", y="Rating", data=df)
plt.title("Figure 6: Rating vs Size")
plt.xlabel("Size (KB)")
plt.ylabel("Rating")
plt.show()

plt.figure(num=7, figsize=(8, 4))
sns.scatterplot(x="Reviews", y="Rating", data=df)
plt.title("Figure 7: Rating vs Reviews")
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.show()

plt.figure(num=8, figsize=(10, 5))
sns.boxplot(x="Content Rating", y="Rating", data=df)
plt.title("Figure 8: Rating vs Content Rating")
plt.xlabel("Content Rating")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.show()

plt.figure(num=9, figsize=(16, 6))
sns.boxplot(x="Category", y="Rating", data=df)
plt.title("Figure 9: Rating vs Category")
plt.xlabel("Category")
plt.ylabel("Rating")
plt.xticks(rotation=90)
plt.show()

# DATA PRE-PROCESSING

inp1 = df.copy()

# PERFORM LOG TRANSFORM FOR RELATIVELY HIGH DATA POINTS
inp1["Reviews"] = np.log1p(inp1["Reviews"])
inp1["Installs"] = np.log1p(inp1["Installs"])

# DROP IRRELEVANT COLUMNS
inp1.drop(["App", "Last Updated", "Current Ver", "Android Ver"], axis=1, inplace=True)

# CREATE DUMMY VARIABLES (INCLUDING "Type")
inp2 = pd.get_dummies(inp1, columns=["Category", "Genres", "Content Rating", "Type"])

# SPLIT THE DATASET INTO TRAINING AND TESTING SUBSET (70-30)
df_train, df_test = train_test_split(inp2, test_size=0.3, random_state=42)

X_train = df_train.drop("Rating", axis=1)
y_train = df_train["Rating"]
X_test = df_test.drop("Rating", axis=1)
y_test = df_test["Rating"]

# BUILD MODEL USING LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)

# R² ON TRAINING
r2_train = r2_score(y_train, lr.predict(X_train))
print(f"R² on Train Set: {r2_train:.4f}")

# PERFORM PREDICTION AND EVALUATION
y_pred = lr.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print(f"R² on Test Set: {r2_test:.4f}")
