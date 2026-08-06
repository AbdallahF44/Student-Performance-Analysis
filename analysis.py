import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Phase 1 - Data Loading
df = pd.read_csv("students.csv")

# Phase 2 - Data Inspection
print("\n***\n", df.head())
print("\n***\n", df.tail())
print("\n***\n", df.shape)
print("\n***\n", df.columns)
print("\n***\n", df.dtypes)
df.info()
print("\n***\n", df.describe())
print("\n***\n", df.isna().sum())
print("\n***\n", df.duplicated().sum())

# Phase 3 - Data Cleaning
df.drop_duplicates(inplace=True)
df["name"] = df["name"].str.strip().str.title()
columns = {
    "CS": "Computer Science",
    "AI": "Artificial Intelligence",
    "IT": "Information Technology",
}
df["department"] = df["department"].str.strip().str.upper().str.replace(".", "")
df["department"] = df["department"].replace(columns)
df["department"] = df["department"].str.title()
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")
df["age"] = df["age"].fillna(df["age"].mean()).astype(int)
df["gpa"] = df["gpa"].fillna(round(df["gpa"].median(), 2))
df["age"] = df["age"].apply(lambda x: int(df["age"].mean()) if x < 18 or x > 25 else x)
df["gpa"] = df["gpa"].apply(lambda x: -x if x < 0 else x if x < 4 else 4)
df["department"] = df["department"].fillna(df["department"].mode()[0])
df["city"] = df["city"].fillna(df["city"].mode()[0])

df.info()

# Phase 4 - Data Analysis
print("\n***\n", len(df))
print("\n***\n", len(df["department"].value_counts()))
print("\n***\n", df["age"].mean())
print("\n***\n", df["gpa"].mean())
print("\n***\n", df["gpa"].max())
print("\n***\n", df["gpa"].min())

print("\n***\n", df["department"].value_counts())
print("\n***\n", df.groupby("department")["gpa"].size())
print("\n***\n", df.groupby("department")["gpa"].mean())
print("\n***\n", df.groupby("department")["gpa"].max())
print("\n***\n", df.groupby("department")["gpa"].min())
print(
    "\n***\n", pd.pivot_table(df, "gpa", "department", aggfunc=["mean", "max", "min"])
)

print("\n***\n", df.sort_values("gpa").tail())
print("\n***\n", df.sort_values("gpa").head())
print("\n***\n", df[df["gpa"] > df["gpa"].mean()])
print("\n***\n", df[df["gpa"] >= 3.5])
print("\n***\n", df[df["gpa"] < 2.5])

# Phase 5 - Feature Engineering
df["passed"] = df["gpa"].apply(lambda x: True if x > 2.5 else False)
df["status"] = df["gpa"].apply(
    lambda x: "Excellent" if x >= 3.7 else "Good" if x >= 3.0 else "Weak"
)
df.insert(
    loc=df.columns.get_loc("name") + 1,
    column="email",
    value=df["name"].str.lower() + "@university.edu",
)
df["gpa_percentage"] = round(df["gpa"] * 25, 2)

print(df)

# Phase 6 - Visualization
counts = df["department"].value_counts()
plt.figure(figsize=(8, 5))
bars = plt.bar(
    counts.index,
    counts.values,
    0.5,
    color="#6fbcf3",
    edgecolor="black",
)
plt.bar_label(bars, fmt="%.0f")
plt.title("Number of Students in each Department")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/students_per_department.png")
# plt.show()
plt.close()

mean_gpa = df.groupby("department")["gpa"].mean()
plt.figure(figsize=(8, 5))
bars = plt.bar(
    mean_gpa.index,
    mean_gpa.values,
    0.5,
    color="#6ff38c",
    edgecolor="black",
)
plt.bar_label(bars, fmt="%.2f")
plt.title("Mean GPA by Department")
plt.xlabel("Department")
plt.ylabel("Mean GPA")
plt.ylim(2, 4)
plt.tight_layout()
plt.savefig("images/average_gpa.png")
# plt.show()
plt.close()

counts = df["department"].value_counts()
my_colors = ["#3333ff", "#3355ff", "#3388ff", "#33aaff"]
plt.figure(figsize=(8, 5))
plt.pie(counts, labels=counts.index, autopct="%1.0f%%", colors=my_colors)
plt.title("Number of Students in each Department")
plt.tight_layout()
plt.savefig("images/departments_pie.png")
# plt.show()
plt.close()

gpa = df["gpa"]
plt.figure(figsize=(8, 5))
plt.hist(
    gpa, bins=[2.0, 2.5, 3.0, 3.5, 4.0], rwidth=0.5, color="#5287ad", edgecolor="black"
)
plt.title("Distribution of GPA")
plt.xlabel("GPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/gpa_histogram.png")
# plt.show()
plt.close()

age = df["age"]
plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(age, rwidth=0.5, color="#7decaf", edgecolor="black")
plt.bar_label(patches, fmt="%d", padding=3, weight="bold")
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("images/age_histogram.png")
# plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(
    df["age"],
    df["gpa"],
    color="#9cbcdf",
    edgecolor="black",
)
plt.title("Age - GPA")
plt.xlabel("Age")
plt.ylabel("GPA")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("images/age_vs_gpa.png")
# plt.show()
plt.close()

# Phase 7 - Final Report
df.to_csv("students_final_report.csv", index=False)
