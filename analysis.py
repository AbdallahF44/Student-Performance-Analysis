import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
df["city"] = df["city"].str.strip().str.title()
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
# df.to_csv("students_final_report.csv", index=False)

# Update

# Statistics
print("*" * 25)
print("Mean GPA:", df["gpa"].mean())
print("Median GPA:", df["gpa"].median())
print("Min GPA:", df["gpa"].min())
print("Max GPA:", df["gpa"].max())
print("Std GPA:", df["gpa"].std())

# EDA
print("*" * 25)
print("Number of students:", len(df))
departments = df.groupby("department")
print("Number of department:", len(departments))
print("*******\nMean GPA in each department:\n", departments["gpa"].mean())
print("*******\nMax GPA in each department:\n", departments["gpa"].max())
print("*******\nMin GPA in each department:\n", departments["gpa"].min())
print(
    "*******\nLargest number of students in department:",
    departments["gpa"].count().idxmax(),
)
print(
    "*******\nHighest GPA in department:",
    departments["gpa"].mean().idxmax(),
)
q1 = df["gpa"].quantile(0.25)
q3 = df["gpa"].quantile(0.75)
median = df["gpa"].median()
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
number_of_outliers = len(df[(df["gpa"] > upper) | (df["gpa"] < lower)])
print("Number of GPA Outliers:", number_of_outliers)
print(
    f"There is no outliers!"
    if number_of_outliers == 0
    else f"There is {number_of_outliers} outliers!"
)
correlation = df[["age", "gpa"]].corr()
print(correlation)
corr_matrix = df.corr(numeric_only=True)

corr_unstacked = corr_matrix.unstack()

corr_filtered = corr_unstacked[corr_unstacked < 1.0]

top_pair = corr_filtered.abs().idxmax()
top_corr_value = corr_matrix.loc[top_pair[0], top_pair[1]]

print(
    f"Strongest Correlation is between '{top_pair[0]}' and '{top_pair[1]}': {top_corr_value:.3f}"
)

# Visualization
print("*" * 25)
sns.boxplot(x=df["gpa"])
stats = {
    "Lower": lower,
    "Q1 (25%)": q1,
    "Median": median,
    "Q3 (75%)": q3,
    "Upper": upper,
}

for label, value in stats.items():
    plt.axvline(x=value, color="red", linestyle="--", alpha=0.7)

    plt.text(
        x=value,
        y=-0.38,  # موقع النص عمودياً (أعلى الصندوق قليلاً)
        s=f"{label}\n({value:.2f})",  # النص المكتوب: الاسم والقيمة
        horizontalalignment="center",  # محاذاة النص بالمنتصف
        fontsize=9,
        fontweight="bold",
        color="darkred",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.8
        ),  # خلفية بيضاء بسيطة للوضوح
    )

plt.title(f"GPA Box Plot with Exact Percentile Values (Median: {median:.2f})", pad=25)
plt.xlabel("GPA")
plt.tight_layout()
plt.savefig("images/gpa_boxplot.png")
# plt.show()
plt.close()


sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.savefig("images/correlation_heatmap.png")
# plt.show()
plt.close()


# Findings

# There is no relationship between GPA and Age
# Max number of students in CS department
# Highest GPA in Business department
# There is no outliers


# ML Mini Project

data = {
    "study_hours": [2, 3, 4, 5, 6, 2, 7, 8, 4, 9, 6, 3, 10, 5, 8],
    "attendance": [60, 65, 70, 75, 80, 55, 85, 90, 72, 95, 88, 62, 98, 78, 92],
    "previous_gpa": [
        2.2,
        2.5,
        2.8,
        3.0,
        3.1,
        2.0,
        3.3,
        3.5,
        2.9,
        3.6,
        3.2,
        2.4,
        3.7,
        3.0,
        3.4,
    ],
    "final_gpa": [
        2.3,
        2.6,
        2.9,
        3.1,
        3.3,
        2.1,
        3.5,
        3.7,
        3.0,
        3.8,
        3.4,
        2.5,
        3.9,
        3.2,
        3.6,
    ],
}
df = pd.DataFrame(data)
X = df[["study_hours", "attendance", "previous_gpa"]]
y = df["final_gpa"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("*" * 15)
print("Actual:")
print(X_test.values)
print(y_test.values)
print("Predictions:")
print(predictions)

print("*" * 15)
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

print("*" * 15)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

print("*" * 15)
r2 = r2_score(y_test, predictions)
print("R²:", r2)

print("*" * 15)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

new_data = pd.DataFrame(
    {"study_hours": [7, 10], "attendance": [85, 85], "previous_gpa": [3.2, 3.5]}
)
prediction = model.predict(new_data)
print("Predicted GPA:", prediction)

print("*" * 15)
baseline_prediction = y_train.mean()

baseline_predictions = [baseline_prediction] * len(y_test)


baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("Baseline MAE:", baseline_mae)
print("Model MAE:", mae)

# Visualization to ML

plt.scatter(y_test, predictions)

plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.savefig("images/actual_gpa_vs_predicted_gpa.png")
# plt.show()
