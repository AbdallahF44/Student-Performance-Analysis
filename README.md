# Student Performance Analysis

A Python project for analyzing student data, cleaning the dataset, exploring the data, creating visualizations, and building a simple machine learning model to predict students' final GPA.

## Project Overview

The goal of this project is to analyze a student dataset and extract useful information about students, their departments, ages, GPA, attendance, and academic performance.

The project started as a basic data analysis project and was extended with statistics, exploratory data analysis, feature engineering, and a small machine learning model.

## Dataset

The dataset contains information about students, including:

- Student ID
- Name
- Email
- Age
- Department
- City
- GPA
- Student Status

For the machine learning part, a separate dataset was created containing:

- Study Hours
- Attendance
- Previous GPA
- Final GPA

## Data Cleaning

Several cleaning steps were applied to improve the quality of the data:

- Removed duplicate rows
- Removed unnecessary spaces from text values
- Standardized student names
- Standardized department names
- Converted Age and GPA to numeric values
- Handled missing Age values
- Handled missing GPA values
- Handled missing Department and City values
- Corrected invalid Age values
- Corrected invalid GPA values

## Data Analysis

The project answers several questions about the dataset, such as:

- Number of students
- Number of departments
- Average GPA
- Highest GPA
- Lowest GPA
- Average age
- Number of students in each department
- Average GPA for each department
- Highest GPA in each department
- Lowest GPA in each department
- Students with GPA above the average
- Students with high or low GPA

## Statistics and EDA

The project also includes statistical analysis using:

- Mean
- Median
- Mode
- Range
- Variance
- Standard Deviation
- Percentiles
- Quartiles
- IQR
- Outlier detection
- Correlation

Exploratory Data Analysis was used to better understand the distribution and relationships between variables.

## Feature Engineering

Several new features were created from the original data:

- Passed
- Status
- GPA Percentage
- Age Group
- GPA Category

Categorical features were also converted using One-Hot Encoding.

Feature scaling was tested using:

- StandardScaler
- MinMaxScaler

## Data Visualization

The project includes several visualizations:

- Number of students in each department
- Average GPA by department
- Department distribution
- GPA distribution
- Age distribution
- Age vs GPA
- GPA Box Plot
- Correlation Heatmap
- Actual GPA vs Predicted GPA

The generated charts are saved inside the `images` directory.

## Machine Learning

A small machine learning project was added to predict students' final GPA.

The features used were:

- Study Hours
- Attendance
- Previous GPA

The target was:

- Final GPA

The workflow was:

Data
→ Train/Test Split
→ Linear Regression
→ Prediction
→ Evaluation

The model was implemented using Scikit-learn.

## Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

A simple baseline prediction was also used to compare the model performance with a basic prediction strategy.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
```text
student-performance-analysis/
├── students.csv
├── analysis.py
├── images/
│   ├── students_per_department.png
│   ├── average_gpa.png
│   ├── departments_pie.png
│   ├── gpa_histogram.png
│   ├── age_histogram.png
│   ├── age_vs_gpa.png
│   ├── gpa_boxplot.png
│   ├── correlation_heatmap.png
│   └── actual_gpa_vs_predicted_gpa.png
├── students_final_report.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Findings

The analysis showed several patterns in the dataset, including differences in student counts between departments and differences in average GPA between departments.

The relationship between age and GPA was weak, meaning that age does not appear to have a strong relationship with GPA in this dataset.

The GPA distribution was also analyzed using standard deviation, quartiles, IQR, and box plots to identify possible outliers.

## What I Learned

Through this project I practiced working with real-world style tabular data and learned how to move from raw data to analysis and then to a simple machine learning model.

The project helped me practice Python, Pandas, NumPy, Matplotlib, Seaborn, statistics, feature engineering, and Scikit-learn in one complete workflow.
