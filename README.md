# Student Performance Analysis

This project is a simple data analysis application built with Python using Pandas, NumPy, and Matplotlib. The goal of the project is to analyze student performance by loading data from a CSV file, cleaning and preparing the dataset, performing statistical analysis, and creating visualizations to better understand the data.

The project includes data inspection, handling missing values, removing duplicate records, converting data types, standardizing department names, and generating useful statistics such as the average GPA, highest and lowest GPA, average age, top-performing students, and department-based summaries. It also creates several charts including students per department, average GPA by department, department distribution, GPA distribution, age distribution, and the relationship between age and GPA.

After the analysis is completed, the cleaned dataset is exported to a new CSV file, and all generated charts are saved in the images folder.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib

## Project Structure

```
student-performance-analysis/
│
├── students.csv
├── analysis.py
├── students_final_report.csv
├── images/
│   ├── students_per_department.png
│   ├── average_gpa.png
│   ├── departments_pie.png
│   ├── gpa_histogram.png
│   ├── age_histogram.png
│   └── age_vs_gpa.png
├── requirements.txt
└── README.md
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python analysis.py
```

## Output

The project generates a cleaned dataset named `students_final_report.csv` and saves all visualization charts inside the `images` folder.

## Author

Abdallah Fawzi
