# Student Dropout Risk Analysis

This project aims to predict and analyze student dropout risk using a combination of Python-based machine learning and interactive Tableau dashboards.
The goal is to help educational institutions identify at-risk students early and take actionable steps to improve retention.


# Project Overview

- **Problem**: High dropout rates negatively affect student futures and institutional performance.
- **Solution**: Analyze student demographics, academic history, and financial data to build a predictive model and a dashboard for early intervention.
- **Tools Used**:
  - Python (Pandas, Seaborn, Matplotlib, Scikit-learn)
  - Tableau Public (for dashboarding)
  - Microsoft Excel (`Predict students' dropout and academic success` dataset from Kaggle)

---

##  Descriptive Analysis

- Cleaned and preprocessed missing data (numerical filled with median, categorical with mode).
- Visualized key distributions:
  - Age at enrollment
  - Unemployment rate
  - Tuition status vs Target
  - Gender vs Dropout
  - Correlation heatmap

---

## Diagnostic Insights

- Dropouts more likely:
  - To be older students
  - To have tuition dues
  - To have more failed subjects
  - To show lower academic performance

---

## Predictive Modeling

- **Model**: Random Forest Classifier
- **Target Encoded**:
  - Dropout: 0
  - Graduate: 1
  - Enrolled: 2

**Accuracy Achieved**: `~77%`
## Tableau Dashboard

**Dashboard Name**: `Student Dropout Risk Dashboard','Dashboard1','Dashboard2','Dashboard3'  
**Features**:
- Target outcome pie chart
- Tuition vs Target (stacked bar)
- Gender vs Dropout
- Application mode and course distribution
- Dropout segmentation by age and gender
- Filters: Gender, Fee Status, Target

 **View Dashboard Online**:  
 [Click to View Dashboard on Tableau Public](https://public.tableau.com/app/profile/naga.anusha.pusuluri/vizzes)

---

## Project Structure

```bash
dropout_project/
├── dropoutanalysis.py         # Python ML code
├── cleaned_dataset.xlsx       # Cleaned and processed dataset
├── Dropout_Report.pdf         # Full PDF report with analysis
├── StudentDropoutDashboard.twbx  # Tableau workbook file
├── README.md                  # Project documentation (this file)

