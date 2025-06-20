import pandas as pd
df=pd.read_excel("dataset.xlsx")
print(df.head())
df['Student_ID']=range(1,len(df)+1)
print(df[['Student_ID']].head())
# Clean column names first
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("\u00ef\u00bb\u00bf", "")

# Check unique values in Gender
print("Unique Gender values BEFORE fixing:")
print(df['Gender'].unique())

# Standardize Gender values
df['Gender'] = df['Gender'].astype(str).str.strip()  # Remove spaces
df['Gender'] = df['Gender'].replace({
    'M': 'Male',
    'F': 'Female',
    '0': 'Male',
    '1': 'Female',
    'male': 'Male',
    'female': 'Female',
    'MALE': 'Male',
    'FEMALE': 'Female'
})

# Print cleaned values
print("\nUnique Gender values AFTER fixing:")
print(df['Gender'].value_counts(dropna=False))  # Show counts including NaN if any
df.to_excel("cleaned_dataset.xlsx", index=False)
#1. Descriptive Analysis
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("cleaned_dataset.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("\u00ef\u00bb\u00bf", "")

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Basic stats
print("\nData Description:\n", df.describe())
print("\nData Types:\n", df.dtypes)

# Distributions
plt.figure(figsize=(8, 4))
sns.histplot(df['Age_at_enrollment'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age at Enrollment')
plt.xlabel('Age at Enrollment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Unemployment_rate'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Count')
plt.show()
plt.savefig("my_plot.pdf")

# Correlation heatmap
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

#2 Diagnostic Analysis

# Boxplot: Age by Target
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Target', y='Age_at_enrollment', palette='Set2')
plt.title('Age at Enrollment by Target Group')
plt.show()

# Tuition Fee Status vs Target Outcome
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Tuition_fees_up_to_date', hue='Target', palette='pastel')
plt.title('Tuition Fee Status vs Target Outcome')
plt.show()

# Gender vs Dropout
plt.figure(figsize=(8, 4))
sns.countplot(data=df[df['Target'] == 'Dropout'], x='Gender', palette='Set3')
plt.title('Dropouts by Gender')
plt.show()

# Segment based on failed subjects
df['1st_sem_fail'] = df['Curricular_units_1st_sem_(enrolled)'] - df['Curricular_units_1st_sem_(approved)'] - df['Curricular_units_1st_sem_(without_evaluations)']
df['2nd_sem_fail'] = df['Curricular_units_2nd_sem_(enrolled)'] - df['Curricular_units_2nd_sem_(approved)'] - df['Curricular_units_2nd_sem_(without_evaluations)']
df['High_Failures'] = df['1st_sem_fail'] + df['2nd_sem_fail']

# Combine into total high failure
df['Failure_Segment'] = pd.cut(
    df['High_Failures'],
    bins=[-1, 2, 5, 10, 20],
    labels=['Low', 'Moderate', 'High', 'Very High']
)

# Group-wise dropout count
fail_seg = df[df['Target'] == 'Dropout'].groupby('Failure_Segment')['Student_ID'].count()
print("\nDropouts by Failure Segment:\n", fail_seg)

#3. Inferential / Predictive Analysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Step 3: Encode the target column
df['Target_encoded'] = df['Target'].map({
    'Dropout': 0,
    'Graduate': 1,
    'Enrolled': 2
})

# Step 4: Select features and target variable
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Target_encoded'])
y = df['Target_encoded']

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))