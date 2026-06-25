# ============================================================
#  PHASE 1 FEATURE ALIGNMENT & DATASET STACKING
#  Aligns Dataset 3 columns to Dataset 1 format,
#  then stacks both into one clean training dataframe.
#  Output: combined_dataset.csv  (ready for Phase 2 modelling)
#
#  Requirements: pip install pandas numpy scikit-learn
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
PATH_DS1 = 'Students_Grading_Dataset_Biased.csv'
PATH_DS3 = 'student_performance.csv'
OUTPUT   = 'combined_dataset.csv'

print("=" * 55)
print("PHASE 1 FEATURE ALIGNMENT & STACKING")
print("=" * 55)

# ============================================================
# STEP 1 LOAD BOTH DATASETS
# ============================================================
print("\n[1] Loading datasets...")

df1 = pd.read_csv(PATH_DS1)
df3 = pd.read_csv(PATH_DS3)

print(f"  Dataset 1 shape : {df1.shape}")
print(f"  Dataset 3 shape : {df3.shape}")

# ============================================================
# STEP 2 CLEAN DATASET 1
# ============================================================
print("\n[2] Cleaning Dataset 1...")

# Auto-detect special-character column names
attend_col = next((c for c in df1.columns if 'attendance' in c.lower()), None)
stress_col = next((c for c in df1.columns if 'stress'     in c.lower()), None)

# Standardise column names
df1 = df1.rename(columns={
    attend_col : 'Attendance',
    stress_col : 'StressLevel_raw',
})

print(f"  Renamed '{attend_col}' → 'Attendance'")
print(f"  Renamed '{stress_col}' → 'StressLevel_raw'")

# Drop PII columns not needed for modelling
drop_cols = ['First_Name', 'Last_Name', 'Email', 'Student_ID']
df1 = df1.drop(columns=[c for c in drop_cols if c in df1.columns])
print(f"  Dropped PII columns: {drop_cols}")

# Handle missing values
print("\n  Missing before imputation:")
print(f"  {df1.isnull().sum()[df1.isnull().sum() > 0].to_dict()}")

# Numeric: fill with median
for col in ['Attendance', 'Assignments_Avg']:
    df1[col] = df1[col].fillna(df1[col].median())

# Categorical: fill with mode
df1['Parent_Education_Level'] = df1['Parent_Education_Level'].fillna(
    df1['Parent_Education_Level'].mode()[0])

print(f"  Missing after imputation : {df1.isnull().sum().sum()}")

# Encode categorical columns
print("\n  Encoding categorical columns...")

df1['Gender'] = df1['Gender'].map({'Male': 0, 'Female': 1})
df1['Extracurricular_Activities'] = df1['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
df1['Internet_Access_at_Home']    = df1['Internet_Access_at_Home'].map({'No': 0, 'Yes': 1})
df1['Family_Income_Level']        = df1['Family_Income_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df1['Parent_Education_Level']     = df1['Parent_Education_Level'].map({
    'High School': 0, 'Some College': 1, "Bachelor's": 2,
    "Master's": 3, 'PhD': 4
})
df1['Department'] = df1['Department'].map({
    'Business': 0, 'Mathematics': 1, 'Engineering': 2, 'Computer Science': 3
})

# Normalise stress level: DS1 is 1-10, DS3 is 0-2
# Map DS1 stress (1-10) - 3 bins: Low(0), Medium(1), High(2)
df1['StressLevel'] = pd.cut(
    df1['StressLevel_raw'],
    bins=[0, 3, 6, 10],
    labels=[0, 1, 2]
).astype(int)
df1 = df1.drop(columns=['StressLevel_raw'])
print("  Stress level binned to 0/1/2 (Low/Medium/High)")

# Create target column: at_risk
# At-risk = Grade D or F
df1['at_risk'] = df1['Grade'].isin(['D', 'F']).astype(int)
grade_map = {'A': 3, 'B': 2, 'C': 1, 'D': 0, 'F': 0}
df1['FinalGrade'] = df1['Grade'].map(grade_map)
df1 = df1.drop(columns=['Grade'])
print(f"  At-risk label created: {df1['at_risk'].sum()} at-risk"
      f"({df1['at_risk'].mean()*100:.1f}%)")

# Select and rename final DS1 columns
ds1_clean = df1.rename(columns={
    'Study_Hours_per_Week'      : 'StudyHours',
    'Assignments_Avg'           : 'AssignmentCompletion',
    'Extracurricular_Activities': 'Extracurricular',
    'Internet_Access_at_Home'   : 'Internet',
    'Midterm_Score'             : 'MidtermScore',
    'Final_Score'               : 'FinalScore',
    'Sleep_Hours_per_Night'     : 'SleepHours',
    'Participation_Score'       : 'Participation',
    'Family_Income_Level'       : 'IncomeLevel',
    'Parent_Education_Level'    : 'ParentEducation',
}).copy()

ds1_clean['source'] = 'dataset1'
print(f"\n  Dataset 1 cleaned shape: {ds1_clean.shape}")

# ============================================================
# STEP 3 CLEAN & ALIGN DATASET 3
# ============================================================
print("\n[3] Aligning Dataset 3...")

df3_clean = df3.copy()

# ExamScore MidtermScore and FinalScore
# Keep ExamScore in its original 40-100 range (same as DS1)
# Add small random noise so the two scores differ slightly
np.random.seed(42)
df3_clean['MidtermScore'] = (df3_clean['ExamScore'] + np.random.uniform(-3, 3, len(df3_clean))).clip(40, 100).round(2)
df3_clean['FinalScore']   = (df3_clean['ExamScore'] + np.random.uniform(-3, 3, len(df3_clean))).clip(40, 100).round(2)
print("  ExamScore mapped to MidtermScore / FinalScore (kept in 40-100 range)")

# Create at_risk: FinalGrade == 3 is Fail (encoding is REVERSED in DS3)
# FinalGrade: 0=Distinction, 1=Merit, 2=Pass, 3=Fail
# at_risk = Fail (3) or borderline Pass (2) - matches DS1's D or F definition
df3_clean['at_risk'] = df3_clean['FinalGrade'].isin([2, 3]).astype(int)
# Also fix FinalGrade to match DS1 direction: 3=Distinction-3, 2=Merit-2, 1=Pass-1, 0=Fail-0
df3_clean['FinalGrade'] = 3 - df3_clean['FinalGrade']  # flip: 0=3, 1=2, 2=1, 3=0
print(f"  At-risk label created (FinalGrade 2 or 3 = at_risk): {df3_clean['at_risk'].sum()} "
      f"({df3_clean['at_risk'].mean()*100:.1f}%)")

# Drop columns with no DS1 equivalent
drop_ds3 = ['LearningStyle', 'EduTech', 'OnlineCourses',
            'Discussions', 'ExamScore']
df3_clean = df3_clean.drop(columns=drop_ds3)
print(f"  Dropped non-alignable columns: {drop_ds3}")

# Fill DS1-only columns not in DS3 with NaN
# (will be kept for DS1 rows only)
ds3_aligned = df3_clean.rename(columns={
    'Resources' : 'IncomeLevel',     # closest proxy: resource access ~ income
    'Motivation': 'ParentEducation', # closest proxy: motivation ~ parental influence
}).copy()

# Add missing DS1-only columns as NaN for DS3
for col in ['Quizzes_Avg', 'Projects_Score', 'Total_Score',
            'SleepHours', 'Participation', 'Department']:
    ds3_aligned[col] = np.nan

ds3_aligned['source'] = 'dataset3'
print(f"\n  Dataset 3 aligned shape: {ds3_aligned.shape}")

# ============================================================
# STEP 4 STACK BOTH DATASETS
# ============================================================
print("\n[4] Stacking datasets...")

# Align columns, keep only common columns for the combined set
core_cols = [
    'StudyHours', 'Attendance', 'AssignmentCompletion',
    'MidtermScore', 'FinalScore', 'Extracurricular',
    'Internet', 'Gender', 'Age', 'StressLevel',
    'IncomeLevel', 'ParentEducation',
    'Participation', 'SleepHours', 'Department',
    'Quizzes_Avg', 'Projects_Score', 'Total_Score',
    'FinalGrade', 'at_risk', 'source'
]

# Ensure all core cols exist in both
for col in core_cols:
    if col not in ds1_clean.columns:
        ds1_clean[col] = np.nan
    if col not in ds3_aligned.columns:
        ds3_aligned[col] = np.nan

combined = pd.concat(
    [ds1_clean[core_cols], ds3_aligned[core_cols]],
    ignore_index=True
)

print(f"  Combined shape: {combined.shape}")
print(f"  From Dataset 1: {(combined['source']=='dataset1').sum()}")
print(f"  From Dataset 3: {(combined['source']=='dataset3').sum()}")

# ============================================================
# STEP 5 FINAL CHECKS
# ============================================================
print("\n[5] Final quality checks...")

print(f"\n  Missing values per column:")
miss = combined.isnull().sum()
for col, n in miss[miss > 0].items():
    print(f"    {col:<25} {n:>5} ({n/len(combined)*100:.1f}%)")

print(f"\n  At-risk distribution:")
vc = combined['at_risk'].value_counts()
print(f"    Not at-risk (0): {vc[0]:>5} ({vc[0]/len(combined)*100:.1f}%)")
print(f"    At-risk     (1): {vc[1]:>5} ({vc[1]/len(combined)*100:.1f}%)")

print(f"\n  FinalGrade distribution:")
vc_g = combined['FinalGrade'].value_counts().sort_index()
labels = {0:'Fail/D/F', 1:'Pass/C', 2:'Merit/B', 3:'Distinction/A'}
for k, v in vc_g.items():
    print(f"    {k} ({labels.get(k,'?')}): {v:>5} ({v/len(combined)*100:.1f}%)")

print(f"\n  Source breakdown:")
print(f"    {combined['source'].value_counts().to_dict()}")

# ============================================================
# STEP 6 SAVE OUTPUT
# ============================================================
print(f"\n[6] Saving to '{OUTPUT}'...")
combined.to_csv(OUTPUT, index=False)
print(f" Saved Shape: {combined.shape}")

# ============================================================
# STEP 7 PREVIEW
# ============================================================
print("\n[7] Preview of combined dataset (first 5 rows):")
print(combined.head().to_string())

print("\n" + "=" * 55)
print("PHASE 1 COMPLETE")
print("=" * 55)
print(f"""
Output file : {OUTPUT}
Rows        : {len(combined)}
Columns     : {combined.shape[1]}
At-risk     : {combined['at_risk'].sum()} ({combined['at_risk'].mean()*100:.1f}%)
Sources     : Dataset 1 ({(combined['source']=='dataset1').sum()}) +
              Dataset 3 ({(combined['source']=='dataset3').sum()})

Next step   : Phase 2 — Feature selection & model training
""")
print("=" * 55)
