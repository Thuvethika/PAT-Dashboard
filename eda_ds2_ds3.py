# ============================================================
#  EDA - Dataset 2 (Engagement) & Dataset 3 (Performance)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10})

# Change paths if needed 
PATH_DS2 = 'student_engagement.csv'
PATH_DS3 = 'student_performance.csv'

df2 = pd.read_csv(PATH_DS2)
df3 = pd.read_csv(PATH_DS3)

# ============================================================
#DATASET 2 - ENGAGEMENT
# ============================================================

print("\n" + "=" * 55)
print("DATASET 2 - ENGAGEMENT")
print("=" * 55)

# Step 1: Basic info 
print("\n[1] Shape & dtypes")
print(f"Shape : {df2.shape[0]} rows × {df2.shape[1]} columns")
print(df2.dtypes)

# Step 2: Missing values
print("\n[2] Missing values")
miss2 = df2.isnull().sum()
print(miss2[miss2 > 0] if miss2.sum() > 0 else "No missing values")

# Step 3: Descriptive statistics
print("\n[3] Descriptive statistics (numeric)")
print(df2.describe().round(2).to_string())

# Step 4: Categorical distributions
print("\n[4] Categorical distributions")
cat_cols2 = df2.select_dtypes(include='object').columns.tolist()
for col in cat_cols2:
    vc = df2[col].value_counts()
    pct = (vc / len(df2) * 100).round(1)
    print(f"\n  {col}:")
    for k, v in vc.items():
        print(f"    {k:<20} {v:>4}  ({pct[k]:.1f}%)")

# Plot 1: Target distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors_eng = {'Low': '#E24B4A', 'Medium': '#EF9F27', 'High': '#1D9E75'}

vc = df2['engagement_level'].value_counts()
bars = axes[0].bar(vc.index, vc.values,
                   color=[colors_eng.get(k, '#888') for k in vc.index],
                   edgecolor='white')
axes[0].set_title('Engagement level distribution')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, vc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontsize=10)

# Plot 2: Device type
vc2 = df2['device_type'].value_counts()
axes[1].pie(vc2.values, labels=vc2.index, autopct='%1.1f%%',
            colors=['#3266ad', '#5349b4', '#e0a400'], startangle=90)
axes[1].set_title('Device type')

# Plot 3: Course difficulty
order = ['Easy', 'Medium', 'Hard']
vc3 = df2['course_difficulty'].value_counts().reindex(order)
axes[2].bar(vc3.index, vc3.values,
            color=['#59a14f', '#e0a400', '#E24B4A'], edgecolor='white')
axes[2].set_title('Course difficulty')
plt.suptitle('Dataset 2 Categorical distributions', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds2_plot01_categorical.png')
plt.show()
print("\nSaved - ds2_plot01_categorical.png")

# Plot :4 Numeric distributions
num_cols2 = df2.select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
palette = sns.color_palette('Blues_d', len(num_cols2))

for i, col in enumerate(num_cols2):
    data = df2[col].dropna()
    axes[i].hist(data, bins=25, color=palette[i], edgecolor='white', linewidth=0.4)
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.2,
                    label=f'Mean: {data.mean():.1f}')
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

for j in range(len(num_cols2), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Dataset 2 - Numeric distributions', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds2_plot02_numeric_distributions.png')
plt.show()
print("Saved - ds2_plot02_numeric_distributions.png")

# Plot 5: Engagement level vs numeric features (boxplot)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
order_eng = ['Low', 'Medium', 'High']

for i, col in enumerate(num_cols2):
    sns.boxplot(data=df2, x='engagement_level', y=col,
                order=order_eng, palette=colors_eng,
                ax=axes[i], linewidth=0.8,
                flierprops=dict(marker='o', markersize=2, alpha=0.4))
    axes[i].set_title(f'{col}\nby engagement level')
    axes[i].set_xlabel('')

for j in range(len(num_cols2), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Dataset 2 - Numeric features by engagement level', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds2_plot03_boxplots_by_engagement.png')
plt.show()
print("Saved - ds2_plot03_boxplots_by_engagement.png")

# Plot 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr2 = df2[num_cols2].corr()
mask = np.triu(np.ones_like(corr2, dtype=bool))
sns.heatmap(corr2, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            ax=ax, annot_kws={'size': 9})
ax.set_title('Dataset 2 - Correlation heatmap')
plt.tight_layout()
plt.savefig('ds2_plot04_correlation.png')
plt.show()
print("Saved - ds2_plot04_correlation.png")

# Step 5: Class imbalance warning
print("\n[5] Class imbalance check")
vc_eng = df2['engagement_level'].value_counts()
for k, v in vc_eng.items():
    flag = "SEVERE" if v < 20 else ("Minor" if v < 100 else "OK")
    print(f"  {k:<10} {v:>4}  ({v/len(df2)*100:.1f}%)  {flag}")

# ============================================================
# DATASET 3 — PERFORMANCE
# ============================================================

print("\n\n" + "=" * 55)
print("DATASET 3 — PERFORMANCE")
print("=" * 55)

# Step 1: Basic info
print("\n[1] Shape & dtypes")
print(f"Shape : {df3.shape[0]} rows × {df3.shape[1]} columns")
print(df3.dtypes)

# Step 2: Missing values
print("\n[2] Missing values")
miss3 = df3.isnull().sum()
print(miss3[miss3 > 0] if miss3.sum() > 0 else "No missing values ✓")

# Step 3: Descriptive statistics
print("\n[3] Descriptive statistics")
print(df3.describe().round(2).to_string())

# Step 4: Decode encoded columns
print("\n[4] Decoded value meanings")
decode_map = {
    'Gender'          : {0: 'Male', 1: 'Female'},
    'Internet'        : {0: 'No', 1: 'Yes'},
    'Extracurricular' : {0: 'No', 1: 'Yes'},
    'Motivation'      : {0: 'Low', 1: 'Medium', 2: 'High'},
    'Resources'       : {0: 'Low', 1: 'Medium', 2: 'High'},
    'StressLevel'     : {0: 'Low', 1: 'Medium', 2: 'High'},
    'EduTech'         : {0: 'No', 1: 'Yes'},
    'Discussions'     : {0: 'No', 1: 'Yes'},
    'LearningStyle'   : {0: 'Visual', 1: 'Auditory', 2: 'Reading', 3: 'Kinesthetic'},
    'FinalGrade'      : {0: 'Fail', 1: 'Pass', 2: 'Merit', 3: 'Distinction'},
}
for col, mapping in decode_map.items():
    vc = df3[col].value_counts().sort_index()
    print(f"\n  {col}:")
    for k, v in vc.items():
        label = mapping.get(k, str(k))
        print(f"    {k} = {label:<15} - {v:>5} ({v/len(df3)*100:.1f}%)")

# Plot 1: FinalGrade distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
grade_labels = {0: 'Fail', 1: 'Pass', 2: 'Merit', 3: 'Distinction'}
grade_colors = ['#E24B4A', '#EF9F27', '#3266ad', '#1D9E75']

vc_g = df3['FinalGrade'].value_counts().sort_index()
bars = axes[0].bar([grade_labels[k] for k in vc_g.index], vc_g.values,
                   color=grade_colors, edgecolor='white')
axes[0].set_title('FinalGrade distribution')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, vc_g.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val}\n({val/len(df3)*100:.1f}%)', ha='center', fontsize=9)

# Cumulative — show Fail as at-risk
at_risk_n = (df3['FinalGrade'] == 0).sum()
not_risk_n = len(df3) - at_risk_n
axes[1].pie([at_risk_n, not_risk_n],
            labels=[f'At-risk (Fail)\n{at_risk_n}', f'Not at-risk\n{not_risk_n}'],
            colors=['#E24B4A', '#3266ad'], autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white'})
axes[1].set_title('At-risk label (FinalGrade = 0 → at-risk)')

plt.suptitle('Dataset 3 — Target variable', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds3_plot01_finalgrade.png')
plt.show()
print("\nSaved - ds3_plot01_finalgrade.png")

# Plot 2: Continuous feature distributions
cont_cols3 = ['StudyHours', 'Attendance', 'ExamScore', 'AssignmentCompletion', 'Age', 'OnlineCourses']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
palette3 = sns.color_palette('Blues_d', len(cont_cols3))

for i, col in enumerate(cont_cols3):
    data = df3[col]
    axes[i].hist(data, bins=30, color=palette3[i], edgecolor='white', linewidth=0.4)
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.2,
                    label=f'Mean: {data.mean():.1f}')
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle('Dataset 3 — Continuous feature distributions', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds3_plot02_continuous_distributions.png')
plt.show()
print("Saved - ds3_plot02_continuous_distributions.png")

# Plot 3: Categorical encoded features
cat_enc_cols = ['Gender', 'Internet', 'Extracurricular', 'Motivation',
                'Resources', 'StressLevel', 'EduTech', 'Discussions', 'LearningStyle']
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.flatten()
colors_cat = sns.color_palette('Set2')

for i, col in enumerate(cat_enc_cols):
    mapping = decode_map.get(col, {})
    vc = df3[col].value_counts().sort_index()
    labels = [mapping.get(k, str(k)) for k in vc.index]
    axes[i].bar(labels, vc.values, color=colors_cat[i % len(colors_cat)], edgecolor='white')
    axes[i].set_title(col)
    for j, v in enumerate(vc.values):
        axes[i].text(j, v + 50, f'{v/len(df3)*100:.0f}%', ha='center', fontsize=9)

plt.suptitle('Dataset 3 — Encoded categorical features (decoded)', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds3_plot03_categorical_encoded.png')
plt.show()
print("Saved - ds3_plot03_categorical_encoded.png")

# Plot 4: Key features vs FinalGrade (boxplots)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
df3_decoded = df3.copy()
df3_decoded['FinalGrade_label'] = df3_decoded['FinalGrade'].map(grade_labels)
order_g = ['Fail', 'Pass', 'Merit', 'Distinction']

for i, col in enumerate(cont_cols3):
    sns.boxplot(data=df3_decoded, x='FinalGrade_label', y=col,
                order=order_g, palette=grade_colors,
                ax=axes[i], linewidth=0.8,
                flierprops=dict(marker='o', markersize=1.5, alpha=0.3))
    axes[i].set_title(f'{col} by FinalGrade')
    axes[i].set_xlabel('')

plt.suptitle('Dataset 3 — Feature vs FinalGrade', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds3_plot04_features_vs_grade.png')
plt.show()
print("Saved - ds3_plot04_features_vs_grade.png")

# Plot 5: Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr3 = df3.corr().round(3)
mask3 = np.triu(np.ones_like(corr3, dtype=bool))
sns.heatmap(corr3, mask=mask3, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            ax=ax, annot_kws={'size': 8})
ax.set_title('Dataset 3 — Correlation heatmap')
plt.tight_layout()
plt.savefig('ds3_plot05_correlation.png')
plt.show()
print("Saved - ds3_plot05_correlation.png")

#  Step 5: Correlations with FinalGrade 
print("\n[5] Correlations with FinalGrade (sorted)")
corr_target = df3.corr()['FinalGrade'].drop('FinalGrade').sort_values(ascending=False)
print(corr_target.round(4).to_string())

# Plot 6: Stress & motivation vs grade
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
stress_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
motiv_labels  = {0: 'Low', 1: 'Medium', 2: 'High'}

ct_stress = pd.crosstab(df3['StressLevel'], df3['FinalGrade'], normalize='index') * 100
ct_stress.index = [stress_labels[i] for i in ct_stress.index]
ct_stress.columns = [grade_labels[c] for c in ct_stress.columns]
ct_stress[['Fail','Pass','Merit','Distinction']].plot(
    kind='bar', stacked=True, ax=axes[0],
    color=grade_colors, edgecolor='white', linewidth=0.3)
axes[0].set_title('Grade distribution by stress level')
axes[0].set_xlabel('Stress level')
axes[0].set_ylabel('%')
axes[0].tick_params(axis='x', rotation=0)

ct_motiv = pd.crosstab(df3['Motivation'], df3['FinalGrade'], normalize='index') * 100
ct_motiv.index = [motiv_labels[i] for i in ct_motiv.index]
ct_motiv.columns = [grade_labels[c] for c in ct_motiv.columns]
ct_motiv[['Fail','Pass','Merit','Distinction']].plot(
    kind='bar', stacked=True, ax=axes[1],
    color=grade_colors, edgecolor='white', linewidth=0.3)
axes[1].set_title('Grade distribution by motivation')
axes[1].set_xlabel('Motivation level')
axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle('Dataset 3 — Psychosocial factors vs grade', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ds3_plot06_stress_motivation.png')
plt.show()
print("Saved - ds3_plot06_stress_motivation.png")

# ============================================================
# SUMMARY REPORT
# ============================================================
print("\n\n" + "=" * 55)
print("SUMMARY REPORT — BOTH DATASETS")
print("=" * 55)

print(f"""
DATASET 2 — ENGAGEMENT
  Rows / Columns   : {df2.shape[0]} / {df2.shape[1]}
  Missing          : None
  Target           : engagement_level
  Class imbalance  : Low={df2['engagement_level'].value_counts()['Low']},
                     Medium={df2['engagement_level'].value_counts()['Medium']},
                     High={df2['engagement_level'].value_counts()['High']}
  Verdict          : NOT suitable for ML (High class has only 9 samples)
                     USE for dashboard engagement insights only

DATASET 3 — PERFORMANCE
  Rows / Columns   : {df3.shape[0]} / {df3.shape[1]}
  Missing          : None
  Target           : FinalGrade (0=Fail, 1=Pass, 2=Merit, 3=Distinction)
  At-risk samples  : {(df3['FinalGrade']==0).sum()} ({(df3['FinalGrade']==0).mean()*100:.1f}%)
  Verdict          : SUITABLE for ML — can be aligned and stacked with Dataset 1

FEATURE ALIGNMENT PLAN (DS3 → DS1):
  ✓  StudyHours          → Study_Hours_per_Week      (same scale)
  ✓  Attendance          → Attendance (%)            (same scale)
  ✓  Extracurricular     → Extracurricular_Activities (0/1 → No/Yes)
  ✓  Internet            → Internet_Access_at_Home   (0/1 → No/Yes)
  ✓  Gender              → Gender                    (0/1 → Male/Female)
  ✓  Age                 → Age                       (same scale)
  ~  ExamScore           → avg(Midterm, Final)        (approximate)
  ~  AssignmentCompletion→ Assignments_Avg            (approximate)
  ~  StressLevel (0-2)   → Stress_Level (1-10)       (normalise 0-9 scale)
  ✗  LearningStyle       → No equivalent — drop
  ✗  EduTech             → No equivalent — drop
  ✗  OnlineCourses       → No equivalent — drop
  ✗  Motivation          → No equivalent — keep as extra feature or drop
  ✗  Resources           → No equivalent — keep as extra feature or drop

OUTPUT FILES:
  ds2_plot01_categorical.png
  ds2_plot02_numeric_distributions.png
  ds2_plot03_boxplots_by_engagement.png
  ds2_plot04_correlation.png
  ds3_plot01_finalgrade.png
  ds3_plot02_continuous_distributions.png
  ds3_plot03_categorical_encoded.png
  ds3_plot04_features_vs_grade.png
  ds3_plot05_correlation.png
  ds3_plot06_stress_motivation.png
""")
print("=" * 55)
print("EDA complete")
print("=" * 55)
