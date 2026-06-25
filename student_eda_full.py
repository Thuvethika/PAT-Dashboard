# ============================================================
# EDA Students Grading Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Global plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10})

# ============================================================
# STEP 1 LOAD THE DATA
# ============================================================
CSV_PATH = 'Students_Grading_Dataset_Biased.csv'

df = pd.read_csv(CSV_PATH)

print("=" * 55)
print("STEP 1 - BASIC INFO")
print("=" * 55)
print(f"Shape : {df.shape[0]} rows × {df.shape[1]} columns\n")
print("Column names:")
for col in df.columns:
    print(f"  {col}")

# ============================================================
# STEP 2 DATA TYPES & OVERVIEW
# ============================================================
print("\n" + "=" * 55)
print("STEP 2 - DATA TYPES")
print("=" * 55)
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head().to_string())

# ============================================================
# STEP 3 MISSING VALUES
# ============================================================
print("\n" + "=" * 55)
print("STEP 3 - MISSING VALUES")
print("=" * 55)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing,
                            'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df)

# Plot missing values
if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_df) * 0.6)))
    bars = ax.barh(missing_df.index, missing_df['Missing %'], color='#D85A30', height=0.5)
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values by Column')
    for bar, val in zip(bars, missing_df['Missing %']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_01_missing_values.png')
    plt.show()
    print("\nSaved plot_01_missing_values.png")


# ============================================================
# STEP 4 DESCRIPTIVE STATISTICS (NUMERIC)
# ============================================================
print("\n" + "=" * 55)
print("STEP 4 DESCRIPTIVE STATISTICS (NUMERIC)")
print("=" * 55)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(df[numeric_cols].describe().round(2).to_string())

# ============================================================
# STEP 5 CATEGORICAL COLUMN DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 55)
print("STEP 5 CATEGORICAL DISTRIBUTIONS")
print("=" * 55)
cat_cols = ['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
            'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']

for col in cat_cols:
    if col in df.columns:
        vc = df[col].value_counts()
        pct = (vc / len(df) * 100).round(1)
        print(f"\n{col}:")
        for k, v in vc.items():
            print(f"  {k:<20} {v:>5}  ({pct[k]:.1f}%)")

# Plot categorical counts
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
colors = sns.color_palette('muted', 8)

for i, col in enumerate(cat_cols):
    if col not in df.columns:
        continue
    vc = df[col].value_counts()
    axes[i].bar(vc.index.astype(str), vc.values, color=colors[i], edgecolor='white')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=30)
    for j, v in enumerate(vc.values):
        axes[i].text(j, v + 10, str(v), ha='center', fontsize=9)

axes[-1].set_visible(False)  # hide extra subplot
plt.suptitle('Categorical Column Distributions', fontsize=14, fontweight='500', y=1.01)
plt.tight_layout()
plt.savefig('plot_02_categorical_distributions.png', bbox_inches='tight')
plt.show()
print("Saved plot_02_categorical_distributions.png")

# ============================================================
# STEP 6 DISTRIBUTION OF KEY NUMERIC SCORES
# ============================================================
print("\n" + "=" * 55)
print("STEP 6 SCORE DISTRIBUTIONS (HISTOGRAMS)")
print("=" * 55)
score_cols = ['Total_Score', 'Midterm_Score', 'Final_Score',
              'Assignments_Avg', 'Quizzes_Avg', 'Projects_Score',
              'Participation_Score', 'Attendance (%)']
score_cols = [c for c in score_cols if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
palette = sns.color_palette('Blues_d', len(score_cols))

for i, col in enumerate(score_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=30, color=palette[i], edgecolor='white', linewidth=0.4)
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean: {data.mean():.1f}')
    axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=1.2, label=f'Median: {data.median():.1f}')
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle('Distribution of Score Columns', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('plot_03_score_distributions.png')
plt.show()
print("Saved plot_03_score_distributions.png")

# ============================================================
# STEP 7 BOXPLOTS: TOTAL SCORE BY CATEGORY
# ============================================================
print("\n" + "=" * 55)
print("STEP 7 TOTAL SCORE BY CATEGORY (BOXPLOTS)")
print("=" * 55)
group_cols = ['Grade', 'Department', 'Gender',
              'Family_Income_Level', 'Internet_Access_at_Home',
              'Extracurricular_Activities']
group_cols = [c for c in group_cols if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
palette2 = sns.color_palette('Set2')

for i, col in enumerate(group_cols):
    order = df.groupby(col)['Total_Score'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=col, y='Total_Score', order=order,
                palette='Set2', ax=axes[i], linewidth=0.8,
                flierprops=dict(marker='o', markersize=2, alpha=0.4))
    axes[i].set_title(f'Total Score by {col}')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=20)

plt.suptitle('Total Score Grouped by Categorical Variables', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('plot_04_boxplots_by_category.png', bbox_inches='tight')
plt.show()
print("Saved plot_04_boxplots_by_category.png")

# ============================================================
# STEP 8 CORRELATION HEATMAP
# ============================================================
print("\n" + "=" * 55)
print("STEP 8 CORRELATION HEATMAP")
print("=" * 55)
corr_cols = ['Total_Score', 'Midterm_Score', 'Final_Score',
             'Assignments_Avg', 'Quizzes_Avg', 'Projects_Score',
             'Participation_Score', 'Attendance (%)',
             'Study_Hours_per_Week', 'Sleep_Hours_per_Night',
             'Stress_Level (1-10)', 'Age']
corr_cols = [c for c in corr_cols if c in df.columns]

corr_matrix = df[corr_cols].corr().round(3)

print("Correlations with Total_Score (sorted):")
print(corr_matrix['Total_Score'].sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5,
            linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation Matrix (lower triangle)', fontsize=13)
plt.tight_layout()
plt.savefig('plot_05_correlation_heatmap.png')
plt.show()
print("Saved plot_05_correlation_heatmap.png")

# ============================================================
# STEP 9 GRADE VS SOCIOECONOMIC FACTORS (STACKED BARS)
# ============================================================
print("\n" + "=" * 55)
print("STEP 9 — GRADE RATES BY SOCIOECONOMIC FACTORS")
print("=" * 55)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
socio_cols = ['Family_Income_Level', 'Internet_Access_at_Home',
              'Extracurricular_Activities']
grade_order = ['A', 'B', 'C', 'D', 'F']
grade_colors = {'A': '#3266ad', 'B': '#59a14f', 'C': '#e0a400',
                'D': '#e05e2a', 'F': '#9d3d3d'}

for i, col in enumerate(socio_cols):
    if col not in df.columns:
        continue
    ct = pd.crosstab(df[col], df['Grade'], normalize='index')[grade_order] * 100
    ct.plot(kind='bar', stacked=True, ax=axes[i],
            color=[grade_colors[g] for g in grade_order],
            edgecolor='white', linewidth=0.3)
    axes[i].set_title(f'Grade distribution\nby {col}')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=20)
    axes[i].legend(title='Grade', bbox_to_anchor=(1, 1), fontsize=8)
    for bar in axes[i].patches:
        h = bar.get_height()
        if h > 5:
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_y() + h/2,
                         f'{h:.0f}%', ha='center', va='center',
                         fontsize=7.5, color='white', fontweight='500')

plt.suptitle('Grade Distribution by Socioeconomic Factors', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('plot_06_socioeconomic_grades.png', bbox_inches='tight')
plt.show()
print("Saved plot_06_socioeconomic_grades.png")

# ============================================================
# STEP 10 PAIRPLOT (KEY SCORE COLUMNS)
# ============================================================
print("\n" + "=" * 55)
print("STEP 10 PAIRPLOT (Colored by Grade)")
print("=" * 55)
pair_cols = ['Total_Score', 'Midterm_Score', 'Final_Score',
             'Study_Hours_per_Week', 'Attendance (%)', 'Grade']
pair_cols = [c for c in pair_cols if c in df.columns]

sample_df = df[pair_cols].dropna().sample(n=min(1000, len(df)), random_state=42)
pair_palette = {'A': '#3266ad', 'B': '#59a14f', 'C': '#e0a400',
                'D': '#e05e2a', 'F': '#9d3d3d'}

g = sns.pairplot(sample_df, hue='Grade', hue_order=['A','B','C','D','F'],
                 palette=pair_palette, plot_kws={'alpha': 0.35, 's': 15},
                 diag_kind='kde', corner=True)
g.fig.suptitle('Pairplot of Key Scores (sample n=1000, colored by grade)',
               fontsize=13, y=1.01)
plt.savefig('plot_07_pairplot.png', bbox_inches='tight')
plt.show()
print("Saved plot_07_pairplot.png")

# ============================================================
# STEP 11 STUDY HOURS vs TOTAL SCORE SCATTER
# ============================================================
print("\n" + "=" * 55)
print("STEP 11 STUDY HOURS vs TOTAL SCORE")
print("=" * 55)
fig, ax = plt.subplots(figsize=(9, 6))
scatter_palette = {'A': '#3266ad', 'B': '#59a14f', 'C': '#e0a400',
                   'D': '#e05e2a', 'F': '#9d3d3d'}
for grade, grp in df.groupby('Grade'):
    ax.scatter(grp['Study_Hours_per_Week'], grp['Total_Score'],
               alpha=0.25, s=12, label=grade,
               color=scatter_palette.get(grade, 'gray'))
ax.set_xlabel('Study Hours per Week')
ax.set_ylabel('Total Score')
ax.set_title('Study Hours vs Total Score (colored by Grade)')
ax.legend(title='Grade', markerscale=2)
plt.tight_layout()
plt.savefig('plot_08_study_hours_vs_score.png')
plt.show()
print("Saved plot_08_study_hours_vs_score.png")

# ============================================================
# STEP 12 STRESS & SLEEP ANALYSIS 
# ============================================================
print("=" * 55)
print("STEP 12 STRESS LEVEL & SLEEP HOURS ANALYSIS")
print("=" * 55)

# Auto-detect the actual stress column name 
stress_col = next((c for c in df.columns if 'stress' in c.lower()), None)
sleep_col  = next((c for c in df.columns if 'sleep'  in c.lower()), None)

print(f"Detected stress column : {stress_col}")
print(f"Detected sleep column  : {sleep_col}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Avg score by stress level
if stress_col:
    stress_avg = df.groupby(stress_col)['Total_Score'].mean()
    axes[0].plot(stress_avg.index, stress_avg.values, marker='o',
                 color='#D85A30', linewidth=2)
    axes[0].fill_between(stress_avg.index, stress_avg.values,
                         alpha=0.15, color='#D85A30')
    axes[0].set_xlabel(stress_col)
    axes[0].set_ylabel('Average Total Score')
    axes[0].set_title('Avg Score by Stress Level')
else:
    axes[0].text(0.5, 0.5, 'Stress column not found',
                 ha='center', va='center', transform=axes[0].transAxes)

# Avg score by sleep bucket
if sleep_col:
    df['Sleep_Bucket'] = pd.cut(df[sleep_col],
                                bins=[3.9, 5, 6, 7, 8, 9.1],
                                labels=['4–5h', '5–6h', '6–7h', '7–8h', '8–9h'])
    sleep_avg = df.groupby('Sleep_Bucket', observed=True)['Total_Score'].mean()
    axes[1].bar(sleep_avg.index.astype(str), sleep_avg.values,
                color='#1d9e75', edgecolor='white')
    axes[1].set_xlabel('Sleep Hours per Night')
    axes[1].set_ylabel('Average Total Score')
    axes[1].set_title('Avg Score by Sleep Bucket')
    axes[1].set_ylim(73, 77)
    for j, v in enumerate(sleep_avg.values):
        axes[1].text(j, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)
else:
    axes[1].text(0.5, 0.5, 'Sleep column not found',
                 ha='center', va='center', transform=axes[1].transAxes)

plt.suptitle('Wellness Factors vs Academic Performance', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('plot_09_stress_sleep_analysis.png')
plt.show()
print("Saved plot_09_stress_sleep_analysis.png")

# ============================================================
# STEP 13 BIAS DETECTION: GRADE RATE BY GENDER, DEPARTMENT
# ============================================================
print("\n" + "=" * 55)
print("STEP 13 BIAS CHECK: GRADE A RATE BY GENDER & DEPT")
print("=" * 55)
if 'Gender' in df.columns and 'Department' in df.columns:
    a_rate = (df[df['Grade'] == 'A']
              .groupby(['Department', 'Gender'])
              .size()
              .div(df.groupby(['Department', 'Gender']).size())
              .mul(100)
              .round(1)
              .reset_index(name='A_Rate_%'))
    print(a_rate.to_string(index=False))

    pivot = a_rate.pivot(index='Department', columns='Gender', values='A_Rate_%')
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind='bar', ax=ax, color=['#3266ad', '#D85A30'],
               edgecolor='white', width=0.6)
    ax.set_ylabel('A Grade Rate (%)')
    ax.set_title('A Grade Rate: Gender × Department')
    ax.tick_params(axis='x', rotation=20)
    ax.set_xlabel('')
    ax.legend(title='Gender')
    plt.tight_layout()
    plt.savefig('plot_10_bias_gender_dept.png')
    plt.show()
    print("Saved plot_10_bias_gender_dept.png")

# ============================================================
# STEP 14  SUMMARY REPORT (printed)
# ============================================================
print("\n" + "=" * 55)
print("STEP 14 — EDA SUMMARY REPORT")
print("=" * 55)

total = len(df)
a_pct = round(df['Grade'].value_counts(normalize=True).get('A', 0) * 100, 1)
f_pct = round(df['Grade'].value_counts(normalize=True).get('F', 0) * 100, 1)
pass_pct = round(df['Grade'].isin(['A','B','C']).sum() / total * 100, 1)

print(f"""
Dataset            : {total} students, {df.shape[1]} features
Target variable    : Grade  (A={a_pct}%,  F={f_pct}%,  Pass rate={pass_pct}%)

Missing values     :
  Parent_Education : {df['Parent_Education_Level'].isnull().sum()} ({df['Parent_Education_Level'].isnull().mean()*100:.1f}%)
  Assignments_Avg  : {df['Assignments_Avg'].isnull().sum()} ({df['Assignments_Avg'].isnull().mean()*100:.1f}%)
  Attendance       : {df['Attendance (%)'].isnull().sum()} ({df['Attendance (%)'].isnull().mean()*100:.1f}%)

Score stats        :
  Total Score      : mean={df['Total_Score'].mean():.1f},  std={df['Total_Score'].std():.1f},  min={df['Total_Score'].min():.1f},  max={df['Total_Score'].max():.1f}
  Study Hours/wk   : mean={df['Study_Hours_per_Week'].mean():.1f},  std={df['Study_Hours_per_Week'].std():.1f}
  Sleep Hours/night: mean={df['Sleep_Hours_per_Night'].mean():.1f},  std={df['Sleep_Hours_per_Night'].std():.1f}
  Stress Level     : mean={df['Stress_Level (1-10)'].mean():.1f},  std={df['Stress_Level (1-10)'].std():.1f}

Categorical counts :
  Gender           : {df['Gender'].value_counts().to_dict()}
  Department       : {df['Department'].value_counts().to_dict()}
  Income Level     : {df['Family_Income_Level'].value_counts().to_dict()}

Key correlations (with Total_Score):
{df[corr_cols].corr()['Total_Score'].drop('Total_Score').sort_values(ascending=False).round(4).to_string()}

Bias flags         :
  - A grade rate is {round(df[df['Family_Income_Level']=='High']['Grade'].eq('A').mean()*100,1)}% for High income vs
    {round(df[df['Family_Income_Level']=='Low']['Grade'].eq('A').mean()*100,1)}% for Low income students.
  - Students WITH internet: A rate = {round(df[df['Internet_Access_at_Home']=='Yes']['Grade'].eq('A').mean()*100,1)}%
    WITHOUT internet: A rate = {round(df[df['Internet_Access_at_Home']=='No']['Grade'].eq('A').mean()*100,1)}%
  - Extracurricular YES: A rate = {round(df[df['Extracurricular_Activities']=='Yes']['Grade'].eq('A').mean()*100,1)}%
    Extracurricular NO : A rate = {round(df[df['Extracurricular_Activities']=='No']['Grade'].eq('A').mean()*100,1)}%

Output files       :
  plot_01_missing_values.png
  plot_02_categorical_distributions.png
  plot_03_score_distributions.png
  plot_04_boxplots_by_category.png
  plot_05_correlation_heatmap.png
  plot_06_socioeconomic_grades.png
  plot_07_pairplot.png
  plot_08_study_hours_vs_score.png
  plot_09_stress_sleep_analysis.png
  plot_10_bias_gender_dept.png
""")
print("=" * 55)
