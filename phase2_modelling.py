import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (classification_report, confusion_matrix,
                                        roc_auc_score, roc_curve,
                                        precision_recall_curve, f1_score,
                                        accuracy_score, ConfusionMatrixDisplay)
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection        import permutation_importance

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print(" XGBoost available")
except ImportError:
    HAS_XGB = False
    print(" XGBoost not installed — using GradientBoosting instead")
    print("  Install with: pip install xgboost")

# Optional: SHAP
try:
    import shap
    HAS_SHAP = True
    print(" SHAP available")
except ImportError:
    HAS_SHAP = False
    print(" SHAP not installed — using permutation importance instead")
    print("  Install with: pip install shap")

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10})

# ── Path ─────────────────────────────────────────────────────
CSV_PATH = 'combined_dataset.csv'

print("\n" + "=" * 60)
print("PHASE 2 — FEATURE SELECTION & MODEL TRAINING")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD & PREPARE DATA
# ============================================================
print("\n[1] Loading combined dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")

# ── Select features available in both datasets ───────────────
# Core features present for ALL rows (no heavy NaN)
CORE_FEATURES = [
    'StudyHours', 'Attendance', 'AssignmentCompletion',
    'MidtermScore', 'FinalScore', 'Extracurricular',
    'Internet', 'Gender', 'Age', 'StressLevel',
    'IncomeLevel', 'ParentEducation'
]

# DS1-only features (NaN for Dataset 3 rows — use separately if needed)
DS1_ONLY = ['Participation', 'SleepHours', 'Quizzes_Avg',
            'Projects_Score', 'Total_Score']

TARGET = 'at_risk'

# Use core features for main models (works for all 19,003 rows)
X = df[CORE_FEATURES].copy()
y = df[TARGET].copy()

print(f"  Features used    : {len(CORE_FEATURES)}")
print(f"  Total samples    : {len(y)}")
print(f"  At-risk (1)      : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Not at-risk (0)  : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Missing values   : {X.isnull().sum().sum()} ✓")

# ============================================================
# STEP 2 — FEATURE SELECTION
# ============================================================
print("\n[2] Feature selection...")

# ── Mutual Information ───────────────────────────────────────
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': CORE_FEATURES, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

print("\n  Mutual Information scores (higher = more predictive):")
for _, row in mi_df.iterrows():
    bar = '█' * int(row['MI_Score'] * 100)
    print(f"    {row['Feature']:<25} {row['MI_Score']:.4f}  {bar}")

# ── Correlation with target ──────────────────────────────────
corr_target = X.join(y).corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\n  Correlation with at_risk (|r|, sorted):")
for feat, val in corr_target.items():
    direction = '↑' if val > 0 else '↓'
    print(f"    {feat:<25} {val:+.4f}  {direction}")

# ── Plot: Feature importance overview ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MI scores
colors_mi = ['#3266ad' if s > mi_df['MI_Score'].median() else '#aec6e8'
             for s in mi_df['MI_Score']]
axes[0].barh(mi_df['Feature'], mi_df['MI_Score'], color=colors_mi)
axes[0].set_title('Mutual Information Score\n(higher = more predictive of at-risk)')
axes[0].set_xlabel('MI Score')
axes[0].invert_yaxis()

# Correlation with target
colors_corr = ['#E24B4A' if v > 0 else '#3266ad' for v in corr_target.values]
axes[1].barh(corr_target.index, corr_target.values, color=colors_corr)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('Correlation with at_risk\n(red = positive, blue = negative)')
axes[1].set_xlabel('Pearson r')
axes[1].invert_yaxis()

plt.suptitle('Phase 2 — Feature Selection Analysis', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('ph2_plot01_feature_selection.png', bbox_inches='tight')
plt.show()
print("\n  Saved - ph2_plot01_feature_selection.png")

# ── Select top features ──────────────────────────────────────
TOP_K = 10
top_features = mi_df.head(TOP_K)['Feature'].tolist()
print(f"\n  Top {TOP_K} features selected: {top_features}")

X_selected = X[top_features]

# ============================================================
# STEP 3 — TRAIN / VALIDATION / TEST SPLIT
# ============================================================
print("\n[3] Splitting data (70% train / 15% val / 15% test)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_selected, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"  Train : {X_train.shape[0]} rows  "
      f"(at-risk: {y_train.sum()} = {y_train.mean()*100:.1f}%)")
print(f"  Val   : {X_val.shape[0]} rows  "
      f"(at-risk: {y_val.sum()} = {y_val.mean()*100:.1f}%)")
print(f"  Test  : {X_test.shape[0]} rows  "
      f"(at-risk: {y_test.sum()} = {y_test.mean()*100:.1f}%)")

# ── Scale features ───────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ── Class weights (handle imbalance) ────────────────────────
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1]}
print(f"\n  Class weights: {class_weight_dict}")

# ============================================================
# STEP 4 — TRAIN MODELS
# ============================================================
print("\n[4] Training models...")

results = {}

# ── Model 1: Logistic Regression (baseline) ─────────────────
print("\n  [4.1] Logistic Regression (baseline)...")
lr = LogisticRegression(
    class_weight='balanced', max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_sc, y_train)

lr_val_pred  = lr.predict(X_val_sc)
lr_val_proba = lr.predict_proba(X_val_sc)[:, 1]
results['Logistic Regression'] = {
    'model'     : lr,
    'val_pred'  : lr_val_pred,
    'val_proba' : lr_val_proba,
    'scaled'    : True
}
print(f"    Val Accuracy : {accuracy_score(y_val, lr_val_pred):.4f}")
print(f"    Val F1       : {f1_score(y_val, lr_val_pred):.4f}")
print(f"    Val ROC-AUC  : {roc_auc_score(y_val, lr_val_proba):.4f}")

# ── Model 2: Random Forest ───────────────────────────────────
print("\n  [4.2] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_val_pred  = rf.predict(X_val)
rf_val_proba = rf.predict_proba(X_val)[:, 1]
results['Random Forest'] = {
    'model'     : rf,
    'val_pred'  : rf_val_pred,
    'val_proba' : rf_val_proba,
    'scaled'    : False
}
print(f"    Val Accuracy : {accuracy_score(y_val, rf_val_pred):.4f}")
print(f"    Val F1       : {f1_score(y_val, rf_val_pred):.4f}")
print(f"    Val ROC-AUC  : {roc_auc_score(y_val, rf_val_proba):.4f}")

# ── Model 3: Gradient Boosting / XGBoost ────────────────────
if HAS_XGB:
    print("\n  [4.3] XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    gb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='logloss', verbosity=0)
    gb.fit(X_train, y_train)
    model3_name = 'XGBoost'
else:
    print("\n  [4.3] Gradient Boosting (sklearn)...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    model3_name = 'Gradient Boosting'

gb_val_pred  = gb.predict(X_val)
gb_val_proba = gb.predict_proba(X_val)[:, 1]
results[model3_name] = {
    'model'     : gb,
    'val_pred'  : gb_val_pred,
    'val_proba' : gb_val_proba,
    'scaled'    : False
}
print(f"    Val Accuracy : {accuracy_score(y_val, gb_val_pred):.4f}")
print(f"    Val F1       : {f1_score(y_val, gb_val_pred):.4f}")
print(f"    Val ROC-AUC  : {roc_auc_score(y_val, gb_val_proba):.4f}")

# ============================================================
# STEP 5 — CROSS-VALIDATION
# ============================================================
print("\n[5] 5-fold cross-validation (on training set)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, info in results.items():
    model = info['model']
    X_cv  = X_train_sc if info['scaled'] else X_train
    cv_scores = cross_val_score(model, X_cv, y_train, cv=cv,
                                scoring='roc_auc', n_jobs=-1)
    info['cv_roc_auc'] = cv_scores
    print(f"  {name:<25} ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# STEP 6 — COMPARE MODELS & SELECT BEST
# ============================================================
print("\n[6] Model comparison on validation set...")

comparison = []
for name, info in results.items():
    comparison.append({
        'Model'    : name,
        'Accuracy' : accuracy_score(y_val, info['val_pred']),
        'F1'       : f1_score(y_val, info['val_pred']),
        'ROC-AUC'  : roc_auc_score(y_val, info['val_proba']),
        'CV ROC-AUC Mean': info['cv_roc_auc'].mean(),
        'CV ROC-AUC Std' : info['cv_roc_auc'].std(),
    })

comp_df = pd.DataFrame(comparison).sort_values('ROC-AUC', ascending=False)
print("\n" + comp_df.round(4).to_string(index=False))

best_name = comp_df.iloc[0]['Model']
best_info = results[best_name]
best_model = best_info['model']
print(f"\n   Best model: {best_name}")

# ── Plot: model comparison bars ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['Accuracy', 'F1', 'ROC-AUC']
colors_m = ['#3266ad', '#1D9E75', '#E24B4A']
model_names = comp_df['Model'].tolist()

for i, metric in enumerate(metrics):
    vals = comp_df[metric].values
    bars = axes[i].bar(model_names, vals,
                       color=['#3266ad' if n != best_name else '#1D9E75'
                              for n in model_names],
                       edgecolor='white')
    axes[i].set_title(metric)
    axes[i].set_ylim(max(0, vals.min() - 0.05), min(1, vals.max() + 0.05))
    axes[i].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='500')

plt.suptitle(f'Model Comparison (validation set)  |  Best: {best_name}',
             fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig('ph2_plot02_model_comparison.png', bbox_inches='tight')
plt.show()
print("  Saved - ph2_plot02_model_comparison.png")

# ── Plot: ROC curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
line_colors = ['#3266ad', '#1D9E75', '#E24B4A']

for (name, info), color in zip(results.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_val, info['val_proba'])
    auc = roc_auc_score(y_val, info['val_proba'])
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)

axes[0].plot([0,1],[0,1],'k--', linewidth=0.8, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — all models')
axes[0].legend(fontsize=9)

# Precision-Recall curve for best model
X_val_best = X_val_sc if best_info['scaled'] else X_val
prec, rec, _ = precision_recall_curve(y_val, best_info['val_proba'])
axes[1].plot(rec, prec, color='#3266ad', linewidth=2,
             label=f'{best_name}')
axes[1].axhline(y_val.mean(), color='gray', linestyle='--',
                label=f'Baseline ({y_val.mean():.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title(f'Precision-Recall — {best_name}')
axes[1].legend(fontsize=9)

plt.suptitle('Phase 2 — ROC & Precision-Recall Curves', fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig('ph2_plot03_roc_prc.png', bbox_inches='tight')
plt.show()
print("  Saved - ph2_plot03_roc_prc.png")

# ============================================================
# STEP 7 — FEATURE IMPORTANCE (best model)
# ============================================================
print(f"\n[7] Feature importance — {best_name}...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Built-in importance ──────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    imp_df = pd.DataFrame({'Feature': top_features, 'Importance': imp})
    imp_df = imp_df.sort_values('Importance', ascending=False)

    colors_imp = ['#3266ad' if i == 0 else
                  '#1D9E75' if i < 3 else '#aec6e8'
                  for i in range(len(imp_df))]
    axes[0].barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp[::-1])
    axes[0].invert_yaxis()
    axes[0].set_title(f'{best_name}\nBuilt-in Feature Importance')
    axes[0].set_xlabel('Importance')

elif hasattr(best_model, 'coef_'):
    coef = np.abs(best_model.coef_[0])
    imp_df = pd.DataFrame({'Feature': top_features, 'Importance': coef})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    colors_imp = ['#3266ad' if i == 0 else
                  '#1D9E75' if i < 3 else '#aec6e8'
                  for i in range(len(imp_df))]
    axes[0].barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp[::-1])
    axes[0].invert_yaxis()
    axes[0].set_title(f'{best_name}\n|Coefficient| Importance')
    axes[0].set_xlabel('|Coefficient|')

print(f"  Top features by model importance:")
for _, row in imp_df.iterrows():
    print(f"    {row['Feature']:<25} {row['Importance']:.4f}")

# ── Permutation importance (model-agnostic) ──────────────────
X_val_perm = X_val_sc if best_info['scaled'] else X_val
perm = permutation_importance(best_model, X_val_perm, y_val,
                              n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature'    : top_features,
    'Importance' : perm.importances_mean,
    'Std'        : perm.importances_std
}).sort_values('Importance', ascending=False)

axes[1].barh(perm_df['Feature'], perm_df['Importance'],
             xerr=perm_df['Std'], color='#5349b4', alpha=0.8,
             error_kw={'elinewidth': 1, 'capsize': 3})
axes[1].invert_yaxis()
axes[1].set_title(f'{best_name}\nPermutation Importance (validation set)')
axes[1].set_xlabel('Mean accuracy decrease')
axes[1].axvline(0, color='black', linewidth=0.8)

plt.suptitle('Phase 2 — Feature Importance Analysis', fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig('ph2_plot04_feature_importance.png', bbox_inches='tight')
plt.show()
print("  Saved - ph2_plot04_feature_importance.png")

# ── SHAP (if available) ──────────────────────────────────────
if HAS_SHAP and hasattr(best_model, 'feature_importances_'):
    print(f"\n  Computing SHAP values for {best_name}...")
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, feature_names=top_features,
                      show=False, plot_size=(10, 6))
    plt.title(f'SHAP Summary — {best_name}')
    plt.tight_layout()
    plt.savefig('ph2_plot05_shap.png', bbox_inches='tight')
    plt.show()
    print("  Saved - ph2_plot05_shap.png")

# ============================================================
# STEP 8 — FINAL EVALUATION ON TEST SET
# ============================================================
print(f"\n[8] Final evaluation on held-out TEST set — {best_name}...")

X_test_best = X_test_sc if best_info['scaled'] else X_test
test_pred  = best_model.predict(X_test_best)
test_proba = best_model.predict_proba(X_test_best)[:, 1]

test_acc = accuracy_score(y_test, test_pred)
test_f1  = f1_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_proba)

print(f"\n  Test Accuracy : {test_acc:.4f}")
print(f"  Test F1 Score : {test_f1:.4f}")
print(f"  Test ROC-AUC  : {test_auc:.4f}")
print(f"\n  Classification report:\n")
print(classification_report(y_test, test_pred,
                             target_names=['Not at-risk', 'At-risk']))

# ── Confusion matrix ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Not at-risk', 'At-risk'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n{best_name} — Test Set')

# ── Risk score distribution ──────────────────────────────────
axes[1].hist(test_proba[y_test == 0], bins=40, alpha=0.6,
             color='#3266ad', label='Not at-risk (true)', density=True)
axes[1].hist(test_proba[y_test == 1], bins=40, alpha=0.6,
             color='#E24B4A', label='At-risk (true)', density=True)
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold 0.5')
axes[1].set_xlabel('Predicted risk probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Risk Score Distribution\n(test set, colored by true label)')
axes[1].legend(fontsize=9)

plt.suptitle(f'Phase 2 — Final Test Evaluation  |  {best_name}',
             fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig('ph2_plot06_test_evaluation.png', bbox_inches='tight')
plt.show()
print("  Saved - ph2_plot06_test_evaluation.png")

# ============================================================
# STEP 9 — BIAS & FAIRNESS CHECK
# ============================================================
print("\n[9] Bias & fairness check by demographic group...")

# Add predictions back to test slice
test_df = X_test.copy()
test_df['at_risk_true']  = y_test.values
test_df['at_risk_pred']  = test_pred
test_df['risk_score']    = test_proba

fairness_cols = {
    'Gender'       : {0: 'Male',   1: 'Female'},
    'Internet'     : {0: 'No',     1: 'Yes'},
    'IncomeLevel'  : {0: 'Low',    1: 'Medium', 2: 'High'},
    'StressLevel'  : {0: 'Low',    1: 'Medium', 2: 'High'},
}

fairness_rows = []
for col, label_map in fairness_cols.items():
    if col not in test_df.columns:
        continue
    for val, label in label_map.items():
        grp = test_df[test_df[col] == val]
        if len(grp) < 10:
            continue
        grp_f1  = f1_score(grp['at_risk_true'], grp['at_risk_pred'], zero_division=0)
        grp_auc = roc_auc_score(grp['at_risk_true'], grp['risk_score']) \
                  if grp['at_risk_true'].nunique() > 1 else np.nan
        grp_tpr = (grp[(grp['at_risk_true']==1) & (grp['at_risk_pred']==1)].shape[0]
                   / max(grp[grp['at_risk_true']==1].shape[0], 1))
        fairness_rows.append({
            'Group'   : f'{col}={label}',
            'N'       : len(grp),
            'F1'      : round(grp_f1, 4),
            'ROC-AUC' : round(grp_auc, 4) if not np.isnan(grp_auc) else 'n/a',
            'Recall'  : round(grp_tpr, 4),
        })

fair_df = pd.DataFrame(fairness_rows)
print("\n" + fair_df.to_string(index=False))

# ── Plot: fairness bars ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(fair_df))
w = 0.3
ax.bar(x - w, fair_df['F1'],    width=w, label='F1',    color='#3266ad')
ax.bar(x,     fair_df['Recall'],width=w, label='Recall', color='#1D9E75')
ax.set_xticks(x)
ax.set_xticklabels(fair_df['Group'], rotation=30, ha='right', fontsize=9)
ax.axhline(f1_score(y_test, test_pred), color='#3266ad',
           linestyle='--', linewidth=1, alpha=0.6, label='Overall F1')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_title('Fairness Check — F1 & Recall by Demographic Group\n'
             '(large gaps indicate potential model bias)')
ax.legend()
plt.tight_layout()
plt.savefig('ph2_plot07_fairness.png', bbox_inches='tight')
plt.show()
print("\n  Saved - ph2_plot07_fairness.png")

# ============================================================
# STEP 10 — SAVE MODEL & ARTEFACTS
# ============================================================
print("\n[10] Saving model and artefacts...")

os.makedirs('model_artefacts', exist_ok=True)
with open('model_artefacts/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model_artefacts/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature list and metadata
meta = {
    'best_model_name' : best_name,
    'top_features'    : top_features,
    'test_accuracy'   : round(test_acc, 4),
    'test_f1'         : round(test_f1, 4),
    'test_roc_auc'    : round(test_auc, 4),
    'scaled_input'    : best_info['scaled'],
}
pd.Series(meta).to_json('model_artefacts/metadata.json', indent=2)

print("  Saved: model_artefacts/best_model.pkl")
print("  Saved: model_artefacts/scaler.pkl")
print("  Saved: model_artefacts/metadata.json")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Best model      : {best_name}
Test Accuracy   : {test_acc:.4f}
Test F1 Score   : {test_f1:.4f}
Test ROC-AUC    : {test_auc:.4f}

Top 3 predictors (by permutation importance):
  1. {perm_df.iloc[0]['Feature']}
  2. {perm_df.iloc[1]['Feature']}
  3. {perm_df.iloc[2]['Feature']}

Output plots:
  ph2_plot01_feature_selection.png
  ph2_plot02_model_comparison.png
  ph2_plot03_roc_prc.png
  ph2_plot04_feature_importance.png
  ph2_plot05_shap.png          (only if SHAP installed)
  ph2_plot06_test_evaluation.png
  ph2_plot07_fairness.png

Saved artefacts:
  model_artefacts/best_model.pkl
  model_artefacts/scaler.pkl
  model_artefacts/metadata.json

""")
print("=" * 60)
