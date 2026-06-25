# ============================================================
#  PHASE 3 — PAT DASHBOARD (Streamlit)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="PAT Risk Dashboard",
   # page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; }
.risk-high   { background:#fde8e8; border-left:4px solid #e53e3e;
               padding:10px 14px; border-radius:6px; margin:4px 0; }
.risk-medium { background:#fef3cd; border-left:4px solid #d97706;
               padding:10px 14px; border-radius:6px; margin:4px 0; }
.risk-low    { background:#e6f4ea; border-left:4px solid #38a169;
               padding:10px 14px; border-radius:6px; margin:4px 0; }
.section-header { font-size:1.1rem; font-weight:600;
                  color:#1a202c; margin:1rem 0 0.5rem; }
.info-box { background:#ebf8ff; border-left:4px solid #3182ce;
            padding:10px 14px; border-radius:6px; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    model    = pickle.load(open('model_artefacts/best_model.pkl', 'rb'))
    meta     = json.load(open('model_artefacts/metadata.json'))
    features = meta['top_features']
    return model, meta, features

@st.cache_data
def load_data(features):
    df = pd.read_csv('combined_dataset.csv')
    # Use DS1 rows for richer display (has more columns)
    ds1 = df[df['source'] == 'dataset1'].copy()

    # Decode encoded columns back to labels for display
    ds1['Gender_label']        = ds1['Gender'].map({0:'Male', 1:'Female'})
    ds1['Internet_label']      = ds1['Internet'].map({0:'No', 1:'Yes'})
    ds1['Extracurricular_label'] = ds1['Extracurricular'].map({0:'No', 1:'Yes'})
    ds1['IncomeLevel_label']   = ds1['IncomeLevel'].map({0:'Low', 1:'Medium', 2:'High'})
    ds1['StressLevel_label']   = ds1['StressLevel'].map({0:'Low', 1:'Medium', 2:'High'})
    ds1['ParentEducation_label'] = ds1['ParentEducation'].map({
        0:'High School', 1:'Some College', 2:"Bachelor's", 3:"Master's", 4:'PhD'
    })
    # Add fake student IDs for demo
    np.random.seed(42)
    ds1['Student_ID'] = ['S' + str(10000 + i) for i in range(len(ds1))]
    return ds1

model, meta, features = load_model()
df = load_data(features)

# Predict risk scores for all DS1 students
X_all = df[features].copy()
df['risk_score'] = model.predict_proba(X_all)[:, 1]
df['risk_pred']  = model.predict(X_all)
df['risk_label'] = pd.cut(
    df['risk_score'],
    bins=[0, 0.35, 0.65, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# ============================================================
# SIDEBAR — FILTERS & INFO
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=60)
    st.title("PAT Dashboard")
    st.caption("Personal Academic Tutor Support System")
    st.divider()

    st.markdown("**Filter Students**")
    risk_filter = st.multiselect(
        "Risk Level",
        options=['High Risk', 'Medium Risk', 'Low Risk'],
        default=['High Risk', 'Medium Risk']
    )
    dept_options = ['All'] + sorted(
        df['Department'].dropna().map(
            {0:'Business', 1:'Mathematics', 2:'Engineering', 3:'Computer Science'}
        ).unique().tolist()
    )
    dept_filter = st.selectbox("Department", dept_options)

    gender_filter = st.multiselect(
        "Gender",
        options=['Male', 'Female'],
        default=['Male', 'Female']
    )
    income_filter = st.multiselect(
        "Income Level",
        options=['Low', 'Medium', 'High'],
        default=['Low', 'Medium', 'High']
    )
    score_range = st.slider(
        "Attendance % range",
        min_value=0, max_value=100,
        value=(0, 100)
    )

    st.divider()
    st.markdown("**Model Info**")
    st.markdown(f"Model: `{meta['best_model_name']}`")
    st.markdown(f"ROC-AUC: `{meta['test_roc_auc']}`")
    st.markdown(f"F1 Score: `{meta['test_f1']}`")
    st.markdown(f"Accuracy: `{meta['test_accuracy']}`")
    st.divider()
    st.caption(" Predictions are decision-support tools. Always apply professional judgement.")

# ============================================================
# APPLY FILTERS
# ============================================================
dept_map_rev = {'Business':0, 'Mathematics':1, 'Engineering':2, 'Computer Science':3}

filtered = df[
    df['risk_label'].isin(risk_filter) &
    df['Gender_label'].isin(gender_filter) &
    df['IncomeLevel_label'].isin(income_filter) &
    df['Attendance'].between(score_range[0], score_range[1])
].copy()

if dept_filter != 'All':
    dept_code = dept_map_rev.get(dept_filter, -1)
    filtered = filtered[filtered['Department'] == dept_code]

# ============================================================
# MAIN CONTENT
# ============================================================
st.title(" Student At-Risk Monitoring Dashboard")
st.caption("Helping Personal Academic Tutors identify and support at-risk students early.")

# ── KPI Metrics ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total       = len(filtered)
high_risk   = (filtered['risk_label'] == 'High Risk').sum()
medium_risk = (filtered['risk_label'] == 'Medium Risk').sum()
low_risk    = (filtered['risk_label'] == 'Low Risk').sum()
avg_score   = filtered['risk_score'].mean()

col1.metric("Total Students",    total)
col2.metric(" High Risk",         high_risk,
            f"{high_risk/max(total,1)*100:.1f}%")
col3.metric(" Medium Risk",       medium_risk,
            f"{medium_risk/max(total,1)*100:.1f}%")
col4.metric(" Low Risk",          low_risk,
            f"{low_risk/max(total,1)*100:.1f}%")
col5.metric(" Avg Risk Score",    f"{avg_score:.2f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Student List",
    " Analytics",
    " Risk Predictor",
    " Fairness & Ethics"
])

# ============================================================
# TAB 1 — STUDENT LIST
# ============================================================
with tab1:
    st.markdown('<p class="section-header">At-Risk Student List</p>',
                unsafe_allow_html=True)

    col_sort, col_n = st.columns([2, 1])
    sort_by = col_sort.selectbox(
        "Sort by", ['risk_score', 'Attendance', 'FinalScore', 'MidtermScore'],
        format_func=lambda x: x.replace('_', ' ')
    )
    top_n = col_n.selectbox("Show top N", [25, 50, 100, 200, len(filtered)],
                             format_func=lambda x: f"Top {x}" if x != len(filtered) else "All")

    display_df = filtered.sort_values(sort_by, ascending=(sort_by == 'Attendance')).head(top_n)

    display_cols = {
        'Student_ID'          : 'Student ID',
        'risk_score'          : 'Risk Score',
        'risk_label'          : 'Risk Level',
        'FinalScore'          : 'Final Score',
        'MidtermScore'        : 'Midterm Score',
        'Attendance'          : 'Attendance %',
        'StudyHours'          : 'Study Hrs/Wk',
        'Gender_label'        : 'Gender',
        'IncomeLevel_label'   : 'Income Level',
        'Internet_label'      : 'Internet',
        'Extracurricular_label': 'Extracurricular',
    }

    table_df = display_df[list(display_cols.keys())].rename(columns=display_cols).copy()
    table_df['Risk Score'] = table_df['Risk Score'].round(3)

    def colour_risk(val):
        if val == 'High Risk':
            return 'background-color: #fde8e8; color: #c53030; font-weight:600'
        elif val == 'Medium Risk':
            return 'background-color: #fef3cd; color: #92400e; font-weight:600'
        return 'background-color: #e6f4ea; color: #276749; font-weight:600'

    def colour_score(val):
        try:
            if float(val) < 50:
                return 'color: #e53e3e; font-weight:600'
            elif float(val) < 65:
                return 'color: #d97706'
        except:
            pass
        return ''

    styled = (table_df.style
              .applymap(colour_risk, subset=['Risk Level'])
              .applymap(colour_score, subset=['Final Score', 'Midterm Score', 'Attendance %'])
              .format({'Risk Score': '{:.3f}',
                       'Final Score': '{:.1f}',
                       'Midterm Score': '{:.1f}',
                       'Attendance %': '{:.1f}',
                       'Study Hrs/Wk': '{:.1f}'}))

    st.dataframe(styled, use_container_width=True, height=450)

    # Download button
    csv = table_df.to_csv(index=False)
    st.download_button(
        " Download filtered list (CSV)",
        data=csv,
        file_name="at_risk_students.csv",
        mime="text/csv"
    )

    st.markdown(
        '<div class="info-box"> <b>PAT tip:</b> Sort by Risk Score and focus '
        'outreach on High Risk students first. Students with Attendance below 70% '
        'combined with low exam scores need immediate contact.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# TAB 2 — ANALYTICS
# ============================================================
with tab2:
    st.markdown('<p class="section-header">Risk Analytics Overview</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Risk level pie
    with c1:
        risk_counts = filtered['risk_label'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'High Risk':   '#e53e3e',
                'Medium Risk': '#d97706',
                'Low Risk':    '#38a169'
            },
            title="Risk Level Distribution",
            hole=0.45
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Risk score histogram
    with c2:
        fig_hist = px.histogram(
            filtered, x='risk_score',
            color='risk_label',
            color_discrete_map={
                'High Risk':   '#e53e3e',
                'Medium Risk': '#d97706',
                'Low Risk':    '#38a169'
            },
            nbins=40,
            title="Risk Score Distribution",
            labels={'risk_score': 'Risk Probability', 'count': 'Students'}
        )
        fig_hist.add_vline(x=0.65, line_dash='dash', line_color='#e53e3e',
                           annotation_text='High threshold')
        fig_hist.add_vline(x=0.35, line_dash='dash', line_color='#d97706',
                           annotation_text='Medium threshold')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter: Attendance vs Final Score coloured by risk
    c3, c4 = st.columns(2)
    with c3:
        sample = filtered.sample(min(500, len(filtered)), random_state=42)
        fig_scatter = px.scatter(
            sample, x='Attendance', y='FinalScore',
            color='risk_label',
            color_discrete_map={
                'High Risk':   '#e53e3e',
                'Medium Risk': '#d97706',
                'Low Risk':    '#38a169'
            },
            opacity=0.6, size_max=6,
            title="Attendance vs Final Score",
            labels={'Attendance': 'Attendance %', 'FinalScore': 'Final Score'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Avg risk score by income level
    with c4:
        income_risk = (filtered.groupby('IncomeLevel_label')['risk_score']
                       .mean().reindex(['Low','Medium','High']).reset_index())
        income_risk.columns = ['Income Level', 'Avg Risk Score']
        fig_income = px.bar(
            income_risk, x='Income Level', y='Avg Risk Score',
            color='Avg Risk Score',
            color_continuous_scale=['#38a169', '#d97706', '#e53e3e'],
            title="Average Risk Score by Income Level",
            range_y=[0, 1]
        )
        st.plotly_chart(fig_income, use_container_width=True)

    # Feature distributions by risk
    st.markdown('<p class="section-header">Key Features by Risk Group</p>',
                unsafe_allow_html=True)

    feat_col = st.selectbox(
        "Select feature to explore",
        ['FinalScore', 'MidtermScore', 'Attendance', 'StudyHours', 'AssignmentCompletion']
    )
    fig_box = px.box(
        filtered, x='risk_label', y=feat_col,
        color='risk_label',
        color_discrete_map={
            'High Risk':   '#e53e3e',
            'Medium Risk': '#d97706',
            'Low Risk':    '#38a169'
        },
        category_orders={'risk_label': ['Low Risk', 'Medium Risk', 'High Risk']},
        title=f'{feat_col} by Risk Group',
        points='outliers'
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# TAB 3 — RISK PREDICTOR (single student)
# ============================================================
with tab3:
    st.markdown('<p class="section-header">Predict Risk for a New Student</p>',
                unsafe_allow_html=True)
    st.caption("Enter a student's details below to get an instant risk prediction.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Academic Scores**")
            final_score   = st.slider("Final Exam Score",      0,  100, 65)
            midterm_score = st.slider("Midterm Exam Score",    0,  100, 60)
            attend        = st.slider("Attendance (%)",         0,  100, 75)
            assign_avg    = st.slider("Assignment Completion (%)", 50, 100, 75)

        with c2:
            st.markdown("**Study Habits**")
            study_hrs = st.slider("Study Hours per Week", 0, 40, 15)
            st.caption(" Minor model impact (1.3%) — shown for context only")
            age       = st.number_input("Age", min_value=17, max_value=35, value=20)

        with c3:
            st.markdown("**Demographics**")
            gender       = st.selectbox("Gender",         ['Male', 'Female'])
            internet     = st.selectbox("Internet Access", ['Yes', 'No'])
            extracurr    = st.selectbox("Extracurricular", ['Yes', 'No'])
            parent_edu   = st.selectbox("Parent Education",
                                        ["High School", "Some College",
                                         "Bachelor's", "Master's", "PhD"])
            income       = st.selectbox("Family Income Level", ['Low', 'Medium', 'High'])

        submitted = st.form_submit_button(" Predict Risk", use_container_width=True)

    if submitted:
        # Build input row

        input_data = {
            'FinalScore'           : final_score,
            'MidtermScore'         : midterm_score,
            'Attendance'           : attend,
            'Internet'             : 1 if internet == 'Yes' else 0,
            'Gender'               : 1 if gender == 'Female' else 0,
            'StudyHours'           : study_hrs,
            'ParentEducation'      : {"High School":0,"Some College":1,
                                    "Bachelor's":2,"Master's":3,"PhD":4}[parent_edu],
            'AssignmentCompletion' : assign_avg,
            'Extracurricular'      : 1 if extracurr == 'Yes' else 0,
            'IncomeLevel'          : {'Low':0, 'Medium':1, 'High':2}[income],
        }
        input_df = pd.DataFrame([input_data])[features]
        risk_prob = model.predict_proba(input_df)[0][1]
        risk_pred = model.predict(input_df)[0]

        # Display result
        st.divider()
        r1, r2, r3 = st.columns([1, 2, 1])

        with r2:
            if risk_prob >= 0.65:
                level, colour, emoji, advice = (
                    'HIGH RISK', '#e53e3e', 'RISK LEVEL',
                    'Immediate outreach recommended. Schedule a meeting with this student '
                    'and connect them with academic support services.'
                )
            elif risk_prob >= 0.35:
                level, colour, emoji, advice = (
                    'MEDIUM RISK', '#d97706', 'RISK LEVEL',
                    'Monitor closely. Consider a check-in email and review attendance '
                    'patterns over the next two weeks.'
                )
            else:
                level, colour, emoji, advice = (
                    'LOW RISK', '#38a169', 'RISK LEVEL',
                    'Student appears on track. Continue routine monitoring.'
                )

            st.markdown(f"""
            <div style='text-align:center; padding:24px; background:#f8f9fa;
                        border-radius:12px; border: 2px solid {colour}'>
                <div style='font-size:3rem'>{emoji}</div>
                <div style='font-size:1.8rem; font-weight:700; color:{colour}'>{level}</div>
                <div style='font-size:2.5rem; font-weight:600; margin:8px 0'>
                    Risk Score: {risk_prob:.1%}
                </div>
                <div style='font-size:0.95rem; color:#555; margin-top:12px'>{advice}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_prob * 100,
                number={'suffix': '%', 'font': {'size': 32}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': colour},
                    'steps': [
                        {'range': [0, 35],  'color': '#e6f4ea'},
                        {'range': [35, 65], 'color': '#fef3cd'},
                        {'range': [65, 100],'color': '#fde8e8'},
                    ],
                    'threshold': {
                        'line': {'color': colour, 'width': 4},
                        'thickness': 0.75,
                        'value': risk_prob * 100
                    }
                },
                title={'text': "Risk Probability"}
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=50,b=10,l=30,r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Key factors
        st.markdown('<p class="section-header">Key Risk Factors for This Student</p>',
                    unsafe_allow_html=True)


        factors = []

        #  Final Score (50% importance)
        if final_score < 60:
            factors.append((f"Final Score is {final_score} — high risk (below 60)", "high"))
        elif final_score < 70:
            factors.append((f"Final Score is {final_score} — moderate concern (below 70)", "medium"))
        else:
            factors.append((f"Final Score is {final_score} — acceptable range", "low"))

        # Midterm Score (22% importance)
        if midterm_score < 60:
            factors.append((f"Midterm Score is {midterm_score} — high risk (below 60)", "high"))
        elif midterm_score < 70:
            factors.append((f"Midterm Score is {midterm_score} — moderate concern (below 70)", "medium"))
        else:
            factors.append((f"Midterm Score is {midterm_score} — acceptable range", "low"))

        #  Parent Education (14% importance)
        if parent_edu == "High School":
            factors.append(("No parental higher education — student may need extra academic guidance", "high"))
        elif parent_edu == "Some College":
            factors.append(("Limited parental higher education background", "medium"))
        else:
            factors.append((f"Parent education: {parent_edu} — positive background factor", "low"))

        #  Attendance (11% importance) 
        if attend < 65:
            factors.append((f"Attendance is {attend}% — critically low (below 65%)", "high"))
        elif attend < 75:
            factors.append((f"Attendance is {attend}% — below recommended 75%", "medium"))
        else:
            factors.append((f"Attendance is {attend}% — satisfactory", "low"))

        #  Assignment Completion (minor)
        if assign_avg < 60:
            factors.append((f"Assignment completion {assign_avg}% — very low", "medium"))

        # Internet access (minor) 
        if internet == 'No':
            factors.append(("No home internet access — may limit study resources", "medium"))

        # Study Hours (informational)
        if study_hrs < 8:
            factors.append((f"Study hours very low ({study_hrs} hrs/week) — consider study skills support", "medium"))
        elif study_hrs > 30:
            factors.append((f"Study hours very high ({study_hrs} hrs/week) — may indicate student is struggling", "medium"))

        # Overall model explanation
        if risk_prob >= 0.65:
            dominant = []
            if final_score < 70:   dominant.append(f"low Final Score ({final_score})")
            if midterm_score < 70: dominant.append(f"low Midterm Score ({midterm_score})")
            if attend < 75:        dominant.append(f"low Attendance ({attend}%)")
            if parent_edu in ["High School", "Some College"]:
                dominant.append(f"parental education ({parent_edu})")
            if dominant:
                st.warning(f"High risk driven primarily by: **{', '.join(dominant)}**")

        for text, level in factors:
            css = f"risk-{level}"
            st.markdown(f'<div class="{css}">{text}</div>',
                        unsafe_allow_html=True)
            
# ============================================================
# TAB 4 — FAIRNESS & ETHICS
# ============================================================
with tab4:
    st.markdown('<p class="section-header">Model Fairness & Ethical Considerations</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box"> This section analyses whether the model performs '
        'equally across different student groups. Large gaps in recall or precision '
        'across groups indicate potential bias that should be addressed before '
        'deploying this system.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Compute fairness metrics per group
    group_configs = {
        'Gender'       : ('Gender_label',      ['Male', 'Female']),
        'Income Level' : ('IncomeLevel_label',  ['Low', 'Medium', 'High']),
        'Internet'     : ('Internet_label',     ['Yes', 'No']),
        'Stress Level' : ('StressLevel_label',  ['Low', 'Medium', 'High']),
    }

    fairness_rows = []
    for group_name, (col, vals) in group_configs.items():
        for val in vals:
            grp = df[df[col] == val]
            if len(grp) < 20:
                continue
            tp = ((grp['risk_pred'] == 1) & (grp['at_risk'] == 1)).sum()
            fn = ((grp['risk_pred'] == 0) & (grp['at_risk'] == 1)).sum()
            fp = ((grp['risk_pred'] == 1) & (grp['at_risk'] == 0)).sum()
            recall    = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            fairness_rows.append({
                'Group'    : group_name,
                'Subgroup' : val,
                'N'        : len(grp),
                'Recall'   : round(recall, 3),
                'Precision': round(precision, 3),
                'F1'       : round(f1, 3),
                'Avg Risk' : round(grp['risk_score'].mean(), 3),
            })

    fair_df = pd.DataFrame(fairness_rows)

    # Plot fairness metrics
    f1, f2 = st.columns(2)
    with f1:
        fig_recall = px.bar(
            fair_df, x='Subgroup', y='Recall',
            color='Group', barmode='group',
            title='Recall by Subgroup<br><sup>Higher = more at-risk students correctly identified</sup>',
            range_y=[0, 1.1]
        )
        fig_recall.add_hline(y=0.8, line_dash='dash', line_color='gray',
                             annotation_text='80% target')
        st.plotly_chart(fig_recall, use_container_width=True)

    with f2:
        fig_f1 = px.bar(
            fair_df, x='Subgroup', y='F1',
            color='Group', barmode='group',
            title='F1 Score by Subgroup<br><sup>Balance of precision and recall</sup>',
            range_y=[0, 1.1]
        )
        fig_f1.add_hline(y=0.8, line_dash='dash', line_color='gray',
                         annotation_text='80% target')
        st.plotly_chart(fig_f1, use_container_width=True)

    # Fairness table
    st.markdown('<p class="section-header">Fairness Metrics Table</p>',
                unsafe_allow_html=True)

    def flag_bias(val):
        if isinstance(val, float) and val < 0.70:
            return 'background-color:#fde8e8; color:#c53030; font-weight:600'
        elif isinstance(val, float) and val < 0.80:
            return 'background-color:#fef3cd; color:#92400e'
        return ''

    styled_fair = (fair_df.style
                   .applymap(flag_bias, subset=['Recall', 'Precision', 'F1'])
                   .format({'Recall':'{:.3f}','Precision':'{:.3f}',
                            'F1':'{:.3f}','Avg Risk':'{:.3f}'}))
    st.dataframe(styled_fair, use_container_width=True)

    # Ethical guidelines
    st.markdown('<p class="section-header">Ethical Use Guidelines for PATs</p>',
                unsafe_allow_html=True)

    guidelines = [
        (" Transparency",
         "Always inform students if they have been flagged as at-risk by an AI system. "
         "They have the right to understand how the decision was made."),
        (" Human oversight",
         "Predictions are decision-support tools, not final decisions. Every flag "
         "should be reviewed by a PAT before any action is taken."),
        (" Fairness monitoring",
         "Regularly audit model performance across demographic groups. If recall "
         "drops below 75% for any group, retrain or adjust the model."),
        (" Data privacy",
         "Student data must be handled in compliance with GDPR and institutional "
         "data protection policies. Do not share individual risk scores externally."),
        (" Continuous improvement",
         "Retrain the model each academic year with new outcome data to prevent "
         "model drift and maintain prediction accuracy."),
        (" Avoiding self-fulfilling prophecies",
         "Ensure that being flagged as at-risk does not lead to reduced expectations "
         "or differential treatment that disadvantages the student further."),
    ]

    for i in range(0, len(guidelines), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(guidelines):
                title, text = guidelines[i + j]
                col.markdown(
                    f'<div class="risk-low" style="background:#f0f9ff;'
                    f'border-left-color:#3182ce">'
                    f'<b>{title}</b><br><span style="font-size:0.88rem">'
                    f'{text}</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown("")
