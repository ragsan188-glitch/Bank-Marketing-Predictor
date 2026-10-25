from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="TermWise | Bank Marketing Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "bank_marketing_knn.pkl"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.toggle("Dark theme", key="dark_mode")

if dark:
    page_bg, panel_bg, input_bg = "#0b1020", "#131b2e", "#0f172a"
    text, muted, border = "#eef2ff", "#aab5cc", "#2b3854"
    accent, result_bg = "#7c9cff", "#172744"
else:
    page_bg, panel_bg, input_bg = "#f5f7fb", "#ffffff", "#ffffff"
    text, muted, border = "#172033", "#5e6b82", "#dce3ef"
    accent, result_bg = "#315ee8", "#edf3ff"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {"dark" if dark else "light"}; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background: {page_bg}; color: {text}; }}
    [data-testid="stHeader"] {{ background: {page_bg}; }}
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ visibility: hidden; }}
    h1, h2, h3, h4, p, li, label, [data-testid="stMarkdownContainer"] {{ color: {text}; }}
    .hero {{
      background: linear-gradient(135deg, {panel_bg}, {result_bg});
      border: 1px solid {border}; border-radius: 20px;
      padding: 28px 30px; margin: 8px 0 24px 0;
    }}
    .eyebrow {{
      color: {accent}; font-size: .78rem; font-weight: 750;
      letter-spacing: .14em; text-transform: uppercase; margin-bottom: 12px;
    }}
    .hero h1 {{ font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.08; margin: 0 0 12px 0; }}
    .hero p {{ color: {muted}; font-size: 1.02rem; margin: 0; max-width: 780px; }}
    .muted {{ color: {muted} !important; }}
    .panel {{
      background: {panel_bg}; border: 1px solid {border};
      border-radius: 16px; padding: 20px 22px; margin-bottom: 16px;
    }}
    .panel h3 {{ margin-top: 0; }}
    .result {{
      background: {result_bg}; border: 1px solid {accent};
      border-radius: 16px; padding: 22px; margin: 20px 0;
    }}
    .result-label {{ color: {muted}; text-transform: uppercase; font-size: .76rem; letter-spacing: .12em; font-weight: 750; }}
    .result-value {{ color: {text}; font-size: clamp(1.5rem, 3vw, 2.15rem); font-weight: 800; margin: 9px 0; }}
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {{
      background: {input_bg} !important; color: {text} !important; border-color: {border} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{ background: {panel_bg} !important; color: {text} !important; }}
    [data-baseweb="menu"] * {{ color: {text} !important; }}
    [data-testid="stForm"] {{
      background: {panel_bg}; border: 1px solid {border};
      border-radius: 16px; padding: 22px;
    }}
    .stButton button, [data-testid="stFormSubmitButton"] button {{
      background: {accent}; color: white !important; border: 0;
      border-radius: 9px; font-weight: 700; min-height: 3rem; width: 100%;
    }}
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{ filter: brightness(1.08); border: 0; }}
    [data-testid="stCaptionContainer"] {{ color: {muted}; }}
    a {{ color: {accent} !important; }}
    hr {{ border-color: {border}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine Learning Project</div>
      <h1>TermWise</h1>
      <p>Bank marketing subscription predictor. Enter a campaign profile to see the classification produced by the trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Place your trained bank_marketing_knn.pkl beside app.py.")
    st.code("termwise_bank_marketing_clean/\n├── app.py\n├── bank_marketing_knn.pkl\n└── requirements.txt")
    st.stop()

@st.cache_resource
def load_model(path_string: str):
    return joblib.load(path_string)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The saved model could not be loaded. Check that this is the fitted scikit-learn pipeline used for the project.")
    st.exception(exc)
    st.stop()

st.markdown("## Client profile")
st.markdown('<p class="muted">Complete the fields, then select <b>Predict subscription</b>. You can submit new values repeatedly without refreshing the page.</p>', unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=17, max_value=100, value=35, step=1)
        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
        month = st.selectbox("Contact month", ["may", "jun", "jul", "aug", "oct", "nov", "dec", "mar", "apr", "sep"])
        day_of_week = st.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"])
        campaign = st.number_input("Contacts in this campaign", min_value=1, max_value=100, value=1, step=1)
    with c2:
        education = st.selectbox("Education", ["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "illiterate", "unknown"])
        default = st.selectbox("Has credit in default?", ["no", "yes", "unknown"])
        housing = st.selectbox("Has housing loan?", ["yes", "no", "unknown"])
        pdays = st.number_input("Days since previous contact (999 = never)", min_value=0, max_value=999, value=999, step=1)
        previous = st.number_input("Contacts before this campaign", min_value=0, max_value=50, value=0, step=1)
        emp_var_rate = st.number_input("Employment variation rate", value=0.0, step=0.1, format="%.2f")
    with c3:
        loan = st.selectbox("Has personal loan?", ["no", "yes", "unknown"])
        contact = st.selectbox("Contact type", ["cellular", "telephone"])
        poutcome = st.selectbox("Previous campaign outcome", ["nonexistent", "failure", "success"])
        cons_price_idx = st.number_input("Consumer price index", value=93.994, step=0.001, format="%.3f")
        cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4, step=0.1, format="%.1f")
        euribor3m = st.number_input("Euribor 3-month rate", value=4.857, step=0.001, format="%.3f")
    nr_employed = st.number_input("Number of employees (thousands)", value=5191.0, step=1.0, format="%.1f")
    submitted = st.form_submit_button("Predict subscription", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "age": int(age), "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day_of_week, "campaign": int(campaign),
        "pdays": int(pdays), "previous": int(previous), "poutcome": poutcome,
        "emp.var.rate": float(emp_var_rate), "cons.price.idx": float(cons_price_idx),
        "cons.conf.idx": float(cons_conf_idx), "euribor3m": float(euribor3m),
        "nr.employed": float(nr_employed),
    }])
    try:
        prediction = int(model.predict(row)[0])
        st.session_state["termwise_result"] = {"prediction": prediction, "inputs": row.iloc[0].to_dict()}
    except Exception as exc:
        st.error("Prediction failed. The saved pipeline may expect a different set of input columns.")
        st.exception(exc)

result = st.session_state.get("termwise_result")
if result:
    if result["prediction"] == 1:
        title = "Potential subscriber"
        detail = "The model predicts that this client may subscribe to the term deposit."
    else:
        title = "Less likely to subscribe"
        detail = "The model predicts that this client is less likely to subscribe to the term deposit."
    st.markdown(
        f'<div class="result"><div class="result-label">Latest model prediction</div><div class="result-value">{title}</div><div class="muted">{detail}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("This is a model classification, not a guarantee. Results depend on the training data and selected inputs.")

st.markdown("---")
st.markdown("## About the model")
left, right = st.columns(2)
with left:
    st.markdown(
        """
        <div class="panel">
          <h3>How it works</h3>
          <p class="muted">This demo loads the fitted K-Nearest Neighbors (KNN) pipeline saved in <code>bank_marketing_knn.pkl</code>. The pipeline preprocesses numeric and categorical fields before predicting the target class.</p>
          <p class="muted">Target: whether a client subscribed to a term deposit (<code>yes</code> or <code>no</code>).</p>
        </div>
        """, unsafe_allow_html=True
    )
with right:
    st.markdown(
        """
        <div class="panel">
          <h3>Limitations</h3>
          <p class="muted">This is a portfolio demonstration, not a production banking tool. The model can be wrong and may reflect patterns or bias in historical data. Do not use it as the sole basis for real customer decisions.</p>
          <p class="muted">The call duration feature is excluded because it is not known before the call.</p>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("## Contact")
st.markdown(
    '[LinkedIn](https://www.linkedin.com/in/ragul-y-s-a05253394/)  ·  '
    '[Email](mailto:ragsan188@gmail.com)'
)
st.markdown(
    '''
    <div class="panel">
      <h3>Enjoyed using TermWise or have suggestions for improvement?</h3>
      <p class="muted">I'd love to hear your feedback. If you have any suggestions, questions, or ideas to improve this project, feel free to reach out through LinkedIn or email.</p>
      <p class="muted">Your feedback is always welcome.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)
st.caption("TermWise · Bank Marketing Prediction Project")
