import streamlit as st
import pandas as pd
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()


def load_guardian_model():
    if 'guardian_model' not in st.session_state:
        st.write("Loading the Granite Guardian model for the first time...")
        model_path_name = "ibm-granite/granite-guardian-3.1-2b"
        st.session_state.guardian_model = AutoModelForCausalLM.from_pretrained(model_path_name)
        st.session_state.guardian_tokenizer = AutoTokenizer.from_pretrained(model_path_name)
        device = torch.device("cpu")
        st.session_state.guardian_model = st.session_state.guardian_model.to(device).eval()
        st.write("Model loaded successfully.")
    return st.session_state.guardian_model, st.session_state.guardian_tokenizer

if 'model_loaded' not in st.session_state:
    guardian_model, guardian_tokenizer = load_guardian_model()
    st.session_state.model_loaded = True
else:
    guardian_model, guardian_tokenizer = st.session_state.guardian_model, st.session_state.guardian_tokenizer

safe_token = "No"
unsafe_token = "Yes"


def parse_output(output, input_len):
    label, prob_of_risk = None, None
    if output.scores:
        list_index_logprobs_i = [torch.topk(token_i, k=20, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i:
            prob = get_probablities(list_index_logprobs_i)
            prob_of_risk = prob[1]
    res = guardian_tokenizer.decode(output.sequences[:, input_len:][0], skip_special_tokens=True).strip()
    if unsafe_token.lower() == res.lower():
        label = unsafe_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"
    return label, prob_of_risk.item() if prob_of_risk is not None else 0.0

def get_probablities(logprobs):
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = guardian_tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += math.exp(logprob)
    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )
    return probabilities

def detect_risk(text, risk_type="social bias"):
    messages = [{"role": "system", "content": f"Detect {risk_type} in the following text."}, {"role": "user", "content": text}]
    input_ids = guardian_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(torch.device("cpu"))
    with torch.no_grad():
        output = guardian_model.generate(input_ids, do_sample=False, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
    label, prob_of_risk = parse_output(output, input_ids.shape[1])
    return label == "Yes", prob_of_risk

def generate_employee_description(row, report_model):
    satisfaction = row["SatisfactionScore"] * 20
    performance = row["PerformanceRating"]
    supervisor_feedback = row["SupervisorFeedback"]
    peer_feedback = row["PeerFeedback"]
    employee_qna = row["EmployeeQnA"]
    prompt = f"""
    Based on an employee's data:
    - Satisfaction: {satisfaction}%
    - Performance: {performance}/5
    - Supervisor Feedback: {supervisor_feedback}
    - Peer Feedback: {peer_feedback}
    - Employee Q&A: {employee_qna}
    Generate a concise summary in bullet points for an HR manager:
    - Strengths: Integrate 1 point from Q&A and external feedback.
    - Areas for Improvement: Use 1 point from Q&A and external feedback.
    - Actionable KPIs: 2 specific and brief suggestions.
    Use a professional tone. Maximum 150 words.
    """
    return report_model.generate_text(prompt)

def generate_department_report(df, department, report_model):
    dept_data = df[df["Department"] == department]
    num_employees = len(dept_data)
    avg_satisfaction = dept_data["SatisfactionScore"].mean() * 20
    avg_performance = dept_data["PerformanceRating"].mean()
    high_risk = len(dept_data[dept_data["SatisfactionScore"] < 3])
    high_satisfaction = len(dept_data[dept_data["SatisfactionScore"] >= 4])
    prompt = f"""
    Based on aggregated data for the '{department}' department:
    - Employees: {num_employees}
    - Average Satisfaction: {avg_satisfaction:.1f}%
    - Average Performance: {avg_performance:.1f}/5
    - At Risk (<60% satisfaction): {high_risk}
    - High Satisfaction (>=80%): {high_satisfaction}
    Generate a concise report in bullet points for the manager:
    - Summary: General team status in 1 sentence.
    - Trends: 1-2 key observations.
    - Actions: 2 specific recommendations.
    Use a professional tone. Maximum 100 words.
    """
    return report_model.generate_text(prompt)


def main_page():
    # Logo and title in columns
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("R.png", width=100)  # Small logo
    with col2:
        st.title("Granite - HR Performance")

    # Container for API credentials input
    with st.container(border=True):
        st.subheader("Enter IBM WatsonX Credentials")
        
        # Verificar si las credenciales ya están en session_state
        if 'api_key' not in st.session_state or 'project_id' not in st.session_state:
            api_key = st.text_input("API Key", type="password", value="")
            project_id = st.text_input("Project ID", value="")
            if st.button("Submit Credentials"):
                if api_key and project_id:
                    try:
                        report_model = ModelInference(
                            model_id="ibm/granite-3-8b-instruct",
                            credentials={"api_key": api_key, "url": "https://us-south.ml.cloud.ibm.com"},
                            project_id=project_id,
                            params={GenParams.MAX_NEW_TOKENS: 600, GenParams.TEMPERATURE: 0.2}
                        )
                        st.session_state.api_key = api_key
                        st.session_state.project_id = project_id
                        st.session_state.report_model = report_model
                        st.success("Credentials validated and model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model with provided credentials: {str(e)}")
                else:
                    st.warning("Please enter both API Key and Project ID.")
        else:
            # Si ya están en session_state, usarlas directamente
            st.write("Credentials already provided.")
            if 'report_model' not in st.session_state:
                try:
                    report_model = ModelInference(
                        model_id="ibm/granite-3-8b-instruct",
                        credentials={"api_key": st.session_state.api_key, "url": "https://us-south.ml.cloud.ibm.com"},
                        project_id=st.session_state.project_id,
                        params={GenParams.MAX_NEW_TOKENS: 600, GenParams.TEMPERATURE: 0.2}
                    )
                    st.session_state.report_model = report_model
                except Exception as e:
                    st.error(f"Error reloading model: {str(e)}")
                    del st.session_state.api_key
                    del st.session_state.project_id
                    st.rerun()


    if 'report_model' not in st.session_state:
        return


    with st.container(border=True):
        st.subheader("Load Data")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("Dataset loaded:")
            st.dataframe(df)
        elif 'df' not in st.session_state:
            example_data = {
                "EmployeeID": ["E001", "E002"],
                "Department": ["IT", "IT"],
                "PerformanceRating": [4, 3],
                "SatisfactionScore": [3, 4],
                "SupervisorFeedback": ["Demonstrates strong technical skills and leadership; however, should improve team communication.", "Good performance but needs more proactivity."],
                "PeerFeedback": ["Very collaborative, though occasionally distant and reserved.", "Friendly and reliable."],
                "Attrition": [0, 0],
                "HireDate": ["2018-06-15", "2019-03-10"],
                "YearsOfService": [6, 5],
                "EducationLevel": ["Bachelor's", "Master's"],
                "JobRole": ["Software Engineer", "Data Analyst"],
                "EmployeeQnA": ["Q: What motivates you in your current job? A: The opportunity to solve complex problems and learn new technologies. Q: How would you describe your team's work environment? A: Collaborative but sometimes distant.", "Q: What motivates you in your current job? A: Analyzing data to make decisions. Q: How would you describe your team's work environment? A: Positive and dynamic."]
            }
            df = pd.DataFrame(example_data)
            st.session_state.df = df
            st.write("Using example dataset:")
            st.dataframe(df)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        report_model = st.session_state.report_model
        

        with st.container(border=True):
            st.subheader("Individual Report")
            employee_ids = df["EmployeeID"].tolist()
            selected_employee = st.selectbox("Select an employee", employee_ids, key="employee_select")
            if st.button("Generate Individual Report"):
                st.session_state.selected_employee = selected_employee
                st.session_state.page = "individual_report"
                st.rerun()
        

        with st.container(border=True):
            st.subheader("Department Manager Report")
            departments = df["Department"].unique().tolist()
            selected_department = st.selectbox("Select a department", departments, key="department_select")
            if st.button("Generate Department Report"):
                st.session_state.selected_department = selected_department
                st.session_state.page = "department_report"
                st.rerun()
    

    with st.container(border=True):
        st.write("""
        ### Instructions:
        1. Enter your IBM WatsonX API Key and Project ID.
        2. Upload a CSV file with columns such as `EmployeeID`, `Department`, `SatisfactionScore`, `PerformanceRating`, `SupervisorFeedback`, `PeerFeedback`, `EmployeeQnA`, `HireDate`, `YearsOfService`, `Attrition` (optional).
        3. Select an employee or department and generate the corresponding report.
        """)

def individual_report_page():
    if 'df' not in st.session_state or 'selected_employee' not in st.session_state or 'report_model' not in st.session_state:
        st.error("No dataset loaded, employee selected, or credentials provided. Return to the main page.")
        if st.button("Return to Main Page"):
            st.session_state.page = "main"
            st.rerun()
        return
    
    df = st.session_state.df
    selected_employee = st.session_state.selected_employee
    report_model = st.session_state.report_model
    st.title(f"Report for {selected_employee}")
    
    employee_row = df[df["EmployeeID"] == selected_employee].iloc[0]
    

    supervisor_bias, sup_prob = detect_risk(employee_row["SupervisorFeedback"], "social bias")
    peer_bias, peer_prob = detect_risk(employee_row["PeerFeedback"], "social bias")
    qna_bias, qna_prob = detect_risk(employee_row["EmployeeQnA"], "social bias")
    

    description = generate_employee_description(employee_row, report_model)
    

    st.write("### Bias Detection Results")
    st.write(f"- Supervisor Feedback: {'Possible bias detected' if supervisor_bias else 'No bias detected'} (probability: {sup_prob:.3f})")
    st.write(f"- Peer Feedback: {'Possible bias detected' if peer_bias else 'No bias detected'} (probability: {peer_prob:.3f})")
    st.write(f"- Employee Q&A: {'Possible bias detected' if qna_bias else 'No bias detected'} (probability: {qna_prob:.3f})")
    
    st.write("### Generated Report")
    st.write(description)
    

    st.write("### Employee Visualizations")
    dept = employee_row["Department"]
    dept_data = df[df["Department"] == dept]
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.barplot(x=["Employee", "Department Average"], 
                y=[employee_row["SatisfactionScore"], dept_data["SatisfactionScore"].mean()], ax=ax)
    ax.set_title(f"Satisfaction: {selected_employee} vs {dept}")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.scatterplot(x="SatisfactionScore", y="PerformanceRating", hue="EmployeeID", data=dept_data, ax=ax)
    ax.scatter(employee_row["SatisfactionScore"], employee_row["PerformanceRating"], color="red", label=selected_employee, s=100)
    ax.legend()
    ax.set_title(f"Satisfaction vs Performance in {dept}")
    st.pyplot(fig)
    
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()

def department_report_page():
    if 'df' not in st.session_state or 'selected_department' not in st.session_state or 'report_model' not in st.session_state:
        st.error("No dataset loaded, department selected, or credentials provided. Return to the main page.")
        if st.button("Return to Main Page"):
            st.session_state.page = "main"
            st.rerun()
        return
    
    df = st.session_state.df
    selected_department = st.session_state.selected_department
    report_model = st.session_state.report_model
    st.title(f"Report for {selected_department} Manager")
    
    dept_report = generate_department_report(df, selected_department, report_model)
    st.write(dept_report)
    
    # Visualizations
    st.write("### Department Visualizations")
    dept_data = df[df["Department"] == selected_department]
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.boxplot(x="Department", y="SatisfactionScore", data=dept_data, ax=ax)
    ax.set_title(f"Satisfaction Distribution in {selected_department}")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.scatterplot(x="SatisfactionScore", y="PerformanceRating", hue="EmployeeID", data=dept_data, ax=ax)
    ax.set_title(f"Satisfaction vs Performance in {selected_department}")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.lineplot(x="YearsOfService", y="SatisfactionScore", data=dept_data, ax=ax)
    ax.set_title(f"Satisfaction by Years of Service in {selected_department}")
    st.pyplot(fig)
    
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()


if 'page' not in st.session_state:
    st.session_state.page = "main"

page_functions = {
    "main": main_page,
    "individual_report": individual_report_page,
    "department_report": department_report_page
}


page_functions[st.session_state.page]()
