import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import os
from datetime import datetime

# Set up the page configuration
st.set_page_config(page_title="PathFinder Pro", layout="wide")
st.title("🎓 PathFinder Pro – Career Guidance after 10th")
st.markdown("Helping students choose the right stream based on their interests, traits & goals.")

# Image path and display
image_path = os.path.join(os.getcwd(), "career.png")
st.image(image_path, use_column_width=True)

# Optional: Background image (avoid if image already used above)
# Commenting it to avoid confusion

# Load data and model
data = pd.read_csv("enhanced_student_survey_data.csv")
model = joblib.load("model/best_stream_model.pkl")
label_encoders = joblib.load("encoders/label_encoders.pkl")
stream_encoder = joblib.load("encoders/stream_encoder.pkl")

# Stream-career mapping
career_paths = {
    "Science": {
        "Engineer": "To become an engineer, excel in subjects like physics, mathematics, and problem-solving. Consider internships or workshops in mechanical, electrical, or civil engineering. A bachelor's degree in engineering is required.",
        "Doctor": "Pursuing medicine requires dedication and a strong understanding of biology and chemistry. A medical degree is mandatory. Consider pre-med programs or internships in hospitals.",
        "Data Scientist": "To pursue a career in data science, focus on developing skills in math, statistics, and computer science. Learn Python, R, and explore machine learning. A degree in computer science or a related field is needed.",
        "Researcher": "A researcher must have a passion for discovery. A career in research requires strong academic credentials, and it helps to pursue research internships and work on projects early on.",
        "Astronomer": "Astronomy is a science-driven field, requiring expertise in physics and mathematics. A degree in astrophysics or astronomy and postgraduate studies will help you establish a career in this field."
    },
    "Commerce": {
        "Accountant": "To become an accountant, focus on mathematics, economics, and financial management. A degree in accounting or finance will help you gain the necessary skills for this role.",
        "Economist": "Economics requires a deep understanding of mathematics, statistics, and societal trends. An undergraduate degree in economics or finance and strong analytical skills are key.",
        "Business Analyst": "A business analyst assesses companies' performance and identifies improvements. Focus on management, finance, and data analysis skills, and pursue a degree in business administration or related fields.",
        "Banker": "A banking career requires good analytical and financial skills. A degree in finance or economics and internships with banks or financial institutions are great starting points.",
        "Entrepreneur": "Entrepreneurs need creativity, business acumen, and risk-taking abilities. Focus on developing leadership and management skills through courses, workshops, and business incubators."
    },
    "Arts": {
        "Writer": "Writing careers require creativity, a strong grasp of language, and a passion for storytelling. Consider pursuing a degree in literature, journalism, or communications.",
        "Psychologist": "Psychology involves understanding human behavior. Consider pursuing a degree in psychology and seek internships in healthcare or educational settings to build practical skills.",
        "Historian": "To become a historian, you need a deep interest in history and research. A degree in history or anthropology is essential, along with strong research and writing abilities.",
        "Social Worker": "Social work focuses on helping communities and individuals. A degree in social work and a genuine interest in solving social issues are essential for this role.",
        "Graphic Designer": "A career in graphic design requires creativity and proficiency in design software. Consider taking courses in graphic design, fine arts, or multimedia arts."
    },
    "Vocational": {
        "Electrician": "Electricians need strong technical skills and knowledge of electrical systems. Vocational training or apprenticeships are key to gaining hands-on experience.",
        "Fashion Designer": "Fashion designing requires a creative mind and knowledge of design principles. A degree in fashion design along with internships in design houses is a good starting point.",
        "Chef": "Chefs need culinary expertise and creativity. Consider attending a culinary school and participating in internships at restaurants to hone your skills.",
        "Mechanic": "Mechanics need technical expertise in vehicle maintenance. Pursue vocational training or an apprenticeship in automotive repair.",
        "Hospitality Manager": "A career in hospitality requires good organizational and interpersonal skills. Consider pursuing a degree in hospitality management or tourism."
    }
}

# Ensure reports/ directory exists
if not os.path.exists("reports"):
    os.makedirs("reports")


# --- FUNCTION TO GENERATE PDF REPORT ---
def generate_pdf_report(fields, stream, domain_paths_with_description, report_filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "PathFinder Pro - Career Recommendation Report", ln=True, align='C')
    pdf.ln(10)

    # Section: Student Details
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Student Information", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    for label, value in fields:
        pdf.multi_cell(0, 10, f"{label}: {value}")
    pdf.ln(5)

    # Section: Recommended Stream
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recommended Stream", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"We recommend you to pursue {stream} based on your interests, skills, and preferences.")
    pdf.ln(10)

    # Section: Career Paths
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Possible Career Paths and Guidance", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    if domain_paths_with_description:
        for career, description in domain_paths_with_description.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, f"- {career}")
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, f"{description}")
            pdf.ln(5)
    else:
        pdf.multi_cell(0, 10, "No career information available for the selected stream.")

    report_path = os.path.join("reports", report_filename)
    pdf.output(report_path)
    return report_path


# --- STREAMLIT FORM ---
with st.form("career_form"):
    student_name = st.text_input("Student Name")
    interest = st.selectbox("Area of Interest", data['interest'].unique())
    subject = st.selectbox("Favorite Subject", data['favorite_subject'].unique())
    personality = st.selectbox("Personality Type", data['personality'].unique())
    grades = st.slider("Average Grade (%)", 40, 100, 75)
    learning_style = st.selectbox("Preferred Learning Style", data['learning_style'].unique())
    extra_curricular = st.selectbox("Extra-Curricular Interest", data['extra_curricular'].unique())
    decision_confidence = st.slider("Confidence in Decision (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([[interest, subject, personality, grades, learning_style, extra_curricular, decision_confidence]],
                              columns=['interest', 'favorite_subject', 'personality', 'grades', 'learning_style', 'extra_curricular', 'decision_confidence'])

    for col in ['interest', 'favorite_subject', 'personality', 'learning_style', 'extra_curricular']:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]
    stream = stream_encoder.inverse_transform([prediction])[0]

    st.success(f"🎯 Recommended Stream: **{stream}**")

    if stream in career_paths:
        st.markdown("### 🛣️ Possible Career Paths:")
        for career, description in career_paths[stream].items():
            st.markdown(f"#### {career}")
            st.markdown(f"{description}")

    # Prepare fields for PDF
    fields = [
        ("Student Name", student_name),
        ("Interest", interest),
        ("Favorite Subject", subject),
        ("Personality", personality),
        ("Average Grade (%)", grades),
        ("Preferred Learning Style", learning_style),
        ("Extra-Curricular Interest", extra_curricular),
        ("Confidence in Decision", decision_confidence),
    ]

    # Generate PDF
    report_filename = f"career_recommendation_report_{student_name.replace(' ', '_')}.pdf"
    report_path = generate_pdf_report(fields, stream, career_paths[stream], report_filename)

    # Download button
    with open(report_path, "rb") as f:
        st.download_button("📥 Download PDF Report", f, file_name=report_filename)


