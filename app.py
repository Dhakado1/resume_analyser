from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

app = Flask(__name__)

# -------------------- Load Models --------------------
try:
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
except:
    rf_classifier_categorization = None
    tfidf_vectorizer_categorization = None

try:
    rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
except:
    rf_classifier_job_recommendation = None
    tfidf_vectorizer_job_recommendation = None


# -------------------- Clean Resume --------------------
def cleanResume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'[^A-Za-z0-9\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


# -------------------- PDF to Text --------------------
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += (page.extract_text() or '') + '\n'
    return text


# -------------------- Validation --------------------
def is_valid_resume(text):
    keywords = [
        "education", "skills", "experience",
        "project", "internship", "b.tech",
        "phone", "email", "linkedin"
    ]

    text = text.lower()
    match_count = sum(1 for word in keywords if word in text)

    return match_count >= 3


# -------------------- Prediction --------------------
def predict_category(text):
    if rf_classifier_categorization and tfidf_vectorizer_categorization:
        text = cleanResume(text)
        vector = tfidf_vectorizer_categorization.transform([text])
        return rf_classifier_categorization.predict(vector)[0]
    return "Model not available"


def job_recommendation(text):
    if rf_classifier_job_recommendation and tfidf_vectorizer_job_recommendation:
        text = cleanResume(text)
        vector = tfidf_vectorizer_job_recommendation.transform([text])
        return rf_classifier_job_recommendation.predict(vector)[0]
    return "Model not available"


# -------------------- Extract Info --------------------
def extract_contact_number_from_resume(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group() if match else "Not Found"


def extract_email_from_resume(text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group() if match else "Not Found"


# 🔥 FINAL NAME FIX
def extract_name_from_resume(text):
    lines = text.split('\n')

    # check top lines
    for line in lines[:5]:
        line = line.strip()

        if len(line) < 3:
            continue

        if any(word in line.lower() for word in ['resume', 'email', 'phone', 'contact']):
            continue

        # Match normal name
        match = re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+$', line)
        if match:
            return match.group()

        # Match ALL CAPS name
        match = re.match(r'^[A-Z\s]+$', line)
        if match:
            return line.title()

    return "Not Found"


def extract_skills_from_resume(text):
    skills_list = ['Python', 'Machine Learning', 'Data Analysis', 'SQL', 'Java', 'C++']
    found = [skill for skill in skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    return found if found else ["Not Found"]


def extract_education_from_resume(text):
    edu_keywords = ['Computer Science', 'Information Technology', 'Mechanical Engineering']
    found = [edu for edu in edu_keywords if re.search(rf"\b{re.escape(edu)}\b", text, re.IGNORECASE)]
    return found if found else ["Not Found"]


# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template('resume.html', message="No file uploaded")

    file = request.files['resume']

    # file type check
    if not file.filename.endswith(('.pdf', '.txt')):
        return render_template('resume.html', message="Only PDF or TXT allowed")

    # extract text
    if file.filename.endswith('.pdf'):
        text = pdf_to_text(file)
    else:
        text = file.read().decode('utf-8')

    # 🔥 validation
    if not is_valid_resume(text):
        return render_template('resume.html', message="Invalid resume! Please upload a proper resume.")

    # prediction
    predicted_category = predict_category(text)
    recommended_job = job_recommendation(text)

    # extraction
    phone = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    name = extract_name_from_resume(text)
    skills = extract_skills_from_resume(text)
    education = extract_education_from_resume(text)

    return render_template(
        'resume.html',
        predicted_category=predicted_category,
        recommended_job=recommended_job,
        phone=phone,
        email=email,
        name=name,
        extracted_skills=skills,
        extracted_education=education
    )


# -------------------- Run --------------------
if __name__ == '__main__':
    app.run(debug=True)