from flask import Flask, request, render_template, jsonify, send_file
import os
import logging
import hashlib
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
from pymongo import MongoClient
from bson.objectid import ObjectId
import google.generativeai as genai
import json
from dotenv import load_dotenv
import time
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_LOGGING_VERBOSITY"] = "ERROR"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
if not os.getenv("MONGO_URI") or not os.getenv("GEMINI_API_KEY"):
    logger.error("Missing MONGO_URI or GEMINI_API_KEY")
    raise ValueError("Missing required environment variables")

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
MAX_UPLOADS = 5
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy initialization for heavy dependencies
client = None
model = None
embeddings = None
vector_store = None
FAISS_AVAILABLE = False
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    FAISS_AVAILABLE = True
    logger.debug("FAISS and embeddings modules available")
except ImportError:
    logger.warning("FAISS not available. Chat functionality will be disabled.")

def init_mongo():
    global client
    if client is None:
        start_time = time.time()
        client = MongoClient(os.getenv("MONGO_URI"))
        client.admin.command('ping')  # Test connection
        logger.debug(f"MongoDB client initialized in {time.time() - start_time:.2f} seconds")
    return client

def init_gemini():
    global model
    if model is None:
        start_time = time.time()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.debug(f"Gemini model initialized in {time.time() - start_time:.2f} seconds")
    return model

def init_embeddings():
    global embeddings, vector_store
    if FAISS_AVAILABLE and embeddings is None:
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.debug(f"Embeddings initialized in {time.time() - start_time:.2f} seconds")
    return embeddings

# MongoDB setup
db = None
resumes_collection = None
def get_db():
    global db, resumes_collection
    if db is None:
        client = init_mongo()
        db = client["resume_analyzer"]
        resumes_collection = db["resumes"]
        resumes_collection.create_index([("job_role", 1), ("role_fit_score", -1)])
        logger.debug("Created MongoDB index on job_role and role_fit_score")
    return resumes_collection

# Role-specific tech stacks
ROLE_TECH_STACKS = {
    "Data Scientist": "Python, R, SQL, Machine Learning, Statistics, Data Visualization, Pandas, Scikit-learn, TensorFlow, PyTorch, Spark, Databricks, AWS, MongoDB",
    "Data Analyst": "SQL, Excel, Python, Power BI, Tableau, Looker, Data Cleaning, Pandas, NumPy, Visualization, Reporting, Dashboards",
    "Business/Data Analytics Engineer": "Python, SQL, Excel, Power BI, Tableau, ETL, Looker, Data Pipelines, BigQuery, Snowflake, Statistics, Python Dash/Streamlit",
    "Generative AI Engineer": "Python, Machine Learning, Deep Learning, GANs, VAEs, Transformers, Diffusion Models, PyTorch, TensorFlow, Hugging Face, LangChain, Cloud (AWS/GCP/Azure), MLOps",
    "AI Engineer / ML Engineer": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, Keras, MLOps, Cloud Platforms, Model Deployment, APIs",
    "Full-Stack Developer": "HTML, CSS, JavaScript, React, Angular, Node.js, Express.js, MongoDB/PostgreSQL, REST APIs, Git, CI/CD, Cloud Hosting (AWS, Azure, GCP)",
    "Front-End Developer": "HTML, CSS, JavaScript, React, Vue.js, Angular, TailwindCSS, Bootstrap, Responsive Design, UI/UX, Webpack, Vite",
    "Back-End Developer": "Node.js, Python (Flask/Django/FastAPI), Express.js, REST APIs, GraphQL, SQL/NoSQL, MongoDB, PostgreSQL, CI/CD, Cloud Deployment",
    "Full-Stack AI Developer": "Python, FastAPI, Flask, Django, React, Node.js, LangChain, Vector Databases (Pinecone, Weaviate, FAISS), OpenAI API, RAG Pipelines, Cloud Functions",
    "AI Research Scientist": "Python, Deep Learning, Reinforcement Learning, Transformers, PyTorch, JAX, TensorFlow, Algorithm Design, HPC, Research Papers",
    "MLOps Engineer": "Python, Docker, Kubernetes, CI/CD, MLflow, Airflow, Kubeflow, Terraform, AWS Sagemaker, GCP Vertex AI, Azure ML, Model Deployment, Monitoring",
    "Computer Vision Engineer": "Python, OpenCV, TensorFlow, PyTorch, YOLO, Detectron2, GANs, Image Segmentation, Video Processing, CUDA, ONNX, Edge AI",
    "Speech AI Engineer": "Python, ASR, TTS, Whisper, Kaldi, DeepSpeech, Hugging Face Audio Models, Audio Preprocessing, Signal Processing, TorchAudio",
    "Reinforcement Learning Engineer": "Python, RLlib, Gymnasium, Stable Baselines3, DQN, PPO, Policy Gradients, PyTorch, JAX, Simulators",
    "Quantum ML Engineer": "Python, Qiskit, PennyLane, Cirq, TensorFlow Quantum, Variational Quantum Circuits, Quantum Annealing, Linear Algebra, Quantum Computing Hardware",
    "BI Engineer / Analyst": "SQL, Excel, Tableau, Power BI, Python, Pandas, Statistics, ETL, Looker, Data Cleaning, Visualization Best Practices",
    "AI Security Engineer": "Python, Cryptography, Adversarial ML, Secure Federated Learning, Blockchain, IAM, Zero-Trust, Cloud Security, Cybersecurity + AI tools"
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_file_hash(file):
    """Compute SHA-256 hash of file content for deduplication."""
    hasher = hashlib.sha256()
    file.seek(0)
    while chunk := file.read(8192):
        hasher.update(chunk)
    file.seek(0)
    return hasher.hexdigest()

def normalize_keys(data):
    """Normalize dictionary keys to lowercase."""
    if isinstance(data, dict):
        return {k.lower(): normalize_keys(v) if isinstance(v, (dict, list)) else v for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_keys(item) for item in data]
    return data

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            logger.debug(f"Extracted text from PDF: {file_path}")
            return text
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {str(e)}")
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        logger.debug(f"Extracted text from DOCX: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
        return f"Error extracting DOCX: {str(e)}"

def parse_resume_with_gemini(text, job_role):
    tech_stack = ROLE_TECH_STACKS.get(job_role, "Python, Machine Learning, Data Analysis")
    prompt = f"""
    Analyze the resume text for the job role: {job_role}.
    Expected skills: {tech_stack}.
    Extract structured data with lowercase keys:
    - name (string)
    - skills (dict with skill names as keys and proficiency scores 0-100 as values)
    - projects (list of dicts with title, relevance_score (0-100), summary)
    - experience_years (float)
    - education (string)
    - role_fit_score (0-100, based on overall match to role)
    Return JSON with lowercase keys only. If a field cannot be extracted, use reasonable defaults (e.g., empty dict/list, 0, 'N/A').
    Resume text: {text}
    """
    for attempt in range(2):  # Retry once if parsing fails
        try:
            response = model.generate_content(prompt)
            if not response.text:
                logger.error("Empty response from Gemini")
                return {"error": "Empty response from Gemini"}
            response_text = response.text.strip("```json\n").strip("```").strip()
            parsed = json.loads(response_text)
            parsed = normalize_keys(parsed)  # Ensure lowercase keys
            if not parsed.get("name") or not parsed.get("skills"):
                logger.warning(f"Partial parse for role {job_role}: {parsed}")
                return {"error": "Incomplete data: missing name or skills"}
            logger.debug(f"Parsed resume for role {job_role}: {parsed}")
            return parsed
        except json.JSONDecodeError as json_err:
            logger.error(f"Invalid JSON from Gemini: {response_text}, error: {str(json_err)}")
            return {"error": f"Invalid JSON response: {str(json_err)}"}
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            if attempt == 1:
                return {"error": f"Failed to parse resume: {str(e)}"}
    return {"error": "Failed to parse resume after retries"}

def convert_objectid_to_str(data):
    if isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    if isinstance(data, dict):
        return {k: convert_objectid_to_str(v) if k != '_id' else str(v) for k, v in data.items()}
    return data

@app.route("/")
def index():
    start_time = time.time()
    logger.debug("Serving index.html")
    response = render_template("index.html")
    logger.debug(f"Index route took {time.time() - start_time:.2f} seconds")
    return response

@app.route("/health")
def health():
    logger.debug("Health check accessed")
    return jsonify({"status": "ok"}), 200

@app.route("/upload", methods=["POST"])
def upload_resumes():
    start_time = time.time()
    init_gemini()  # Initialize Gemini only when needed
    init_embeddings()  # Initialize embeddings only when needed
    resumes_collection = get_db()
    
    if "resumes" not in request.files or "job_role" not in request.form:
        logger.error("Missing resumes or job role")
        return jsonify({"error": "Missing resumes or job role"}), 400
    
    job_role = request.form["job_role"]
    files = request.files.getlist("resumes")
    
    if len(files) > MAX_UPLOADS:
        logger.error(f"Too many files: {len(files)} exceeds {MAX_UPLOADS}")
        return jsonify({"error": f"Cannot upload more than {MAX_UPLOADS} resumes"}), 400
    
    results = []
    documents = []

    for file in files:
        if not file or not allowed_file(file.filename):
            results.append({"filename": file.filename, "error": "Invalid file type"})
            logger.error(f"Invalid file type: {file.filename}")
            continue
        
        content_length = getattr(file, 'content_length', None)
        if content_length and content_length > MAX_FILE_SIZE:
            results.append({"filename": file.filename, "error": "File size exceeds 5MB"})
            logger.error(f"File size exceeds limit: {file.filename} ({content_length} bytes)")
            continue

        filename = secure_filename(file.filename)
        file_hash = compute_file_hash(file)
        if resumes_collection.find_one({"file_hash": file_hash, "job_role": job_role}):
            results.append({"filename": filename, "error": "Resume already uploaded"})
            logger.warning(f"Duplicate resume: {filename} (hash: {file_hash})")
            continue

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file.save(file_path)
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                results.append({"filename": filename, "error": "File size exceeds 5MB"})
                logger.error(f"File size exceeds limit after save: {filename}")
                continue

            text = extract_text_from_pdf(file_path) if filename.endswith(".pdf") else extract_text_from_docx(file_path)
            if text.startswith("Error"):
                results.append({"filename": filename, "error": text})
                logger.error(f"Text extraction failed for {filename}: {text}")
                continue
            
            parsed_data = parse_resume_with_gemini(text, job_role)
            if "error" in parsed_data:
                results.append({"filename": filename, "error": parsed_data["error"]})
                continue
            
            parsed_data["filename"] = filename
            parsed_data["job_role"] = job_role
            parsed_data["file_hash"] = file_hash
            result = resumes_collection.insert_one(parsed_data)
            parsed_data["_id"] = result.inserted_id
            results.append(parsed_data)
            logger.debug(f"Inserted resume: {filename}, ID: {result.inserted_id}")
            if FAISS_AVAILABLE:
                documents.append(Document(page_content=text, metadata={"filename": filename, "_id": str(result.inserted_id)}))
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
    
    if FAISS_AVAILABLE and documents:
        try:
            global vector_store
            if vector_store is None:
                vector_store = FAISS.from_documents(documents, embeddings)
                logger.debug("Initialized FAISS vector store")
            else:
                vector_store.add_documents(documents)
                logger.debug(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"FAISS error: {str(e)}")
            results.append({"warning": f"Failed to update vector store: {str(e)}"})
    
    results = convert_objectid_to_str(results)
    logger.debug(f"Upload route took {time.time() - start_time:.2f} seconds")
    return jsonify({"results": results})

@app.route("/candidates", methods=["GET"])
def get_candidates():
    start_time = time.time()
    resumes_collection = get_db()
    job_role = request.args.get("job_role", "GenAI Engineer")
    limit = int(request.args.get("limit", 50))  # Limit to 50 candidates
    try:
        candidates = list(resumes_collection.find({"job_role": job_role, "role_fit_score": {"$gt": 0}}).limit(limit))
        logger.debug(f"Fetched {len(candidates)} candidates for {job_role} in {time.time() - start_time:.2f} seconds")
        seen_hashes = set()
        unique_candidates = []
        for c in candidates:
            file_hash = c.get("file_hash")
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                c = normalize_keys(dict(c))
                unique_candidates.append(c)
        for c in unique_candidates:
            c["_id"] = str(c["_id"])
        unique_candidates.sort(key=lambda x: x.get("role_fit_score", 0), reverse=True)
        logger.debug(f"Sorted candidates: {[(c.get('name', 'Unknown'), c.get('role_fit_score', 0)) for c in unique_candidates]}")
        return jsonify(unique_candidates)
    except Exception as e:
        logger.error(f"Error fetching candidates: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to fetch candidates: {str(e)}"}), 500

@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    start_time = time.time()
    resumes_collection = get_db()
    job_role = request.args.get("job_role", "GenAI Engineer")
    candidates = list(resumes_collection.find({"job_role": job_role, "role_fit_score": {"$gt": 0}}, {"_id": 0}))
    candidates = [normalize_keys(dict(c)) for c in candidates]
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for candidate in candidates:
        story.append(Paragraph(f"Name: {candidate.get('name', 'Unknown')}", styles["Heading2"]))
        story.append(Paragraph(f"Role Fit Score: {candidate.get('role_fit_score', 0)}", styles["Normal"]))
        skills = ", ".join([f"{k}: {v}" for k, v in candidate.get("skills", {}).items()])
        story.append(Paragraph(f"Skills: {skills}", styles["Normal"]))
        story.append(Paragraph(f"Experience: {candidate.get('experience_years', 0)} years", styles["Normal"]))
        story.append(Paragraph(f"Education: {candidate.get('education', 'N/A')}", styles["Normal"]))
        projects = ", ".join([p.get("title", "") for p in candidate.get("projects", [])])
        story.append(Paragraph(f"Projects: {projects}", styles["Normal"]))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    logger.debug(f"Generated PDF for {job_role} in {time.time() - start_time:.2f} seconds")
    return send_file(buffer, as_attachment=True, download_name=f"{job_role}_candidates.pdf", mimetype="application/pdf")

@app.route("/export/csv", methods=["GET"])
def export_csv():
    start_time = time.time()
    resumes_collection = get_db()
    job_role = request.args.get("job_role", "GenAI Engineer")
    candidates = list(resumes_collection.find({"job_role": job_role, "role_fit_score": {"$gt": 0}}, {"_id": 0}))
    candidates = [normalize_keys(dict(c)) for c in candidates]
    
    data = [{
        "Name": c.get("name", "Unknown"),
        "Role Fit Score": c.get("role_fit_score", 0),
        "Skills": ", ".join([f"{k}: {v}" for k, v in c.get("skills", {}).items()]),
        "Experience (Years)": c.get("experience_years", 0),
        "Education": c.get("education", "N/A"),
        "Projects": ", ".join([p.get("title", "") for p in c.get("projects", [])])
    } for c in candidates]
    
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    logger.debug(f"Generated CSV for {job_role} in {time.time() - start_time:.2f} seconds")
    return send_file(buffer, as_attachment=True, download_name=f"{job_role}_candidates.csv", mimetype="text/csv")

@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()
    init_embeddings()
    resumes_collection = get_db()
    query = request.json.get("query")
    if not query:
        logger.error("Missing query in chat request")
        return jsonify({"error": "Missing query"}), 400
    
    if not FAISS_AVAILABLE or vector_store is None:
        logger.warning("Chat attempted but FAISS/vector store unavailable")
        return jsonify({"error": "No resumes indexed or FAISS unavailable"}), 400
    
    try:
        results = vector_store.similarity_search(query, k=3)
        candidate_data = []
        for doc in results:
            _id = doc.metadata.get("_id")
            if _id:
                candidate = resumes_collection.find_one({"_id": ObjectId(_id)})
                if candidate:
                    candidate = normalize_keys(dict(candidate))
                    candidate["_id"] = str(candidate["_id"])
                    candidate_data.append(candidate)
        
        if not candidate_data:
            logger.warning(f"No relevant candidates found for query: {query}")
            return jsonify({"error": "No relevant candidates found"}), 404
        
        data_str = json.dumps(candidate_data)
        init_gemini()
        prompt = f"""
        Based on candidate data: {data_str}
        Answer the query: {query}
        Be concise and factual. Use role_fit_score, skills, and experience for comparisons.
        """
        response = model.generate_content(prompt)
        logger.debug(f"Chat response for '{query}': {response.text} in {time.time() - start_time:.2f} seconds")
        return jsonify({"answer": response.text})
    except Exception as e:
        logger.error(f"Chat query failed: {str(e)}")
        return jsonify({"error": f"Chat query failed: {str(e)}"}), 500

@app.route("/clear", methods=["POST"])
def clear_database():
    start_time = time.time()
    resumes_collection = get_db()
    try:
        result = resumes_collection.delete_many({})
        global vector_store
        vector_store = None  # Reset vector store
        logger.debug(f"Cleared {result.deleted_count} records from database in {time.time() - start_time:.2f} seconds")
        return jsonify({"message": f"Cleared {result.deleted_count} records"}), 200
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({"error": f"Failed to clear database: {str(e)}"}), 500

if __name__ == "__main__":
    start_time = time.time()
    port = int(os.environ.get("PORT", 8080))  # Changed default to 8080 for Render compatibility
    app.run(host="0.0.0.0", port=port)
    logger.debug(f"Application fully started in {time.time() - start_time:.2f} seconds")