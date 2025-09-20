# AI Resume Analyzer

## Overview
The AI Resume Analyzer is a Flask web application designed to help users upload resumes, extract relevant information, and analyze candidates based on their qualifications. The application utilizes various libraries for file handling, data extraction, and machine learning to provide insights into resumes.

## Features
- Upload multiple resume files in PDF or DOCX format.
- Extract structured data from resumes, including skills, experience, and education.
- Store and manage resumes in a MongoDB database.
- Export candidate information in PDF and CSV formats.
- Chat functionality to query candidate data based on user input.

## Setup Instructions

### Prerequisites
- Python 3.x
- MongoDB (local or cloud instance)
- API key for Google Generative AI

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-resume-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your environment variables:
   ```
   MONGO_URI=<your_mongo_uri>
   GEMINI_API_KEY=<your_api_key>
   ```

### Running the Application
1. Start the Flask application:
   ```
   python main.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage
- Use the upload feature to submit resumes for analysis.
- Select a job role to tailor the analysis based on specific qualifications.
- View and export candidate data in various formats.
- Utilize the chat feature to ask questions about the candidates based on their resumes.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.