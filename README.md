# Swiggy Annual Report RAG (Gemini Powered)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural language questions about the **Swiggy Annual Report FY 2023–24**.

The system extracts information from the PDF, creates embeddings for document chunks, stores them in a FAISS vector database, and uses **Google Gemini** to generate answers grounded in the report.

The application supports both:

* **Command Line Interface (CLI)**
* **Streamlit Web Interface**

---

## Architecture

The project follows a standard **RAG pipeline**:

1. **PDF Ingestion**

   * Extract text from the Swiggy annual report using `pypdf`.

2. **Text Chunking**

   * Split document into smaller overlapping chunks.

3. **Embedding Generation**

   * Generate vector embeddings using:

   ```
   sentence-transformers/all-MiniLM-L6-v2
   ```

4. **Vector Storage**

   * Store embeddings in **FAISS vector index**.

5. **Retrieval**

   * Retrieve the most relevant chunks for a given query.

6. **LLM Generation**

   * Send retrieved context to **Google Gemini** to generate grounded answers.

---

## Tech Stack

* Python
* FAISS (vector search)
* Sentence Transformers
* Google Gemini API
* PyPDF
* Streamlit
* NumPy
* Python-dotenv

---

## Project Structure

```
swiggy_rag_gemini/
│
├── main.py
├── app.py
├── requirements.txt
├── swiggy_annual_report.pdf
├── .env
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```
git clone <https://github.com/sanikad20/swiggy-rag-gemini>
cd swiggy_rag_gemini
```

---

### 2. Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Add Gemini API Key

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

Generate API key from **Google AI Studio**.

---

## Running the CLI Version

```
python main.py
```

Example:

```
Ask a question: What was total income in FY24?
```

Output:

```
ANSWER:
Total Income in FY24 was ₹116,343 Million.

SOURCES:
Page: 5
Page: 6
Page: 8
```


<img width="826" height="653" alt="image" src="https://github.com/user-attachments/assets/a3019f93-4bba-4ec2-9840-aab9ca9fae0c" />

---

## Running the Web App

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Example Questions

* What was total income in FY24?
* What was revenue from operations in FY24?
* What was Swiggy's net loss in FY24?
* What are the key business segments of Swiggy?
* What risks are mentioned in the annual report?

---

## Key Features

* Query financial documents using natural language
* Retrieval-Augmented Generation architecture
* Vector similarity search with FAISS
* Context-grounded answers from Gemini
* Source page references for transparency
* CLI and Web Interface support

---

## Future Improvements

* Support multiple documents
* Add document upload feature
* Improve chunking strategy
* Add citation highlighting
* Deploy as a web service

---

## License

This project is intended for educational and demonstration purposes.

