##Medical RAG: Evidence-Based Medical Chatbot

This project is a "Medical RAG" (Retrieval-Augmented Generation) system designed to answer medical questions by providing evidence-based responses. It uses a Large Language Model (Llama 3 via Groq) to understand user queries and generate answers, with a retrieval step that dynamically fetches relevant medical abstracts from PubMed.

The application is built with Streamlit and features a chat interface, a real-time retrieval pipeline, and a built-in RAG evaluation dashboard using the RAGAS framework.

🚀 Features
RAG Pipeline: Implements a full Retrieval-Augmented Generation pipeline to ground LLM answers in factual, external data.

Real-Time Medical Data: Instead of a static vector store, this project retrieves live medical abstracts from PubMed using Biopython, ensuring up-to-date information.

LLM Processing: Uses the high-speed Llama 3 model (via Groq) for two key tasks:

Extracting relevant medical topics from the user's question.

Synthesizing a final, coherent answer from the retrieved PubMed abstracts.

Built-in Evaluation: Includes a "RAG Evaluation" tab in the Streamlit app that uses the RAGAS library to score the pipeline's performance on metrics like faithfulness, answer_relevancy, context_precision, and context_recall.

Interactive UI: A simple and clean chat interface built with Streamlit.

⚙️ How It Works: The RAG Pipeline
User Input: The user asks a medical question in the Streamlit chat interface (e.g., "What are the new treatments for migraine?").

Topic Extraction: The question is sent to the Llama 3 model (via llama_processor.py) to extract key search topics.

Data Retrieval (R): The extracted topics are used to query the PubMed database in real-time using Biopython (Entrez). The system fetches relevant medical article abstracts.

Answer Generation (G): The original question and the retrieved abstracts (as context) are passed back to the Llama 3 model.

Synthesized Answer: The model generates a comprehensive answer based only on the provided context, citing its sources. The application then displays this evidence-based answer to the user.

🛠️ Technologies Used
Application: Streamlit

LLM: Llama 3 via Groq (using langchain-groq)

Orchestration: LangChain

Data Retrieval: Biopython (for PubMed API)

Evaluation: RAGAS, Pandas

Core: Python

(See requirements.txt for all dependencies)

🏁 Getting Started
1. Prerequisites
Python 3.9+

A Groq API Key

An Entrez Email (for NCBI/PubMed API, as required by Biopython)

2. Installation
Clone the repository:

Bash

git clone https://github.com/your-username/medical-rag.git
cd medical-rag
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

Bash

pip install -r requirements.txt
3. Environment Variables
Create a .env file in the root of the project directory and add your API keys:

GROQ_API_KEY="your-groq-api-key"
ENTREZ_EMAIL="your-email@example.com"
The application uses python-dotenv to load these keys.

4. Run the Application
Launch the Streamlit app:

Bash

streamlit run app.py
