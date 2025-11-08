# 🩺 Medical RAG: Evidence-Based Medical Chatbot

This project is a **Medical RAG (Retrieval-Augmented Generation)** system designed to answer **medical questions** with **evidence-based responses**.  
It integrates a **Large Language Model (Llama 3 via Groq)** with **real-time PubMed retrieval**, ensuring every answer is grounded in the latest scientific literature.

---

##  Features

###  RAG Pipeline
Implements a complete **Retrieval-Augmented Generation (RAG)** workflow to ensure that LLM-generated answers are **factually supported** by external data.

###  Real-Time Medical Data
Instead of relying on a static vector store, this project retrieves **live PubMed abstracts** using **Biopython**, providing **up-to-date medical insights**.

###  LLM Processing
Powered by **Llama 3 (via Groq)**, the model performs two main tasks:
1. **Topic Extraction** – Identifies key medical terms and topics from user queries.  
2. **Answer Synthesis** – Generates coherent, evidence-based answers using retrieved PubMed abstracts.

###  Built-in Evaluation
Includes a **RAG Evaluation Dashboard** (in Streamlit) using the **RAGAS framework**, evaluating the pipeline on:
- *Faithfulness*
- *Answer Relevancy*
- *Context Precision*
- *Context Recall*

###  Interactive UI
A clean, minimal **Streamlit chat interface** allows users to ask questions and view synthesized, cited answers.

---

## ⚙️ How It Works: The RAG Pipeline

1. **User Input**  
   The user submits a medical question in the Streamlit chat interface.  
   *Example:* “What are the new treatments for migraine?”

2. **Topic Extraction**  
   The question is sent to the **Llama 3 model** (via `llama_processor.py`) to extract relevant search terms.

3. **Data Retrieval (R)**  
   Extracted topics are used to query **PubMed** in real-time using **Biopython (Entrez)**, fetching relevant abstracts.

4. **Answer Generation (G)**  
   The retrieved abstracts, along with the user’s question, are passed back to the LLM.

5. **Synthesized Answer**  
   The model generates a **comprehensive, evidence-grounded response**, citing PubMed sources.  
   The answer is displayed in the Streamlit interface.

---

##  Technologies Used

| Component        | Technology / Library |
|------------------|----------------------|
| **Application**  | Streamlit |
| **LLM**          | Llama 3 via Groq (`langchain-groq`) |
| **Orchestration**| LangChain |
| **Data Retrieval**| Biopython (PubMed API) |
| **Evaluation**   | RAGAS, Pandas |
| **Core Language**| Python |

> 📦 See `requirements.txt` for the full list of dependencies.


