import streamlit as st
import json
from typing import List
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="Medical RAG Chatbot - Kelly",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL FIX: Patch httpx Client BEFORE importing groq
import httpx
_OriginalClient = httpx.Client
class _PatchedClient(_OriginalClient):
    def __init__(self, *args, **kwargs):
        kwargs.pop('proxies', None)
        kwargs.pop('proxy', None)
        kwargs.pop('mounts', None)
        super().__init__(*args, **kwargs)
httpx.Client = _PatchedClient

# NOW import groq and other modules
from groq import Groq
from pubmed_scraper import get_pubmed_term_entries, get_abstracts_text
from llama_processor import Llama3ExtractTopicFromText, Llama3ExtractRelationsFromText
from evaluation import RAGEvaluator

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

# Initialize clients
@st.cache_resource
def initialize_clients():
    """Initialize Groq client and processors with API key from secrets"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        
        client = Groq(api_key=api_key)
        topic_extractor = Llama3ExtractTopicFromText(api_key)
        relations_extractor = Llama3ExtractRelationsFromText(api_key)
        evaluator = RAGEvaluator(api_key)
        
        return client, topic_extractor, relations_extractor, evaluator
        
    except KeyError:
        st.error("❌ GROQ_API_KEY not found in secrets.")
        st.info("Add it in: App Menu → Settings → Secrets")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        st.stop()

client, topic_extractor, relations_extractor, evaluator = initialize_clients()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    num_articles = st.slider(
        "Number of PubMed articles per topic",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values provide more context"
    )
    
    enable_evaluation = st.checkbox(
        "Enable RAG Evaluation",
        value=True,
        help="Calculate evaluation metrics"
    )
    
    st.markdown("---")
    st.markdown("## 📊 Session Statistics")
    st.metric("Total Queries", len(st.session_state.chat_history))
    
    if st.session_state.evaluation_results:
        avg_score = sum(r['overall_score'] for r in st.session_state.evaluation_results) / len(st.session_state.evaluation_results)
        st.metric("Avg Quality Score", f"{avg_score:.3f}")
    
    st.markdown("---")
    st.markdown("## 🔬 Sample Questions")
    st.markdown("""
    - Latest treatments for hypertension?
    - Non-pharmacological interventions for diabetes?
    - Cardiovascular disease prevention?
    - Hypertension in elderly patients?
    """)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.evaluation_results = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🏥 Medical RAG Chatbot - Kelly</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-Based Medical Information Powered by PubMed & Llama 3.3</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat", "📈 Evaluation Metrics"])

with tab1:
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🧑‍⚕️ Question {i+1}:** {chat['question']}")
            st.markdown(f"**🤖 Kelly's Answer:**")
            st.info(chat['answer'])
            
            if chat.get('topics'):
                with st.expander(f"📚 Topics ({len(chat['topics'])} topics)"):
                    st.write(", ".join(chat['topics']))
            
            if chat.get('evaluation'):
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", f"{chat['evaluation']['faithfulness']:.3f}")
                col2.metric("Relevancy", f"{chat['evaluation']['answer_relevancy']:.3f}")
                col3.metric("Precision", f"{chat['evaluation']['context_precision']:.3f}")
            
            st.markdown("---")
    
    # Input form
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_area(
            "🔍 Ask a medical question:",
            placeholder="e.g., What are the latest treatments for hypertension?",
            height=100
        )
        submit_button = st.form_submit_button("🚀 Search & Generate Answer", use_container_width=True)
    
    if submit_button and user_query:
        with st.spinner("🔬 Processing..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract topics
                status_text.text("🔍 Extracting topics...")
                progress_bar.progress(20)
                topics_json = topic_extractor.prompt_llama3(user_query)
                topics = json.loads(topics_json).get('topics', [])
                
                if not topics:
                    topics = [user_query]
                
                # Retrieve PubMed articles
                status_text.text(f"📚 Searching PubMed...")
                progress_bar.progress(40)
                all_abstracts = []
                
                for topic in topics:
                    records = get_pubmed_term_entries(topic, num_articles)
                    ids = records['IdList']
                    if ids:
                        abstracts = get_abstracts_text(ids)
                        all_abstracts.extend(abstracts)
                
                progress_bar.progress(60)
                status_text.text("🤖 Generating answer...")
                
                # Generate response
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Kelly, a medical expert. "
                                "Provide evidence-based information using medical literature. "
                                "Be clear, accurate, and actionable."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Question: {user_query}\n\nContext from literature:\n{' '.join(all_abstracts[:3])}\n\nProvide a comprehensive answer."
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                answer = response.choices[0].message.content
                progress_bar.progress(80)
                
                # Evaluate
                eval_metrics = None
                if enable_evaluation and all_abstracts:
                    status_text.text("📊 Evaluating response...")
                    eval_metrics = evaluator.evaluate(
                        question=user_query,
                        answer=answer,
                        contexts=all_abstracts[:3],
                        relations=[]
                    )
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.chat_history.append({
                    'question': user_query,
                    'answer': answer,
                    'topics': topics,
                    'evaluation': eval_metrics
                })
                st.session_state.evaluation_results.append(eval_metrics) if eval_metrics else None
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.markdown("## 📊 Evaluation Metrics")
    
    if st.session_state.evaluation_results:
        latest_eval = st.session_state.evaluation_results[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faithfulness", f"{latest_eval['faithfulness']:.3f}")
        col2.metric("Relevancy", f"{latest_eval['answer_relevancy']:.3f}")
        col3.metric("Precision", f"{latest_eval['context_precision']:.3f}")
        col4.metric("Overall", f"{latest_eval['overall_score']:.3f}")
        
        if len(st.session_state.evaluation_results) > 1:
            st.markdown("### Historical Performance")
            df = pd.DataFrame(st.session_state.evaluation_results)
            df['Query'] = range(1, len(df) + 1)
            chart_data = df[['Query', 'faithfulness', 'answer_relevancy', 'context_precision']].set_index('Query')
            st.line_chart(chart_data)
    else:
        st.info("No evaluation data yet.")

st.markdown("---")
st.caption("⚠️ Medical Disclaimer: Not a substitute for professional medical advice.")
