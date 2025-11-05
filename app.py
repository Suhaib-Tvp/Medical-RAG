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

# Save original Client class
_OriginalClient = httpx.Client

# Create patched Client that filters out proxies argument
class _PatchedClient(_OriginalClient):
    def __init__(self, *args, **kwargs):
        # Remove problematic proxy-related arguments
        kwargs.pop('proxies', None)
        kwargs.pop('proxy', None)
        kwargs.pop('mounts', None)
        super().__init__(*args, **kwargs)

# Replace httpx.Client globally
httpx.Client = _PatchedClient

# NOW it's safe to import groq (after httpx is patched)
from groq import Groq

# Import other modules
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
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = []

# Initialize clients - NOW THIS WILL WORK
@st.cache_resource
def initialize_clients():
    """Initialize Groq client and processors with API key from secrets"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        
        # Create clients (httpx is already patched, so this works)
        client = Groq(api_key=api_key)
        topic_extractor = Llama3ExtractTopicFromText(api_key)
        relations_extractor = Llama3ExtractRelationsFromText(api_key)
        evaluator = RAGEvaluator(api_key)
        
        return client, topic_extractor, relations_extractor, evaluator
        
    except KeyError:
        st.error("❌ GROQ_API_KEY not found in secrets.")
        st.info("Go to: App Menu → Settings → Secrets and add your Groq API key")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Initialize all clients
client, topic_extractor, relations_extractor, evaluator = initialize_clients()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    num_articles = st.slider(
        "Number of PubMed articles per topic",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values provide more context but increase processing time"
    )
    
    enable_evaluation = st.checkbox(
        "Enable RAG Evaluation",
        value=True,
        help="Calculate faithfulness, relevancy, and precision metrics"
    )
    
    st.markdown("---")
    st.markdown("## 📊 Session Statistics")
    st.metric("Total Queries", len(st.session_state.chat_history))
    st.metric("Knowledge Triples", len(st.session_state.knowledge_graph))
    
    if st.session_state.evaluation_results:
        avg_score = sum(r['overall_score'] for r in st.session_state.evaluation_results) / len(st.session_state.evaluation_results)
        st.metric("Avg Quality Score", f"{avg_score:.3f}")
    
    st.markdown("---")
    st.markdown("## 🔬 Sample Medical Queries")
    st.markdown("""
    - Latest treatments for hypertension in elderly patients?
    - Non-pharmacological interventions for Type 2 Diabetes?
    - Recent advances in cardiovascular disease prevention?
    - Hypertension management in chronic kidney disease?
    - Side effects of ACE inhibitors?
    """)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.evaluation_results = []
        st.session_state.knowledge_graph = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🏥 Medical RAG Chatbot - Kelly</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-Based Medical Information Powered by PubMed & Llama 3.3</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Evaluation Metrics", "🕸️ Knowledge Graph"])

with tab1:
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🧑‍⚕️ Question {i+1}:** {chat['question']}")
            st.markdown(f"**🤖 Kelly's Answer:**")
            st.info(chat['answer'])
            
            if chat.get('topics'):
                with st.expander(f"📚 Topics Extracted ({len(chat['topics'])} topics)"):
                    st.write(", ".join(chat['topics']))
            
            if chat.get('relations_count'):
                st.caption(f"🔗 {chat['relations_count']} knowledge triples extracted from PubMed")
            
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
            placeholder="e.g., What are the latest recommended non-pharmacological interventions for managing hypertension?",
            height=100
        )
        submit_button = st.form_submit_button("🚀 Search & Generate Answer", use_container_width=True)
    
    if submit_button and user_query:
        with st.spinner("🔬 Processing your query..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract topics
                status_text.text("🔍 Step 1/5: Extracting medical topics...")
                progress_bar.progress(20)
                topics_json = topic_extractor.prompt_llama3(user_query)
                topics = json.loads(topics_json)['topics']
                
                if not topics:
                    topics = [user_query]
                
                # Step 2: Retrieve PubMed articles
                status_text.text(f"📚 Step 2/5: Searching PubMed for {len(topics)} topics...")
                progress_bar.progress(40)
                all_relations = []
                all_abstracts = []
                
                for topic in topics:
                    records = get_pubmed_term_entries(topic, num_articles)
                    ids = records['IdList']
                    if ids:
                        abstracts = get_abstracts_text(ids)
                        abstracts_text = "\n".join(abstracts)
                        all_abstracts.extend(abstracts)
                        
                        if abstracts_text.strip():
                            rels = relations_extractor.prompt_llama3(abstracts_text)
                            rels_data = json.loads(rels)
                            if 'triples' in rels_data:
                                all_relations.extend(rels_data['triples'])
                
                progress_bar.progress(60)
                status_text.text("🤖 Step 3/5: Generating evidence-based answer...")
                
                # Step 3: Generate response
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Kelly, a medical expert AI assistant. "
                                "Provide evidence-based information using structured knowledge from medical literature. "
                                "Be clear, accurate, and actionable in your responses."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Question: {user_query}\n\nMedical knowledge:\n{all_relations}\n\nProvide a comprehensive answer."
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                answer = response.choices[0].message.content
                progress_bar.progress(80)
                
                # Step 4: Evaluate
                eval_metrics = None
                if enable_evaluation and all_abstracts:
                    status_text.text("📊 Step 4/5: Evaluating response quality...")
                    eval_metrics = evaluator.evaluate(
                        question=user_query,
                        answer=answer,
                        contexts=all_abstracts[:5],
                        relations=all_relations
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ Step 5/5: Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.chat_history.append({
                    'question': user_query,
                    'answer': answer,
                    'topics': topics,
                    'relations': all_relations,
                    'relations_count': len(all_relations),
                    'evaluation': eval_metrics
                })
                st.session_state.knowledge_graph.extend(all_relations)
                if eval_metrics:
                    st.session_state.evaluation_results.append(eval_metrics)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.markdown("## 📊 RAG Evaluation Metrics")
    st.markdown("""
    Assessment of RAG system quality:
    - **Faithfulness** (0-1): Factual consistency with retrieved context
    - **Answer Relevancy** (0-1): Alignment with the question
    - **Context Precision** (0-1): Quality of retrieved documents
    """)
    
    if st.session_state.evaluation_results:
        latest_eval = st.session_state.evaluation_results[-1]
        
        st.markdown("### 🎯 Latest Query Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Faithfulness", f"{latest_eval['faithfulness']:.3f}")
            st.caption("Factual consistency")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Relevancy", f"{latest_eval['answer_relevancy']:.3f}")
            st.caption("Query alignment")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{latest_eval['context_precision']:.3f}")
            st.caption("Retrieval quality")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Overall Score", f"{latest_eval['overall_score']:.3f}")
            st.caption("Average metric")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if len(st.session_state.evaluation_results) > 1:
            st.markdown("### 📈 Historical Performance")
            df = pd.DataFrame(st.session_state.evaluation_results)
            df['Query'] = range(1, len(df) + 1)
            chart_data = df[['Query', 'faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']].set_index('Query')
            st.line_chart(chart_data)
            
            st.markdown("### 📋 Summary Statistics")
            summary_df = df[['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']].describe()
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("📝 No evaluation data yet. Ask a question to see metrics.")

with tab3:
    st.markdown("## 🕸️ Medical Knowledge Graph")
    st.markdown("Structured medical relationships extracted from PubMed literature")
    
    if st.session_state.knowledge_graph:
        st.markdown("### 📊 Knowledge Graph Statistics")
        col1, col2, col3 = st.columns(3)
        
        df_kg = pd.DataFrame(st.session_state.knowledge_graph)
        col1.metric("Total Triples", len(df_kg))
        col2.metric("Unique Subjects", df_kg['head'].nunique())
        col3.metric("Unique Relations", df_kg['predicate'].nunique())
        
        st.markdown("### 📋 Knowledge Triples")
        st.dataframe(
            df_kg,
            use_container_width=True,
            column_config={
                "head": st.column_config.TextColumn("Subject", width="medium"),
                "predicate": st.column_config.TextColumn("Relationship", width="medium"),
                "tail": st.column_config.TextColumn("Object", width="medium")
            }
        )
        
        st.markdown("### 🔗 Relationship Distribution")
        relation_counts = df_kg['predicate'].value_counts()
        st.bar_chart(relation_counts)
        
        csv = df_kg.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Knowledge Graph as CSV",
            data=csv,
            file_name="medical_knowledge_graph.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("📝 No knowledge graph yet. Ask a medical question to build it!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    <p>⚠️ <strong>Medical Disclaimer:</strong> This AI provides information from literature but is <strong>not a substitute for professional medical advice</strong>.</p>
    <p>🔬 Data: PubMed (NCBI) | 🤖 Model: Llama 3.3 70B (Groq) | 🎨 Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
