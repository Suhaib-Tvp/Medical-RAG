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

# Initialize clients with bulletproof error handling
@st.cache_resource
def initialize_clients():
    """Initialize Groq client and processors with API key from secrets"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        
        # Import inside function to avoid circular imports
        from groq import Groq as OriginalGroq
        from pubmed_scraper import get_pubmed_term_entries, get_abstracts_text
        from llama_processor import Llama3ExtractTopicFromText, Llama3ExtractRelationsFromText
        from evaluation import RAGEvaluator
        
        # Wrapper to handle proxies parameter issue
        class SafeGroqClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self._client = None
                self._initialize_client()
            
            def _initialize_client(self):
                """Initialize Groq client with error handling for proxies"""
                try:
                    self._client = OriginalGroq(api_key=self.api_key)
                except TypeError as e:
                    if 'proxies' in str(e):
                        # Patch the __init__ to remove proxies
                        import inspect
                        original_init = OriginalGroq.__init__
                        
                        def new_init(self, *args, **kwargs):
                            # Remove proxies from kwargs if present
                            kwargs.pop('proxies', None)
                            kwargs.pop('proxy', None)
                            original_init(self, *args, **kwargs)
                        
                        OriginalGroq.__init__ = new_init
                        self._client = OriginalGroq(api_key=self.api_key)
                    else:
                        raise
            
            def __getattr__(self, name):
                """Delegate all attributes to the wrapped client"""
                return getattr(self._client, name)
        
        client = SafeGroqClient(api_key)
        topic_extractor = Llama3ExtractTopicFromText(api_key)
        relations_extractor = Llama3ExtractRelationsFromText(api_key)
        evaluator = RAGEvaluator(api_key)
        
        return client, topic_extractor, relations_extractor, evaluator, get_pubmed_term_entries, get_abstracts_text
        
    except KeyError:
        st.error("❌ GROQ_API_KEY not found in secrets. Please add it in Streamlit Cloud settings.")
        st.info("Go to: App Menu → Settings → Secrets")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

client, topic_extractor, relations_extractor, evaluator, get_pubmed_term_entries, get_abstracts_text = initialize_clients()

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
            placeholder="e.g., What are the latest recommended non-pharmacological interventions for managing hypertension in elderly patients?",
            height=100
        )
        submit_button = st.form_submit_button("🚀 Search & Generate Answer", use_container_width=True)
    
    if submit_button and user_query:
        with st.spinner("🔬 Processing your query..."):
            try:
                # Progress tracking
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
                                "You are Kelly, a medical expert AI assistant providing evidence-based information. "
                                "You have access to structured knowledge from recent medical literature. "
                                "Use this as a factual foundation while supplementing with relevant medical knowledge. "
                                "Prioritize accuracy, provide clear explanations, and give actionable insights."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Question: {user_query}\n\nRelevant knowledge:\n{all_relations}\n\n"
                                       f"Provide a comprehensive, evidence-based answer."
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
                status_text.text("✅ Complete!")
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
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.markdown("## 📊 RAG Evaluation Metrics")
    
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

with tab3:
    st.markdown("## 🕸️ Medical Knowledge Graph")
    
    if st.session_state.knowledge_graph:
        df_kg = pd.DataFrame(st.session_state.knowledge_graph)
        st.dataframe(df_kg, use_container_width=True)
        
        csv = df_kg.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name="knowledge_graph.csv",
            mime="text/csv"
        )
    else:
        st.info("No knowledge graph data yet.")

st.markdown("---")
st.caption("⚠️ Medical Disclaimer: Not a substitute for professional medical advice.")
