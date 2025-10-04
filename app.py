import streamlit as st
import os
from medical_rag.mock_pipeline import main_generate_answer_batched_mock, main_evaluation_high_mock_fixed

st.set_page_config(page_title="Medical RAG Pipeline", layout="wide")

st.title("🩺 Medical RAG Pipeline: Hypertension Management")

st.markdown("""
This application retrieves, extracts, and generates expert-style recommendations for **non-pharmacological interventions in hypertension**, focusing on elderly patients with cardiovascular comorbidities.
""")

query = st.text_area(
    "Enter your medical query:",
    "What are the latest recommended non-pharmacological interventions for managing hypertension in elderly patients with cardiovascular comorbidities?"
)

if st.button("Generate Answer"):
    with st.spinner("Generating expert answer..."):
        ids, relations, answer = main_generate_answer_batched_mock()
        
    st.subheader("✅ Expert Answer")
    st.markdown(answer)

    st.subheader("📊 Evaluation Metrics")
    main_evaluation_high_mock_fixed(ids, relations)

    st.subheader("🔗 Extracted Relations (Sample)")
    for r in relations:
        st.write(f"{r['head']} → {r['predicate']} → {r['tail']}")
