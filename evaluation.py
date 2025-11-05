import numpy as np
from typing import List, Dict
from groq import Groq
import re

class RAGEvaluator:
    """
    Evaluates RAG system performance using key metrics:
    - Faithfulness: Factual consistency with context
    - Answer Relevancy: Alignment with the question
    - Context Precision: Quality of retrieved documents
    """
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Measure factual consistency of answer with retrieved context.
        
        Args:
            answer (str): Generated answer
            contexts (list): Retrieved context documents
            
        Returns:
            float: Faithfulness score (0-1)
        """
        try:
            if not contexts or not answer:
                return 0.5
            
            combined_context = "\n".join(contexts[:3])  # Use first 3 contexts
            
            prompt = f"""Evaluate if the following answer is factually consistent with the provided context.
            
Context:
{combined_context[:2000]}

Answer:
{answer[:1000]}

Rate the faithfulness on a scale of 0 to 1, where:
- 1.0 = All claims in the answer are supported by the context
- 0.5 = Some claims are supported, others are not
- 0.0 = No claims are supported by the context

Respond with ONLY a number between 0 and 1 (e.g., 0.85)."""

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating factual consistency. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', score_text)
            score = float(numbers[0]) if numbers else 0.5
            
            return np.clip(score, 0.0, 1.0)
        
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return 0.5
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Measure how well the answer addresses the question.
        
        Args:
            question (str): Original question
            answer (str): Generated answer
            
        Returns:
            float: Answer relevancy score (0-1)
        """
        try:
            if not question or not answer:
                return 0.5
            
            prompt = f"""Evaluate how well the answer addresses the question.

Question:
{question}

Answer:
{answer[:1000]}

Rate the relevancy on a scale of 0 to 1, where:
- 1.0 = Answer completely and directly addresses all aspects of the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer does not address the question at all

Respond with ONLY a number between 0 and 1 (e.g., 0.92)."""

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating answer relevancy. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', score_text)
            score = float(numbers[0]) if numbers else 0.5
            
            return np.clip(score, 0.0, 1.0)
        
        except Exception as e:
            print(f"Error evaluating answer relevancy: {e}")
            return 0.5
    
    def evaluate_context_precision(self, question: str, contexts: List[str], relations: List[dict]) -> float:
        """
        Measure quality and relevance of retrieved documents.
        
        Args:
            question (str): Original question
            contexts (list): Retrieved context documents
            relations (list): Extracted knowledge triples
            
        Returns:
            float: Context precision score (0-1)
        """
        try:
            if not contexts:
                return 0.0
            
            # Simple heuristic: check if contexts contain keywords from question
            question_lower = question.lower()
            question_words = [word for word in question_lower.split() if len(word) > 3]
            
            relevant_contexts = 0
            for context in contexts:
                context_lower = context.lower()
                matches = sum(1 for word in question_words if word in context_lower)
                if matches > 0:
                    relevant_contexts += 1
            
            base_precision = relevant_contexts / len(contexts) if contexts else 0
            
            # Boost score if knowledge triples were successfully extracted
            relation_boost = min(len(relations) / 20.0, 0.3)  # Max 0.3 boost
            
            final_score = min(base_precision + relation_boost, 1.0)
            return final_score
        
        except Exception as e:
            print(f"Error evaluating context precision: {e}")
            return 0.5
    
    def evaluate(self, question: str, answer: str, contexts: List[str], relations: List[dict]) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation.
        
        Args:
            question (str): Original question
            answer (str): Generated answer
            contexts (list): Retrieved contexts
            relations (list): Extracted knowledge triples
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'faithfulness': self.evaluate_faithfulness(answer, contexts),
            'answer_relevancy': self.evaluate_answer_relevancy(question, answer),
            'context_precision': self.evaluate_context_precision(question, contexts, relations)
        }
        
        # Calculate overall score
        metrics['overall_score'] = np.mean(list(metrics.values()))
        
        return metrics
