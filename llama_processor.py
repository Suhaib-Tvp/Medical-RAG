from groq import Groq
from pydantic import BaseModel, Field
from typing import List
import json

# Pydantic models
class Question(BaseModel):
    topics: List[str] = Field(
        default=[], 
        max_length=10, 
        description="Extracted medical topics from the question"
    )

class Triple(BaseModel):
    head: str = Field(..., description="Subject of the relation")
    predicate: str = Field(..., description="Relationship type")
    tail: str = Field(..., description="Object of the relation")

class Triples(BaseModel):
    triples: List[Triple] = Field(default=[], description="List of knowledge triples")

# Base processor class
class Llama3Processor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.user_input = None
        self.system_prompt = None
        self.response_model = None
    
    def prompt_llama3(self, text: str) -> str:
        """
        Send prompt to Llama 3.3 and get structured JSON response.
        
        Args:
            text (str): Input text to process
            
        Returns:
            str: JSON formatted response
        """
        user_input = f"{self.user_input}:\n{text}"
        
        # Get JSON schema from Pydantic model
        try:
            schema = self.response_model.model_json_schema()
        except:
            schema = self.response_model.schema()
            
        format_instructions = f"You must respond with valid JSON that matches this exact schema: {json.dumps(schema)}"
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}\n\n{format_instructions}"
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(content)
            except:
                # If not valid JSON, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'``````', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                    parsed_json = json.loads(content)
                else:
                    # Return empty structure
                    if self.response_model == Question:
                        return json.dumps({"topics": []})
                    elif self.response_model == Triples:
                        return json.dumps({"triples": []})
                    return "{}"
            
            # Validate with Pydantic
            try:
                parsed = self.response_model.model_validate(parsed_json)
            except:
                parsed = self.response_model(**parsed_json)
                
            try:
                return parsed.model_dump_json()
            except:
                return json.dumps(parsed.dict())
            
        except Exception as e:
            print(f"Error in prompt_llama3: {e}")
            # Return empty structure if error
            if self.response_model == Question:
                return json.dumps({"topics": []})
            elif self.response_model == Triples:
                return json.dumps({"triples": []})
            return "{}"

# Topic extraction processor
class Llama3ExtractTopicFromText(Llama3Processor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.system_prompt = (
            "You are a medical topic extraction expert. Extract key medical topics from the user's question. "
            "Focus on specific conditions, treatments, and relevant subtopics for literature search. "
            "Avoid redundancy and extract only meaningful, medically significant topics. "
            "Return a JSON object with a 'topics' array containing the extracted topics."
        )
        self.user_input = (
            "Extract key medical topics from this question. "
            "Include primary medical conditions, subtopics, treatments, and related concepts. "
            "For example, 'treating hypertension' should yield 'Hypertension' and 'Hypertension treatment'."
        )
        self.response_model = Question

# Relations extraction processor
class Llama3ExtractRelationsFromText(Llama3Processor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.system_prompt = (
            "You are a medical knowledge extraction expert. Extract meaningful relationships from the provided text. "
            "Identify entities like diseases, treatments, drugs, symptoms, and risk factors, and relationships between them. "
            "Represent relationships as triples with 'head' (subject), 'predicate' (relationship), and 'tail' (object). "
            "Focus on medically relevant relationships: 'treats', 'causes', 'associated with', 'risk factor for', 'prevents', 'reduces'. "
            "Return a JSON object with a 'triples' array."
        )
        self.user_input = (
            "Extract structured medical knowledge as relationship triples. "
            "Example: 'Aspirin treats headaches' becomes {\"head\": \"Aspirin\", \"predicate\": \"treats\", \"tail\": \"Headache\"}. "
            "Focus on actionable medical relationships."
        )
        self.response_model = Triples
