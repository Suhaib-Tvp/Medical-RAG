from groq import Groq
from pydantic import BaseModel, Field
from typing import List
import json

# =====================
# PYDANTIC MODELS
# =====================

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


# =====================
# BASE PROCESSOR CLASS
# =====================

class Llama3Processor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.user_input = None
        self.system_prompt = None
        self.response_model = None
    
    def prompt_llama3(self, text: str) -> str:
        """
        Send prompt to Llama 3.3 and get structured JSON response.
        """
        text = text[:2000]
        user_input = f"{self.user_input}:\n{text}"
        
        try:
            schema = self.response_model.model_json_schema()
        except:
            schema = self.response_model.schema()
        
        format_instructions = f"Return valid JSON: {json.dumps(schema)}"
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}\n\nRespond with ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"{format_instructions}\n\n{user_input}"
                    }
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown blocks
            if content.startswith('```
                content = content.replace('```json', '').replace('```
            
            parsed_json = json.loads(content)
            
            try:
                parsed = self.response_model.model_validate(parsed_json)
                return parsed.model_dump_json()
            except:
                return json.dumps(parsed_json)
        
        except json.JSONDecodeError:
            if self.response_model == Question:
                return json.dumps({"topics": []})
            elif self.response_model == Triples:
                return json.dumps({"triples": []})
            return "{}"
        
        except Exception as e:
            print(f"Error: {e}")
            if self.response_model == Question:
                return json.dumps({"topics": []})
            elif self.response_model == Triples:
                return json.dumps({"triples": []})
            return "{}"


# =====================
# TOPIC EXTRACTION
# =====================

class Llama3ExtractTopicFromText(Llama3Processor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.system_prompt = (
            "Extract medical topics from the question. "
            "Focus on diseases, treatments, and conditions."
        )
        self.user_input = (
            "Extract topics. Return JSON: {\"topics\": [\"topic1\", \"topic2\"]}"
        )
        self.response_model = Question


# =====================
# RELATIONS EXTRACTION
# =====================

class Llama3ExtractRelationsFromText(Llama3Processor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.system_prompt = (
            "Extract medical relationships from text. "
            "Identify diseases, treatments, drugs, and relationships."
        )
        self.user_input = (
            "Extract relationships. "
            "Return JSON: {\"triples\": [{\"head\": \"X\", \"predicate\": \"Y\", \"tail\": \"Z\"}]}"
        )
        self.response_model = Triples
