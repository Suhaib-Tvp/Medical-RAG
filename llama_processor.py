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
        # Limit text to avoid token overflow
        text = text[:2000]
        user_input = f"{self.user_input}:\n{text}"
        
        try:
            schema = self.response_model.model_json_schema()
        except:
            schema = self.response_model.schema()
            
        format_instructions = f"Return valid JSON matching this schema: {json.dumps(schema)}"
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown, no extra text."
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
            
            # Remove markdown code blocks if present
            if content.startswith('```
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            
            # Clean up potential markdown
            if 'json' in content[:10]:
                content = content.replace('```json', '').replace('```
            
            content = content.strip()
            
            # Parse JSON
            parsed_json = json.loads(content)
            
            # Validate with Pydantic
            try:
                parsed = self.response_model.model_validate(parsed_json)
            except:
                parsed = self.response_model(**parsed_json)
            
            try:
                return parsed.model_dump_json()
            except:
                return json.dumps(parsed.dict())
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            print(f"[ERROR] Content: {content[:200]}")
            # Return empty structure
            if self.response_model == Question:
                return json.dumps({"topics": []})
            elif self.response_model == Triples:
                return json.dumps({"triples": []})
            return "{}"
            
        except Exception as e:
            print(f"[ERROR] Error in prompt_llama3: {e}")
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
            "Focus on specific conditions, treatments, and relevant subtopics. "
            "Always return valid JSON."
        )
        self.user_input = (
            "Extract medical topics from this question. "
            "Return JSON: {\"topics\": [\"topic1\", \"topic2\", ...]}"
        )
        self.response_model = Question

# Relations extraction processor
class Llama3ExtractRelationsFromText(Llama3Processor):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.system_prompt = (
            "You are a medical knowledge extraction expert. Extract medical relationships from medical text. "
            "Identify diseases, treatments, drugs, symptoms, and their relationships. "
            "Return triples in JSON format. Always respond with valid JSON only."
        )
        self.user_input = (
            "Extract medical relationships from this text. "
            "Return JSON: {\"triples\": [{\"head\": \"subject\", \"predicate\": \"relationship\", \"tail\": \"object\"}, ...]}. "
            "Focus on: treats, causes, associated_with, risk_factor_for, prevents, reduces. "
            "Extract at least 2-5 relationships."
        )
        self.response_model = Triples
