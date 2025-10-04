from groq import Groq
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

class Llama3ExtractTopicFromText:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def prompt_llama3(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medical expert. Extract key medical topics from the user's query."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content

class Llama3ExtractRelationsFromText:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def prompt_llama3(self, abstracts):
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medical expert extracting relationships in triples (head, predicate, tail) from medical abstracts."},
                {"role": "user", "content": f"Extract relationships from the following abstracts:\n{abstracts}"}
            ],
        )
        return response.choices[0].message.content
