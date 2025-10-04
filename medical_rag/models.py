from pydantic import BaseModel

class Triple(BaseModel):
    head: str
    predicate: str
    tail: str
