from Bio import Entrez
from Bio.Entrez import efetch
from typing import List
import xml.etree.ElementTree as ET

# Set email for NCBI (required)
Entrez.email = "your.email@example.com"

def get_pubmed_term_entries(term: str, num_ids: int = 5) -> dict:
    """
    Search PubMed for articles related to a specific term.
    
    Args:
        term (str): Medical term to search for
        num_ids (int): Maximum number of articles to retrieve
        
    Returns:
        dict: PubMed search results with IdList
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=num_ids)
        record = Entrez.read(handle)
        handle.close()
        return record
    except Exception as e:
        print(f"Error searching PubMed for '{term}': {e}")
        return {'IdList': [], 'Count': 0}

def get_articles_in_xml(ids: List[str]) -> str:
    """
    Retrieve multiple articles in XML format by their PubMed IDs.
    
    Args:
        ids (list): List of PubMed article IDs
        
    Returns:
        str: XML content of the articles
    """
    if not ids:
        return ""
    
    try:
        id_str = ",".join(ids)
        handle = efetch(db='pubmed', id=id_str, retmode='xml', rettype='abstract')
        content = handle.read()
        handle.close()
        return content
    except Exception as e:
        print(f"Error fetching articles for IDs {ids}: {e}")
        return ""

def get_abstracts_from_xml(abstracts_xml: str) -> List[str]:
    """
    Extract abstract texts from XML content.
    
    Args:
        abstracts_xml (str): XML content containing abstracts
        
    Returns:
        list: List of abstract texts
    """
    if not abstracts_xml:
        return []
    
    try:
        response_xml = ET.fromstring(abstracts_xml)
        abstract_texts = response_xml.findall('.//Abstract')
        
        joined_parts = [
            " ".join([
                "".join(child.itertext()) 
                for child in abstract 
                if child.tag == 'AbstractText'
            ]) 
            for abstract in abstract_texts
        ]
        return [text for text in joined_parts if text.strip()]
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return []

def get_abstracts_text(ids: List[str]) -> List[str]:
    """
    Get abstracts from PubMed articles by their IDs.
    
    Args:
        ids (list): List of PubMed article IDs
        
    Returns:
        list: List of abstract texts
    """
    articles_xml = get_articles_in_xml(ids)
    return get_abstracts_from_xml(articles_xml)
