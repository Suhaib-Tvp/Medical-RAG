from Bio import Entrez

ENTREZ_EMAIL = "suhaib.tvp.in@gmail.com"
Entrez.email = ENTREZ_EMAIL

def get_pubmed_term_entries(term, max_count=5):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_count)
    records = Entrez.read(handle)
    handle.close()
    return records

def get_abstracts_text(id_list):
    abstracts = []
    if not id_list:
        return abstracts
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    abstracts_text = handle.read()
    handle.close()
    return [abstracts_text]
