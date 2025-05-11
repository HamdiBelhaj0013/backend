LEGAL_CONCEPTS = {
    "création": ["10", "11", "12"],
    "constitution": ["10", "11", "12"],
    "statuts": ["10", "16", "19"],
    "membres": ["8", "9", "15", "17", "18"],
    "adhésion": ["17"],
    "ressources": ["34", "35", "36", "37"],
    "financement": ["34", "35", "36", "37", "41"],
    "dissolution": ["19", "33", "45"],
    "sanctions": ["45"],
    "réseau": ["26", "27", "28", "29", "30", "31"],
    "étrangères": ["20", "21", "22", "23", "24", "25"],
}


LEGAL_TERMS = {
    "personnalité morale": "Capacité d'une association d'exercer des droits et des obligations distinctement de ses membres",
    "liquidateur": "Personne chargée de régler les affaires d'une association lors de sa dissolution",
    "statuts": "Document fondamental qui définit l'organisation, le fonctionnement et les objectifs d'une association",
    "dissolution": "Acte juridique mettant fin à l'existence d'une association",
    "déclaration": "Procédure administrative pour la constitution d'une association",
    "aides étrangères": "Soutien financier provenant de sources non tunisiennes, soumis à déclaration",
}


def get_relevant_articles(query):
    """
    Identify relevant articles based on query keywords
    """
    query_lower = query.lower()
    relevant_articles = []

    for concept, articles in LEGAL_CONCEPTS.items():
        if concept in query_lower or any(term in query_lower for term in concept.split()):
            relevant_articles.extend(articles)

    # Remove duplicates while preserving order
    unique_articles = []
    for article in relevant_articles:
        if article not in unique_articles:
            unique_articles.append(article)

    return unique_articles


def get_legal_term_definitions(query):
    """
    Extract legal term explanations relevant to the query
    """
    query_lower = query.lower()
    relevant_terms = {}

    for term, definition in LEGAL_TERMS.items():
        if term in query_lower:
            relevant_terms[term] = definition

    return relevant_terms