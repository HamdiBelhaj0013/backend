import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .utils import find_relevant_chunks


class DirectChatView(APIView):
    """Simple view for direct testing without conversations"""
    permission_classes = [AllowAny]

    def post(self, request):
        # Extract query from request
        query = request.data.get('query', '')
        if not query:
            return Response({"error": "No query provided. Please include a 'query' field."}, status=400)

        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(query)

        # Use rule-based responses instead of LLM for reliability
        response_text = self._generate_rule_based_response(query, relevant_chunks)

        # Return response
        return Response({
            "query": query,
            "response": response_text,
            "relevant_chunks": [
                {"content": chunk.content[:100] + "..."}
                for chunk in relevant_chunks[:3]
            ]
        })

    def _generate_rule_based_response(self, query, relevant_chunks):
        """Generate a rule-based response"""
        query_lower = query.lower()

        # Create responses for common questions
        if any(word in query_lower for word in ["créer", "constituer", "fonder"]) and "association" in query_lower:
            return """Pour créer une association en Tunisie selon le décret-loi n° 2011-88, vous devez:

1. Adresser une lettre recommandée au secrétaire général du gouvernement contenant:
   - Une déclaration avec le nom, l'objet, les objectifs et le siège de l'association
   - Copies des cartes d'identité des fondateurs
   - Deux exemplaires des statuts signés

2. Après réception de l'accusé, déposer une annonce à l'Imprimerie Officielle dans les 7 jours.

L'association est légalement constituée dès l'envoi de la lettre et acquiert la personnalité morale après publication au Journal Officiel."""

        elif any(word in query_lower for word in ["statut", "statuts"]):
            return """Selon l'article 10 du décret-loi n° 2011-88, les statuts d'une association doivent contenir:

1. La dénomination officielle en arabe et éventuellement en langue étrangère
2. L'adresse du siège principal
3. Une présentation des objectifs et des moyens de réalisation
4. Les conditions d'adhésion et les droits/obligations des membres
5. L'organigramme, le mode d'élection et les prérogatives des organes
6. L'organe responsable des modifications et des décisions de dissolution/fusion
7. Les modes de prise de décision et de règlement des différends
8. Le montant de la cotisation s'il existe"""

        elif any(word in query_lower for word in ["finance", "financement", "ressource", "budget", "argent"]):
            return """Selon l'article 34 du décret-loi n° 2011-88, les ressources d'une association se composent de:

1. Les cotisations de ses membres
2. Les aides publiques
3. Les dons, donations et legs d'origine nationale ou étrangère
4. Les recettes résultant de ses biens, activités et projets

L'association est tenue de consacrer ses ressources aux activités nécessaires à la réalisation de ses objectifs (article 37).

Il est interdit d'accepter des aides d'États n'ayant pas de relations diplomatiques avec la Tunisie (article 35)."""

        elif "dissolution" in query_lower:
            return """La dissolution d'une association selon le décret-loi n° 2011-88 peut être:

1. Volontaire: par décision de ses membres conformément aux statuts
2. Judiciaire: par jugement du tribunal

En cas de dissolution volontaire, l'association doit:
- Informer le secrétaire général dans les 30 jours suivant la décision
- Désigner un liquidateur judiciaire
- Présenter un état de ses biens pour s'acquitter de ses obligations

Le reliquat sera distribué selon les statuts ou attribué à une association similaire."""

        elif any(word in query_lower for word in ["membre", "adhérer", "adhésion"]):
            return """Selon l'article 17 du décret-loi n° 2011-88, un membre d'association doit:

1. Être de nationalité tunisienne ou résident en Tunisie
2. Avoir au moins 13 ans
3. Accepter par écrit les statuts de l'association
4. Verser la cotisation requise

Les fondateurs et dirigeants ne peuvent pas être en charge de responsabilités dans des partis politiques (article 9).

Les membres et salariés doivent éviter les conflits d'intérêts (article 18)."""

        else:
            # For other questions, return a generic response
            return """D'après le décret-loi n° 2011-88 sur les associations en Tunisie, je n'ai pas de réponse spécifique préparée pour cette question.

Les principales catégories d'informations disponibles concernent:
- La création d'associations
- Les statuts d'associations
- Le financement et les ressources
- La dissolution d'associations
- Les conditions d'adhésion et les membres

Pourriez-vous reformuler votre question dans l'une de ces catégories?"""
def unload_model(self):
    """Unload the model to free memory"""
    if self.llm is not None:
        del self.llm
        import gc
        gc.collect()
        self.llm = None
        self.model_loaded = False
        logger.info("LLM unloaded from memory")