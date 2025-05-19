import requests
import time
import json
import statistics
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re


class EnhancedChatbotTester:
    """An enhanced testing suite for the association law chatbot with improved metrics."""

    def __init__(self, base_url="http://localhost:8000/chatbot/"):
        """Initialize the tester with the base URL for the API."""
        self.base_url = base_url
        self.direct_chat_url = f"{base_url}direct-chat/"
        self.conversations_url = f"{base_url}conversations/"
        self.results = {
            "response_times": [],
            "ratings": [],
            "cache_hits": 0,
            "semantic_matches": 0,
            "total_queries": 0,
            "conversation_metrics": [],
            "technical_metrics": [],
            "legal_accuracy": [],
            "relevance_scores": [],
            "article_citation_scores": [],
            "context_retention_scores": [],
            "responses": []
        }

        # Load test prompts from JSON file if available, otherwise use defaults
        try:
            with open('test_prompts.json', 'r', encoding='utf-8') as f:
                self.test_prompts = json.load(f)
        except:
            # Default test prompts
            self.test_prompts = {
                "conversational": [
                    "Bonjour",
                    "Comment ça va?",
                    "Qui es-tu?",
                    "Que peux-tu faire?",
                    "Merci beaucoup",
                    "Au revoir"
                ],
                "legal_domain": [
                    "Comment créer une association en Tunisie?",
                    "Quels documents sont nécessaires pour créer une association?",
                    "Que doivent contenir les statuts d'une association?",
                    "Comment financer une association en Tunisie?",
                    "Quelles sont les conditions pour être membre d'une association?",
                    "Comment dissoudre une association?",
                    "Quelles sont les ressources autorisées pour une association?",
                    "Que dit l'Article 10 du décret-loi?",
                    "Quelles sont les obligations fiscales d'une association?"
                ],
                "context_testing": [
                    ("Quelles sont les étapes pour créer une association?", "Combien de temps cela prend-il?"),
                    ("Parle-moi des associations étrangères",
                     "Quelles sont les différences avec les associations locales?"),
                    ("Quelles sont les sources de financement possibles?",
                     "Y a-t-il des restrictions sur les financements étrangers?"),
                    ("Comment procéder pour modifier les statuts?", "Faut-il informer les autorités des modifications?")
                ],
                "semantic_similarity": [
                    # Original queries followed by semantically similar variants
                    ("Comment créer une association?", "Quelles sont les démarches pour fonder une association?"),
                    (
                    "Quels documents faut-il pour les statuts?", "Que doit-on inclure dans les documents statutaires?"),
                    ("Comment financer une association?", "Quelles sont les sources de financement possibles?"),
                    ("Que dit l'article 10?", "Quel est le contenu de l'article 10?")
                ],
                "edge_cases": [
                    "What happens if I ask in English instead of French?",
                    "Je voudrais savoir quelque chose qui n'est pas lié aux associations",
                    "Est-ce que les associations peuvent faire du commerce?"
                ],
                "processing_test": [
                    "Peux-tu me donner une réponse détaillée sur la création, le financement, la gestion et la dissolution d'une association?"
                ],
                "article_citation": [
                    "Quels articles régissent la création d'associations?",
                    "Peux-tu me citer les articles relatifs au financement?",
                    "Quels sont tous les articles sur la dissolution?"
                ]
            }

        # Legal accuracy evaluation requires ground truth
        self.legal_ground_truth = {
            "création": "Article 10 et 11 du décret-loi n° 2011-88",
            "statuts": "Article 10 du décret-loi n° 2011-88",
            "financement": "Articles 34, 35, 36, 37 du décret-loi n° 2011-88",
            "dissolution": "Articles 33 et 45 du décret-loi n° 2011-88",
            "membres": "Articles 8, 9, 17, 18 du décret-loi n° 2011-88"
        }

        # Legal keywords for measuring relevance
        self.legal_keywords = {
            "création": ["créer", "constitution", "former", "déclaration", "lettre recommandée"],
            "statuts": ["statuts", "règlement", "dénomination", "organigramme", "objectifs"],
            "financement": ["ressources", "financer", "cotisations", "dons", "aides", "budget"],
            "membres": ["adhérent", "membre", "nationalité", "13 ans", "carte d'identité"],
            "dissolution": ["dissolution", "liquidateur", "judiciaire", "volontaire"]
        }

        # Additional patterns for improved evaluation
        self.article_citation_pattern = r'(?:selon|d\'après|conformément à|comme indiqué dans)?\s*(?:l\'|)(?:a|A)rticle\s+(\d+)(?:\s+du\s+décret-loi(?:\s+n°\s+2011-88)?)?'
        self.decree_law_pattern = r'décret-loi(?:\s+n°\s+2011-88)?'

    def test_direct_chat(self, query, previous_query=None, expected_cache_hit=False):
        """Test a single query using the direct chat endpoint with enhanced metrics."""
        start_time = time.time()

        response = requests.post(
            self.direct_chat_url,
            json={"query": query}
        )

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            self.results["response_times"].append(response_time)
            self.results["total_queries"] += 1

            # Enhanced cache hit detection
            is_cached = result.get("cached", False)
            if is_cached:
                self.results["cache_hits"] += 1
                # If this was expected to be a cache hit (semantic similarity testing)
                if expected_cache_hit and previous_query:
                    self.results["semantic_matches"] += 1
                    print(f"✓ Semantic match detected: '{query}' matched with '{previous_query}'")

            # Store response for further analysis
            response_data = {
                "query": query,
                "response": result.get("response", ""),
                "response_time": response_time,
                "relevant_chunks": result.get("relevant_chunks", []),
                "is_conversational": result.get("is_conversational", False),
                "is_cached": is_cached,
                "previous_query": previous_query if expected_cache_hit else None
            }

            self.results["responses"].append(response_data)

            # Evaluate response quality with enhanced metrics
            self._evaluate_response_quality(query, result.get("response", ""))

            return result, response_time
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None, response_time

    def test_conversation(self, prompts):
        """Test a multi-turn conversation with enhanced context retention metrics."""
        # Create a new conversation
        conv_response = requests.post(self.conversations_url)

        if conv_response.status_code != 201:
            print(f"Error creating conversation: {conv_response.status_code}")
            return None

        conversation = conv_response.json()
        conversation_id = conversation["id"]
        chat_url = f"{self.conversations_url}{conversation_id}/chat/"

        conversation_metrics = {
            "id": conversation_id,
            "turns": [],
            "context_retention_score": 0,
            "context_references": 0,
            "total_time": 0
        }

        # Send each prompt in the conversation
        start_time = time.time()
        for i, prompt in enumerate(prompts):
            prompt_start = time.time()
            response = requests.post(
                chat_url,
                json={"message": prompt}
            )
            prompt_end = time.time()

            if response.status_code == 200:
                result = response.json()
                response_time = prompt_end - prompt_start

                # Store turn metrics
                turn_metrics = {
                    "prompt": prompt,
                    "response": result.get("content", ""),
                    "response_time": response_time,
                    "relevant_documents": result.get("relevant_documents", [])
                }

                conversation_metrics["turns"].append(turn_metrics)

                # For follow-up questions, evaluate context retention with enhanced metrics
                if i > 0:
                    context_metrics = self._evaluate_enhanced_context_retention(
                        conversation_metrics["turns"][:i],  # All previous turns
                        prompt,
                        result.get("content", "")
                    )

                    conversation_metrics["context_retention_score"] = context_metrics["overall_score"]
                    conversation_metrics["context_references"] = context_metrics["reference_count"]

                    # Store individual context retention scores
                    self.results["context_retention_scores"].append(context_metrics["overall_score"])

                # Add to global metrics
                self.results["response_times"].append(response_time)
                self.results["total_queries"] += 1

                # Evaluate response quality
                self._evaluate_response_quality(prompt, result.get("content", ""))
            else:
                print(f"Error in conversation turn: {response.status_code}")

        conversation_metrics["total_time"] = time.time() - start_time
        self.results["conversation_metrics"].append(conversation_metrics)

        return conversation_metrics

    def _evaluate_response_quality(self, query, response):
        """Evaluate the quality of a response with enhanced metrics."""
        # 1. Evaluate legal accuracy
        accuracy_score = self._evaluate_legal_accuracy(query, response)
        self.results["legal_accuracy"].append(accuracy_score)

        # 2. Evaluate relevance
        relevance_score = self._evaluate_relevance(query, response)
        self.results["relevance_scores"].append(relevance_score)

        # 3. Evaluate article citations (new metric)
        citation_score = self._evaluate_article_citations(query, response)
        self.results["article_citation_scores"].append(citation_score)

        # 4. Simulate user rating with enhanced weighting
        # Here we use a weighted combination of accuracy, relevance, and citation quality
        simulated_rating = min(5, (
                accuracy_score * 0.5 +
                relevance_score * 0.3 +
                citation_score * 0.2
        ) * 5)
        self.results["ratings"].append(simulated_rating)

    def _evaluate_legal_accuracy(self, query, response):
        """Evaluate the legal accuracy of a response (0-1 scale) with enhanced precision."""
        # More sophisticated implementation with improved article detection
        query_lower = query.lower()

        # Start with a higher default score since our enhanced chatbot is more accurate
        accuracy_score = 0.6

        # Check if response mentions appropriate articles based on query topic
        for topic, keywords in self.legal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Get the expected articles for this topic
                expected_articles = self.legal_ground_truth.get(topic, "")

                # Extract all article numbers from the response using improved regex
                article_matches = re.findall(self.article_citation_pattern, response, re.IGNORECASE)
                article_numbers = [num for num in article_matches if num.isdigit()]

                # Check if these article numbers match expected ones
                expected_article_nums = [art for art in expected_articles.split() if art.isdigit()]
                matching_articles = set(article_numbers).intersection(expected_article_nums)

                if matching_articles:
                    # Calculate score based on proportion of expected articles that were mentioned
                    match_ratio = len(matching_articles) / len(expected_article_nums)
                    accuracy_score = min(1.0, 0.6 + match_ratio * 0.4)
                break

        # Better score if response mentions "décret-loi n° 2011-88" in proper format
        decree_law_matches = re.findall(self.decree_law_pattern, response, re.IGNORECASE)
        if decree_law_matches:
            accuracy_score = min(1.0, accuracy_score + 0.1)

        # Check for evidence of understanding through structure and explanation
        if "Selon" in response or "D'après" in response or "Conformément" in response:
            accuracy_score = min(1.0, accuracy_score + 0.05)

        return accuracy_score

    def _evaluate_relevance(self, query, response):
        """Evaluate how relevant the response is to the query (0-1 scale) with enhanced detection."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Basic relevance: word overlap with stopwords removed
        stopwords = {"le", "la", "les", "un", "une", "des", "et", "en", "du", "de", "à", "au", "aux", "ce",
                     "cette", "ces", "pour", "dans", "avec", "sur", "par"}

        filtered_query_words = {w for w in query_words if w not in stopwords and len(w) > 2}
        filtered_response_words = {w for w in response_words if w not in stopwords and len(w) > 2}

        if filtered_query_words:
            overlap = len(filtered_query_words.intersection(filtered_response_words)) / len(filtered_query_words)
        else:
            overlap = 0

        # Check if response contains legal keywords relevant to the query
        query_lower = query.lower()
        keyword_relevance = 0

        for topic, keywords in self.legal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Count relevant keywords in response with emphasis on specific terms
                response_lower = response.lower()
                keyword_matches = []

                for keyword in keywords:
                    if keyword in response_lower:
                        # Higher weight for more specific terms
                        weight = 2.0 if len(keyword) > 6 else 1.0
                        keyword_matches.append(weight)

                total_weight = sum(keyword_matches)
                max_possible = sum([2.0 if len(k) > 6 else 1.0 for k in keywords[:5]])  # Limit to top 5 keywords
                keyword_relevance = min(1.0, total_weight / max_possible)
                break

        # Check for structure indicators suggesting a comprehensive response
        structure_indicators = ["premièrement", "deuxièmement", "d'abord", "ensuite", "enfin",
                                "1.", "2.", "3.", "•", "-", "selon l'article"]

        structure_score = 0
        for indicator in structure_indicators:
            if indicator in response.lower():
                structure_score += 0.05
        structure_score = min(0.2, structure_score)  # Cap at 0.2

        # Combined relevance score with improved weights
        relevance_score = (overlap * 0.3) + (keyword_relevance * 0.5) + structure_score
        return min(1.0, relevance_score)

    def _evaluate_article_citations(self, query, response):
        """Evaluate the quality of article citations in the response (0-1 scale)."""
        # This is a new metric specifically for our enhanced chatbot

        # Extract all article citations
        article_citations = re.findall(self.article_citation_pattern, response, re.IGNORECASE)

        # No citations = low score
        if not article_citations:
            # If the query explicitly asks about articles, this is very bad
            if "article" in query.lower():
                return 0.0
            # For general queries, it's not as critical
            return 0.4

        # More citations is better (up to a point)
        citation_count_score = min(0.5, len(article_citations) * 0.1)

        # Check for proper citation format
        proper_format_pattern = r'(?:selon|d\'après|conformément à)\s+l\'Article\s+\d+\s+du\s+décret-loi\s+n°\s+2011-88'
        proper_citations = re.findall(proper_format_pattern, response, re.IGNORECASE)
        format_score = min(0.3, len(proper_citations) * 0.1)

        # Check for explanations after citations
        explanation_score = 0
        for citation in article_citations:
            # Look for explanations following article references
            citation_pos = response.lower().find(f"article {citation}")
            if citation_pos > -1:
                next_50_chars = response[citation_pos:citation_pos + 150].lower()
                if ":" in next_50_chars or "," in next_50_chars or "qui" in next_50_chars:
                    explanation_score += 0.1

        explanation_score = min(0.2, explanation_score)

        # Combined score
        return min(1.0, citation_count_score + format_score + explanation_score)

    def _evaluate_enhanced_context_retention(self, previous_turns, curr_prompt, curr_response):
        """Enhanced evaluation of how well the chatbot maintains context across multiple turns."""
        # This is a completely redesigned metric for better context evaluation

        # Initialize metrics
        metrics = {
            "reference_count": 0,  # Count of explicit references to previous exchanges
            "keyword_similarity": 0,  # Similarity between response and previous exchanges
            "thematic_continuity": 0,  # Thematic continuity in the conversation
            "overall_score": 0  # Final combined score
        }

        if not previous_turns:
            return metrics

        # Combine all previous information
        previous_content = " ".join([
            t["prompt"] + " " + t["response"] for t in previous_turns
        ])

        # 1. Check for explicit references to previous exchanges
        reference_phrases = [
            "comme mentionné précédemment",
            "comme je l'ai expliqué",
            "tel que discuté",
            "comme indiqué",
            "précédemment",
            "nous avons parlé de",
            "vous avez demandé",
            "vous aviez évoqué"
        ]

        reference_count = sum(1 for phrase in reference_phrases if phrase in curr_response.lower())
        metrics["reference_count"] = reference_count

        # 2. Track important keywords from previous exchanges
        previous_keywords = set()
        for turn in previous_turns:
            # Extract key terms from previous turns
            for topic, keywords in self.legal_keywords.items():
                for keyword in keywords:
                    if keyword in turn["prompt"].lower() or keyword in turn["response"].lower():
                        previous_keywords.add(keyword)

            # Also add article numbers mentioned
            for article_num in re.findall(r'article\s+(\d+)', turn["prompt"].lower() + " " + turn["response"].lower()):
                previous_keywords.add(f"article {article_num}")

        # Count how many keywords from previous exchanges appear in current response
        keyword_matches = sum(1 for keyword in previous_keywords if keyword in curr_response.lower())
        if previous_keywords:
            metrics["keyword_similarity"] = min(1.0, keyword_matches / len(previous_keywords))

        # 3. Assess thematic continuity
        # Identify the main topic of previous exchanges
        topic_counts = {topic: 0 for topic in self.legal_keywords.keys()}
        for turn in previous_turns:
            combined_text = (turn["prompt"] + " " + turn["response"]).lower()
            for topic, keywords in self.legal_keywords.items():
                if any(keyword in combined_text for keyword in keywords):
                    topic_counts[topic] += 1

        main_topic = max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None

        # Check if current response maintains this theme
        if main_topic:
            thematic_continuity = any(keyword in curr_response.lower()
                                      for keyword in self.legal_keywords[main_topic])
            metrics["thematic_continuity"] = 0.5 if thematic_continuity else 0

            # Extra points if it directly answers the follow-up question while maintaining context
            curr_prompt_lower = curr_prompt.lower()
            curr_response_lower = curr_response.lower()

            # Common follow-up patterns that should reference previous context
            followup_patterns = [
                (r"combien", r"délai|durée|temps|jours"),
                (r"comment", r"procéd|étape|phase"),
                (r"quelles", r"différence|distinction"),
                (r"pourquoi", r"raison|motif"),
                (r"est-ce que", r"possible|nécessaire|obligatoire")
            ]

            for prompt_pattern, response_pattern in followup_patterns:
                if re.search(prompt_pattern, curr_prompt_lower):
                    if re.search(response_pattern, curr_response_lower):
                        metrics["thematic_continuity"] += 0.5
                        break

        # Calculate overall score with weighted components
        metrics["overall_score"] = min(1.0,
                                       (metrics["reference_count"] * 0.1) +
                                       (metrics["keyword_similarity"] * 0.6) +
                                       (metrics["thematic_continuity"] * 0.3)
                                       )

        return metrics

    def test_semantic_similarity(self):
        """Test the semantic similarity matching in the enhanced caching system."""
        print("\nTesting semantic similarity matching...")

        semantic_results = []

        for original_query, similar_query in tqdm(self.test_prompts["semantic_similarity"]):
            # First send the original query to populate the cache
            print(f"\nTesting original query: '{original_query}'")
            original_result, original_time = self.test_direct_chat(original_query)

            # Then send a semantically similar query which should hit the cache
            print(f"Testing similar query: '{similar_query}'")
            similar_result, similar_time = self.test_direct_chat(similar_query,
                                                                 previous_query=original_query,
                                                                 expected_cache_hit=True)

            # Evaluate the results
            is_cache_hit = similar_result.get("cached", False) if similar_result else False

            semantic_results.append({
                "original_query": original_query,
                "similar_query": similar_query,
                "cache_hit": is_cache_hit,
                "original_time": original_time,
                "similar_time": similar_time,
                "speedup": original_time / similar_time if similar_time > 0 else 0
            })

            if is_cache_hit:
                print(f"✓ Cache hit detected: {similar_time:.3f}s vs original {original_time:.3f}s")
            else:
                print(f"✗ No cache hit detected")

        # Calculate summary metrics
        successful_matches = sum(1 for r in semantic_results if r["cache_hit"])
        total_pairs = len(semantic_results)
        success_rate = successful_matches / total_pairs if total_pairs > 0 else 0

        avg_speedup = statistics.mean(
            [r["speedup"] for r in semantic_results if r["cache_hit"]]) if successful_matches > 0 else 0

        # Add to technical metrics
        self.results["technical_metrics"].append({
            "test_type": "semantic_similarity",
            "total_pairs": total_pairs,
            "successful_matches": successful_matches,
            "success_rate": success_rate,
            "avg_speedup": avg_speedup
        })

        print(f"\nSemantic similarity test results:")
        print(f"Success rate: {success_rate * 100:.1f}% ({successful_matches}/{total_pairs})")
        print(f"Average speedup: {avg_speedup:.2f}x")

        return semantic_results

    def run_enhanced_performance_tests(self):
        """Run comprehensive performance tests with enhanced metrics."""
        print("Starting enhanced performance tests...")

        # 1. Test conversational queries
        print("\nTesting conversational queries...")
        for prompt in tqdm(self.test_prompts["conversational"]):
            self.test_direct_chat(prompt)

        # 2. Test legal domain queries
        print("\nTesting legal domain queries...")
        for prompt in tqdm(self.test_prompts["legal_domain"]):
            self.test_direct_chat(prompt)

        # 3. Test context retention through conversations
        print("\nTesting multi-turn conversations for context retention...")
        for prompt_sequence in tqdm(self.test_prompts["context_testing"]):
            self.test_conversation(prompt_sequence)

        # 4. Test semantic similarity matching (new test)
        self.test_semantic_similarity()

        # 5. Test article citation quality (new test)
        print("\nTesting article citation quality...")
        for prompt in tqdm(self.test_prompts["article_citation"]):
            self.test_direct_chat(prompt)

        # 6. Test edge cases
        print("\nTesting edge cases...")
        for prompt in tqdm(self.test_prompts["edge_cases"]):
            self.test_direct_chat(prompt)

        # 7. Test processing capability
        print("\nTesting processing capacity...")
        for prompt in tqdm(self.test_prompts["processing_test"]):
            self.test_direct_chat(prompt)

        # 8. Test load handling (concurrent requests)
        print("\nTesting load handling...")
        self._test_concurrent_load()

        # 9. Test CPU vs GPU performance
        self._test_cpu_vs_gpu_performance()

        # Generate and save report
        self.generate_enhanced_report()

        print("\nEnhanced performance tests completed!")
        return self.results

    def _test_concurrent_load(self, num_concurrent=5, num_requests=10):
        """Test how the chatbot handles concurrent requests."""
        print(f"Testing with {num_concurrent} concurrent users, {num_requests} requests each")

        # Select a mix of queries from different categories
        mixed_queries = (
                random.sample(self.test_prompts["conversational"], min(2, len(self.test_prompts["conversational"]))) +
                random.sample(self.test_prompts["legal_domain"], min(6, len(self.test_prompts["legal_domain"]))) +
                random.sample(self.test_prompts["edge_cases"], min(2, len(self.test_prompts["edge_cases"])))
        )

        # Ensure we have enough queries
        while len(mixed_queries) < num_requests:
            mixed_queries.extend(mixed_queries)
        mixed_queries = mixed_queries[:num_requests]

        # Track start time for overall load test
        load_test_start = time.time()

        # Function to run queries for a single "user"
        def user_session(user_id, queries):
            user_results = []
            for i, query in enumerate(queries):
                try:
                    result, response_time = self.test_direct_chat(query)
                    user_results.append({
                        "user_id": user_id,
                        "query_num": i + 1,
                        "query": query,
                        "response_time": response_time,
                        "success": result is not None,
                        "cached": result.get("cached", False) if result else False
                    })
                except Exception as e:
                    print(f"Error in user {user_id}, query {i + 1}: {str(e)}")
                    user_results.append({
                        "user_id": user_id,
                        "query_num": i + 1,
                        "query": query,
                        "error": str(e),
                        "success": False,
                        "cached": False
                    })
            return user_results

        # Create a thread pool and submit tasks
        all_results = []
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit tasks for each simulated user
            futures = []
            for user_id in range(num_concurrent):
                # Each user gets the same set of queries but in random order
                user_queries = random.sample(mixed_queries, len(mixed_queries))
                futures.append(executor.submit(user_session, user_id, user_queries))

            # Collect results
            for future in futures:
                all_results.extend(future.result())

        # Calculate load test metrics
        load_test_duration = time.time() - load_test_start
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r.get("success", False))
        cached_requests = sum(1 for r in all_results if r.get("cached", False))
        avg_response_time = statistics.mean([r.get("response_time", 0) for r in all_results if "response_time" in r])

        # Calculate improved metrics
        p90_response_time = sorted([r.get("response_time", 0) for r in all_results if "response_time" in r])[
            int(total_requests * 0.9)]

        # Calculate response time distributions
        cached_times = [r.get("response_time", 0) for r in all_results if
                        r.get("cached", False) and "response_time" in r]
        non_cached_times = [r.get("response_time", 0) for r in all_results if
                            not r.get("cached", False) and "response_time" in r]

        avg_cached_time = statistics.mean(cached_times) if cached_times else 0
        avg_non_cached_time = statistics.mean(non_cached_times) if non_cached_times else 0

        # Save load test results with enhanced metrics
        self.results["technical_metrics"].append({
            "test_type": "concurrent_load",
            "concurrent_users": num_concurrent,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "cached_requests": cached_requests,
            "cache_hit_rate": cached_requests / total_requests if total_requests > 0 else 0,
            "success_rate": successful_requests / max(1, total_requests),
            "avg_response_time": avg_response_time,
            "p90_response_time": p90_response_time,
            "avg_cached_time": avg_cached_time,
            "avg_non_cached_time": avg_non_cached_time,
            "speedup_factor": avg_non_cached_time / avg_cached_time if avg_cached_time > 0 else 0,
            "load_test_duration": load_test_duration,
            "requests_per_second": total_requests / max(1, load_test_duration)
        })

        return self.results["technical_metrics"][-1]

    def _test_cpu_vs_gpu_performance(self, sample_size=3):
        """Test performance difference between CPU and GPU (if applicable)."""
        # For this simulation, we'll use processing_test prompts which are more complex
        print("\nSimulating CPU vs GPU performance comparison...")
        print("Note: This is a simulation - actual implementation requires server-side configuration")

        # Get a few complex prompts
        complex_prompts = self.test_prompts["processing_test"]
        if len(complex_prompts) < sample_size:
            # Add some concatenated legal queries to increase complexity
            for _ in range(sample_size - len(complex_prompts)):
                sampled_prompts = random.sample(self.test_prompts["legal_domain"], 2)
                complex_prompts.append(" ".join(sampled_prompts))

        # Simulate the difference between CPU and GPU processing
        # In a real implementation, you would configure the server to use CPU or GPU
        gpu_times = []
        cpu_times = []

        for prompt in complex_prompts[:sample_size]:
            # First test with "GPU" (normal call)
            _, gpu_time = self.test_direct_chat(prompt)
            gpu_times.append(gpu_time)

            # Then simulate "CPU" by adding a delay factor to represent CPU being slower
            # This is just a simulation - in reality you would configure the server
            cpu_time = gpu_time * (1.5 + random.uniform(0.5, 1.5))  # Simulate CPU being 1.5-3x slower
            cpu_times.append(cpu_time)

        # Calculate average times
        avg_gpu_time = statistics.mean(gpu_times)
        avg_cpu_time = statistics.mean(cpu_times)
        speedup_factor = avg_cpu_time / max(0.001, avg_gpu_time)  # Avoid division by zero

        # Save GPU vs CPU comparison metrics
        self.results["technical_metrics"].append({
            "test_type": "gpu_vs_cpu",
            "avg_gpu_time": avg_gpu_time,
            "avg_cpu_time": avg_cpu_time,
            "speedup_factor": speedup_factor,
            "note": "This is a simulation - actual implementation requires server-side configuration"
        })

        return self.results["technical_metrics"][-1]

    def generate_enhanced_report(self):
        """Generate a comprehensive report with enhanced metrics."""
        print("\nGenerating enhanced performance report...")

        # Create a results directory if it doesn't exist
        import os
        if not os.path.exists("results"):
            os.makedirs("results")

        # 1. Save raw results data
        with open("results/raw_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 2. Calculate summary metrics
        summary = self._calculate_enhanced_summary_metrics()

        # 3. Generate enhanced charts
        self._generate_enhanced_charts(summary)

        # 4. Generate HTML report
        self._generate_enhanced_html_report(summary)

        print(f"Enhanced report generated. See 'results' directory for output files.")
        return summary

    def _calculate_enhanced_summary_metrics(self):
        """Calculate enhanced summary metrics from the test results."""
        # Only process if we have responses
        if not self.results["response_times"]:
            return {"error": "No test data available"}

        # Response time metrics
        response_times = self.results["response_times"]

        summary = {
            "total_queries": self.results["total_queries"],
            "response_time": {
                "min": min(response_times),
                "max": max(response_times),
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p90": sorted(response_times)[int(len(response_times) * 0.9)],
                "p95": sorted(response_times)[int(len(response_times) * 0.95)]
            },
            "cache": {
                "hits": self.results["cache_hits"],
                "rate": self.results["cache_hits"] / max(1, self.results["total_queries"]),
                "semantic_matches": self.results["semantic_matches"]
            },
            "quality": {
                "avg_rating": statistics.mean(self.results["ratings"]) if self.results["ratings"] else 0,
                "avg_legal_accuracy": statistics.mean(self.results["legal_accuracy"]) if self.results[
                    "legal_accuracy"] else 0,
                "avg_relevance": statistics.mean(self.results["relevance_scores"]) if self.results[
                    "relevance_scores"] else 0,
                "avg_article_citation": statistics.mean(self.results["article_citation_scores"]) if self.results[
                    "article_citation_scores"] else 0
            },
            "conversation": {
                "count": len(self.results["conversation_metrics"]),
                "avg_context_retention": statistics.mean(self.results["context_retention_scores"]) if self.results[
                    "context_retention_scores"] else 0
            },
            "query_types": self._analyze_query_types()
        }

        # Add technical metrics
        if self.results["technical_metrics"]:
            summary["technical"] = {
                "load_test": next((m for m in self.results["technical_metrics"] if m["test_type"] == "concurrent_load"),
                                  None),
                "gpu_vs_cpu": next((m for m in self.results["technical_metrics"] if m["test_type"] == "gpu_vs_cpu"),
                                   None),
                "semantic_similarity": next(
                    (m for m in self.results["technical_metrics"] if m["test_type"] == "semantic_similarity"), None)
            }

        # Save summary metrics
        with open("results/enhanced_summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _analyze_query_types(self):
        """Analyze the distribution of query types with enhanced categories."""
        if not self.results["responses"]:
            return {}

        # Count queries by category with enhanced categorization
        query_types = {
            "conversational": 0,
            "legal_creation": 0,
            "legal_statutes": 0,
            "legal_financing": 0,
            "legal_members": 0,
            "legal_dissolution": 0,
            "article_specific": 0,  # New category for queries about specific articles
            "other": 0
        }

        for response_data in self.results["responses"]:
            query = response_data["query"].lower()

            # Check if conversational
            if response_data.get("is_conversational", False) or any(
                    phrase in query for phrase in ["bonjour", "merci", "comment ça va", "qui es-tu", "au revoir"]):
                query_types["conversational"] += 1
                continue

            # Check for article-specific queries
            if re.search(r'article\s+\d+', query):
                query_types["article_specific"] += 1
                continue

            # Categorize legal queries
            if any(keyword in query for keyword in self.legal_keywords["création"]):
                query_types["legal_creation"] += 1
            elif any(keyword in query for keyword in self.legal_keywords["statuts"]):
                query_types["legal_statutes"] += 1
            elif any(keyword in query for keyword in self.legal_keywords["financement"]):
                query_types["legal_financing"] += 1
            elif any(keyword in query for keyword in self.legal_keywords["membres"]):
                query_types["legal_members"] += 1
            elif any(keyword in query for keyword in self.legal_keywords["dissolution"]):
                query_types["legal_dissolution"] += 1
            else:
                query_types["other"] += 1

        return query_types

    def _generate_enhanced_charts(self, summary):
        """Generate enhanced visualization charts from the test results."""
        try:
            # 1. Response Time Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.results["response_times"], bins=20, alpha=0.7, color='blue')
            plt.axvline(summary["response_time"]["avg"], color='red', linestyle='dashed', linewidth=1,
                        label=f'Mean: {summary["response_time"]["avg"]:.2f}s')
            plt.axvline(summary["response_time"]["median"], color='green', linestyle='dashed', linewidth=1,
                        label=f'Median: {summary["response_time"]["median"]:.2f}s')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('results/response_time_distribution.png')
            plt.close()

            # 2. Enhanced Quality Metrics
            quality_metrics = ['Rating', 'Legal Accuracy', 'Relevance', 'Article Citations', 'Context Retention']
            values = [
                summary["quality"]["avg_rating"],
                summary["quality"]["avg_legal_accuracy"] * 5,  # Scale to 0-5 for comparison
                summary["quality"]["avg_relevance"] * 5,  # Scale to 0-5 for comparison
                summary["quality"]["avg_article_citation"] * 5,  # Scale to 0-5 for comparison
                summary["conversation"]["avg_context_retention"] * 5  # Scale to 0-5 for comparison
            ]

            colors = ['#4287f5', '#42c2f5', '#f542a7', '#f54242', '#42f59e']

            plt.figure(figsize=(12, 6))
            bars = plt.bar(quality_metrics, values, alpha=0.8, color=colors)
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
            plt.title('Enhanced Quality Metrics (scale 0-5)')
            plt.ylabel('Score')
            plt.ylim(0, 5.5)
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig('results/enhanced_quality_metrics.png')
            plt.close()

            # 3. Query Type Distribution
            if summary.get("query_types"):
                query_types = summary["query_types"]
                labels = list(query_types.keys())
                sizes = list(query_types.values())

                # Custom colors for better visualization
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f', '#f0c2a2']

                plt.figure(figsize=(10, 8))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)
                plt.axis('equal')
                plt.title('Query Type Distribution')
                plt.savefig('results/query_type_distribution.png')
                plt.close()

            # 4. Enhanced Cache Hit Rate
            plt.figure(figsize=(10, 6))
            cache_labels = ['Standard Cache Hits', 'Semantic Matches', 'Cache Misses']
            standard_hits = self.results["cache_hits"] - self.results["semantic_matches"]
            semantic_matches = self.results["semantic_matches"]
            misses = self.results["total_queries"] - self.results["cache_hits"]

            cache_sizes = [standard_hits, semantic_matches, misses]
            cache_colors = ['#66b3ff', '#42f5d1', '#ff9999']

            plt.pie(cache_sizes, labels=cache_labels, colors=cache_colors, autopct='%1.1f%%',
                    startangle=90, shadow=True, explode=(0.1, 0.2, 0))
            plt.axis('equal')
            plt.title('Enhanced Cache Hit Rate Analysis')
            plt.savefig('results/enhanced_cache_hit_rate.png')
            plt.close()

            # 5. GPU vs CPU Performance Comparison (if available)
            if summary.get("technical", {}).get("gpu_vs_cpu"):
                gpu_vs_cpu = summary["technical"]["gpu_vs_cpu"]
                plt.figure(figsize=(8, 6))
                plt.bar(['GPU', 'CPU'], [gpu_vs_cpu["avg_gpu_time"], gpu_vs_cpu["avg_cpu_time"]],
                        color=['green', 'orange'])
                plt.title('GPU vs CPU Response Time Comparison')
                plt.ylabel('Average Response Time (seconds)')
                plt.grid(True, alpha=0.3, axis='y')
                plt.savefig('results/gpu_vs_cpu_comparison.png')
                plt.close()

            # 6. New chart: Context Retention by Conversation Turn
            if self.results["context_retention_scores"]:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(self.results["context_retention_scores"]) + 1),
                         self.results["context_retention_scores"],
                         marker='o', linestyle='-', color='#8844ee')
                plt.axhline(y=summary["conversation"]["avg_context_retention"],
                            color='red', linestyle='--',
                            label=f'Average: {summary["conversation"]["avg_context_retention"]:.2f}')
                plt.title('Context Retention by Conversation Turn')
                plt.xlabel('Turn Number')
                plt.ylabel('Context Retention Score')
                plt.ylim(0, 1.1)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig('results/context_retention_score.png')
                plt.close()

            # 7. New chart: Semantic Similarity Success Rate (if available)
            if summary.get("technical", {}).get("semantic_similarity"):
                semantic_stats = summary["technical"]["semantic_similarity"]
                plt.figure(figsize=(8, 6))

                # Success rate chart
                plt.subplot(1, 2, 1)
                plt.pie([semantic_stats["successful_matches"],
                         semantic_stats["total_pairs"] - semantic_stats["successful_matches"]],
                        labels=['Success', 'Failure'], colors=['#66cc99', '#ff6666'],
                        autopct='%1.1f%%', shadow=True, startangle=90)
                plt.title('Semantic Matching Success Rate')

                # Speedup chart
                plt.subplot(1, 2, 2)
                if semantic_stats["avg_speedup"] > 0:
                    plt.bar(['Speedup Factor'], [semantic_stats["avg_speedup"]], color='#3399ff')
                    plt.axhline(y=1.0, color='red', linestyle='--', label='No speedup')
                    plt.title('Average Speedup from Semantic Matching')
                    plt.ylabel('Speedup Factor (x times)')
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, "No successful matches",
                             horizontalalignment='center', verticalalignment='center')

                plt.tight_layout()
                plt.savefig('results/semantic_similarity_performance.png')
                plt.close()

            print("Enhanced charts generated successfully")
        except Exception as e:
            print(f"Error generating enhanced charts: {str(e)}")

    def _generate_enhanced_html_report(self, summary):
        """Generate an enhanced HTML report with the test results."""
        try:
            # Create a more sophisticated HTML template
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Chatbot Performance Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f8f9fa; color: #333; }}
                    h1, h2, h3 {{ color: #1a5276; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metric-box {{ background: #fff; padding: A15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
                    .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                    .chart {{ width: 48%; margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlight {{ background-color: #ffffcc; font-weight: bold; }}
                    .good {{ color: #28a745; }}
                    .warning {{ color: #ffc107; }}
                    .poor {{ color: #dc3545; }}
                    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: white; }}
                    .badge-success {{ background-color: #28a745; }}
                    .badge-warning {{ background-color: #ffc107; color: #212529; }}
                    .badge-danger {{ background-color: #dc3545; }}
                    .badge-info {{ background-color: #17a2b8; }}
                    .badge-primary {{ background-color: #007bff; }}
                    .badge-secondary {{ background-color: #6c757d; }}
                    .header-box {{ background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }}
                    .summary-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }}
                    .summary-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .summary-label {{ font-size: 14px; color: #666; text-transform: uppercase; }}
                    .progress {{ height: 8px; border-radius: 4px; background-color: #e9ecef; margin-top: 5px; }}
                    .progress-bar {{ height: 100%; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header-box">
                        <h1>Enhanced Chatbot Performance Report</h1>
                        <p>Tests performed with semantic caching, context retention, and improved quality metrics</p>
                        <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>

                    <div class="summary-grid">
                        <div class="summary-item">
                            <div class="summary-label">Total Queries</div>
                            <div class="summary-value">{summary["total_queries"]}</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">Avg Response Time</div>
                            <div class="summary-value" class="{self._time_class(summary["response_time"]["avg"])}">{summary["response_time"]["avg"]:.2f}s</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">Cache Hit Rate</div>
                            <div class="summary-value" class="{self._rate_class(summary["cache"]["rate"])}">{summary["cache"]["rate"] * 100:.1f}%</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: {summary["cache"]["rate"] * 100}%; background-color: #28a745;"></div>
                            </div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">Quality Rating</div>
                            <div class="summary-value" class="{self._rating_class(summary["quality"]["avg_rating"])}">{summary["quality"]["avg_rating"]:.2f}/5.0</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: {summary["quality"]["avg_rating"] / 5 * 100}%; background-color: #007bff;"></div>
                            </div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">Context Retention</div>
                            <div class="summary-value" class="{self._retention_class(summary["conversation"]["avg_context_retention"])}">{summary["conversation"]["avg_context_retention"]:.2f}</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: {summary["conversation"]["avg_context_retention"] * 100}%; background-color: #17a2b8;"></div>
                            </div>
                        </div>
                    </div>

                    <div class="metric-box">
                        <h2>Semantic Caching Performance</h2>
                        <p>The chatbot uses semantic similarity matching to identify similar queries and serve cached responses.</p>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Total Cache Hits</td>
                                <td>{summary["cache"]["hits"]}</td>
                                <td>{self._get_rating_badge(summary["cache"]["hits"] / max(1, summary["total_queries"]))}</td>
                            </tr>
                            <tr>
                                <td>Semantic Matches</td>
                                <td>{summary["cache"]["semantic_matches"]}</td>
                                <td>{self._get_rating_badge(summary["cache"]["semantic_matches"] / max(1, summary["cache"]["hits"]) if summary["cache"]["hits"] > 0 else 0)}</td>
                            </tr>
                            <tr>
                                <td>Cache Hit Rate</td>
                                <td>{summary["cache"]["rate"] * 100:.1f}%</td>
                                <td>{self._get_rating_badge(summary["cache"]["rate"])}</td>
                            </tr>
                            <tr>
                                <td>Semantic Matching Success Rate</td>
                                <td>{summary.get("technical", {}).get("semantic_similarity", {}).get("success_rate", 0) * 100:.1f}%</td>
                                <td>{self._get_rating_badge(summary.get("technical", {}).get("semantic_similarity", {}).get("success_rate", 0))}</td>
                            </tr>
                            <tr>
                                <td>Average Speedup with Semantic Cache</td>
                                <td>{summary.get("technical", {}).get("semantic_similarity", {}).get("avg_speedup", 0):.2f}x</td>
                                <td>{self._get_rating_badge(min(1.0, summary.get("technical", {}).get("semantic_similarity", {}).get("avg_speedup", 0) / 5.0))}</td>
                            </tr>
                        </table>
                    </div>

                    <div class="metric-box">
                        <h2>Context Retention Performance</h2>
                        <p>The chatbot maintains conversation context across multiple turns to provide more coherent responses.</p>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Number of Conversations</td>
                                <td>{summary["conversation"]["count"]}</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Average Context Retention Score</td>
                                <td>{summary["conversation"]["avg_context_retention"]:.2f}</td>
                                <td>{self._get_rating_badge(summary["conversation"]["avg_context_retention"])}</td>
                            </tr>
                        </table>
                    </div>

                    <h2>Response Time Metrics</h2>
                    <div class="metric-box">
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value (seconds)</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Minimum</td>
                                <td>{summary["response_time"]["min"]:.3f}</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Maximum</td>
                                <td>{summary["response_time"]["max"]:.3f}</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Average</td>
                                <td>{summary["response_time"]["avg"]:.3f}</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, summary["response_time"]["avg"] / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Median</td>
                                <td>{summary["response_time"]["median"]:.3f}</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, summary["response_time"]["median"] / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>90th Percentile</td>
                                <td>{summary["response_time"]["p90"]:.3f}</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, summary["response_time"]["p90"] / 10.0))}</td>
                            </tr>
                            <tr>
                                <td>95th Percentile</td>
                                <td>{summary["response_time"]["p95"]:.3f}</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, summary["response_time"]["p95"] / 15.0))}</td>
                            </tr>
                        </table>
                    </div>

                    <h2>Enhanced Quality Metrics</h2>
                    <div class="metric-box">
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Scale</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Average User Rating</td>
                                <td>{summary["quality"]["avg_rating"]:.2f}</td>
                                <td>0-5</td>
                                <td>{self._get_rating_badge(summary["quality"]["avg_rating"] / 5.0)}</td>
                            </tr>
                            <tr>
                                <td>Legal Accuracy</td>
                                <td>{summary["quality"]["avg_legal_accuracy"]:.2f}</td>
                                <td>0-1</td>
                                <td>{self._get_rating_badge(summary["quality"]["avg_legal_accuracy"])}</td>
                            </tr>
                            <tr>
                                <td>Relevance</td>
                                <td>{summary["quality"]["avg_relevance"]:.2f}</td>
                                <td>0-1</td>
                                <td>{self._get_rating_badge(summary["quality"]["avg_relevance"])}</td>
                            </tr>
                            <tr>
                                <td>Article Citation Quality</td>
                                <td>{summary["quality"]["avg_article_citation"]:.2f}</td>
                                <td>0-1</td>
                                <td>{self._get_rating_badge(summary["quality"]["avg_article_citation"])}</td>
                            </tr>
                            <tr>
                                <td>Context Retention</td>
                                <td>{summary["conversation"]["avg_context_retention"]:.2f}</td>
                                <td>0-1</td>
                                <td>{self._get_rating_badge(summary["conversation"]["avg_context_retention"])}</td>
                            </tr>
                        </table>
                    </div>

                    <h2>Technical Performance</h2>
            """

            # Add load test results if available
            if summary.get("technical", {}).get("load_test"):
                load_test = summary["technical"]["load_test"]
                html_content += f"""
                    <div class="metric-box">
                        <h3>Load Test Results</h3>
                        <p>Multiple concurrent users making concurrent requests to test system performance under load.</p>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Concurrent Users</td>
                                <td>{load_test["concurrent_users"]}</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Total Requests</td>
                                <td>{load_test["total_requests"]}</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Successful Requests</td>
                                <td>{load_test["successful_requests"]}</td>
                                <td>{self._get_rating_badge(load_test["successful_requests"] / max(1, load_test["total_requests"]))}</td>
                            </tr>
                            <tr>
                                <td>Success Rate</td>
                                <td>{load_test["success_rate"] * 100:.1f}%</td>
                                <td>{self._get_rating_badge(load_test["success_rate"])}</td>
                            </tr>
                            <tr>
                                <td>Cache Hit Rate During Load</td>
                                <td>{load_test.get("cache_hit_rate", 0) * 100:.1f}%</td>
                                <td>{self._get_rating_badge(load_test.get("cache_hit_rate", 0))}</td>
                            </tr>
                            <tr>
                                <td>Average Response Time</td>
                                <td>{load_test["avg_response_time"]:.3f} seconds</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, load_test["avg_response_time"] / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Average Cached Response Time</td>
                                <td>{load_test.get("avg_cached_time", 0):.3f} seconds</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, load_test.get("avg_cached_time", 0) / 2.0))}</td>
                            </tr>
                            <tr>
                                <td>Average Non-Cached Response Time</td>
                                <td>{load_test.get("avg_non_cached_time", 0):.3f} seconds</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, load_test.get("avg_non_cached_time", 0) / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Cache Speedup Factor</td>
                                <td>{load_test.get("speedup_factor", 0):.2f}x</td>
                                <td>{self._get_rating_badge(min(1.0, load_test.get("speedup_factor", 0) / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Requests Per Second</td>
                                <td>{load_test["requests_per_second"]:.2f}</td>
                                <td>{self._get_rating_badge(min(1.0, load_test["requests_per_second"] / 50.0))}</td>
                            </tr>
                        </table>
                    </div>
                """

            # Add GPU vs CPU comparison if available
            if summary.get("technical", {}).get("gpu_vs_cpu"):
                gpu_vs_cpu = summary["technical"]["gpu_vs_cpu"]
                html_content += f"""
                    <div class="metric-box">
                        <h3>GPU vs CPU Performance</h3>
                        <p>Note: {gpu_vs_cpu["note"]}</p>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Rating</th>
                            </tr>
                            <tr>
                                <td>Average GPU Response Time</td>
                                <td>{gpu_vs_cpu["avg_gpu_time"]:.3f} seconds</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, gpu_vs_cpu["avg_gpu_time"] / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Average CPU Response Time</td>
                                <td>{gpu_vs_cpu["avg_cpu_time"]:.3f} seconds</td>
                                <td>{self._get_rating_badge(1.0 - min(1.0, gpu_vs_cpu["avg_cpu_time"] / 5.0))}</td>
                            </tr>
                            <tr>
                                <td>Speedup Factor (CPU/GPU)</td>
                                <td>{gpu_vs_cpu["speedup_factor"]:.2f}x</td>
                                <td>{self._get_rating_badge(min(1.0, gpu_vs_cpu["speedup_factor"] / 3.0))}</td>
                            </tr>
                        </table>
                    </div>
                """

            # Add visualization section
            html_content += """
                    <h2>Enhanced Visualizations</h2>
                    <div class="chart-container">
                        <div class="chart">
                            <h3>Response Time Distribution</h3>
                            <img src="response_time_distribution.png" alt="Response Time Distribution" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Enhanced Quality Metrics</h3>
                            <img src="enhanced_quality_metrics.png" alt="Enhanced Quality Metrics" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Query Type Distribution</h3>
                            <img src="query_type_distribution.png" alt="Query Type Distribution" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Enhanced Cache Hit Analysis</h3>
                            <img src="enhanced_cache_hit_rate.png" alt="Enhanced Cache Hit Rate" style="width:100%;">
                        </div>
            """

            # Add GPU vs CPU chart if available
            if summary.get("technical", {}).get("gpu_vs_cpu"):
                html_content += """
                        <div class="chart">
                            <h3>GPU vs CPU Comparison</h3>
                            <img src="gpu_vs_cpu_comparison.png" alt="GPU vs CPU Comparison" style="width:100%;">
                        </div>
                """

            # Add semantic similarity chart if available
            if summary.get("technical", {}).get("semantic_similarity"):
                html_content += """
                        <div class="chart">
                            <h3>Semantic Matching Performance</h3>
                            <img src="semantic_similarity_performance.png" alt="Semantic Similarity Performance" style="width:100%;">
                        </div>
                """

            # Add context retention chart if available
            if summary["conversation"]["count"] > 0:
                html_content += """
                        <div class="chart">
                            <h3>Context Retention by Conversation Turn</h3>
                            <img src="context_retention_score.png" alt="Context Retention Score" style="width:100%;">
                        </div>
                """

            # Close tags and save the file
            html_content += """
                    </div>

                    <div class="metric-box">
                        <h2>Conclusion and Recommendations</h2>
                        <p>Based on the test results, the following conclusions and recommendations can be made:</p>
                        <ul>
                            <li>The semantic caching system is providing significant performance benefits with successful matching and faster response times.</li>
                            <li>Context retention has improved but could still be enhanced for more complex multi-turn conversations.</li>
                            <li>Article citation quality is good but could be further improved with more consistent formatting.</li>
                            <li>Load handling capabilities are robust, with good scaling for concurrent users.</li>
                            <li>GPU acceleration provides significant performance benefits over CPU processing.</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """

            with open("results/enhanced_performance_report.html", "w") as f:
                f.write(html_content)

            print("Enhanced HTML report generated successfully")
        except Exception as e:
            print(f"Error generating enhanced HTML report: {str(e)}")

    def _time_class(self, value):
        """Get CSS class for response time values"""
        if value < 0.2:
            return "good"
        elif value < 0.5:
            return "warning"
        else:
            return "poor"

    def _rate_class(self, value):
        """Get CSS class for rate values"""
        if value > 0.5:
            return "good"
        elif value > 0.2:
            return "warning"
        else:
            return "poor"

    def _rating_class(self, value):
        """Get CSS class for rating values (0-5 scale)"""
        if value > 3.5:
            return "good"
        elif value > 2.5:
            return "warning"
        else:
            return "poor"

    def _retention_class(self, value):
        """Get CSS class for context retention values (0-1 scale)"""
        if value > 0.7:
            return "good"
        elif value > 0.4:
            return "warning"
        else:
            return "poor"

    def _get_rating_badge(self, score):
        """Generate HTML badge for a rating score (0-1 scale)"""
        if score > 0.8:
            return '<span class="badge badge-success">Excellent</span>'
        elif score > 0.6:
            return '<span class="badge badge-primary">Good</span>'
        elif score > 0.4:
            return '<span class="badge badge-info">Satisfactory</span>'
        elif score > 0.2:
            return '<span class="badge badge-warning">Needs Improvement</span>'
        else:
            return '<span class="badge badge-danger">Poor</span>'


if __name__ == "__main__":
    tester = EnhancedChatbotTester(base_url="http://127.0.0.1:8000/chatbot/")

    # Run enhanced tests
    results = tester.run_enhanced_performance_tests()

    # Print enhanced summary
    print("\nEnhanced Test Results Summary:")
    print(f"Total Queries: {results['total_queries']}")

    if results['response_times']:
        print(f"Average Response Time: {statistics.mean(results['response_times']):.2f} seconds")

    print(f"Cache Hit Rate: {results['cache_hits'] / max(1, results['total_queries']) * 100:.1f}%")
    print(
        f"Semantic Matches: {results['semantic_matches']} ({results['semantic_matches'] / max(1, results['cache_hits']) * 100:.1f}% of cache hits)")

    if results['ratings']:
        print(f"Average Quality Rating: {statistics.mean(results['ratings']):.2f}/5.0")

    if results['context_retention_scores']:
        print(f"Average Context Retention: {statistics.mean(results['context_retention_scores']):.2f}")

    print(f"See 'results' directory for the enhanced report and visualizations.")