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


class ChatbotTester:
    """A comprehensive testing suite for the association law chatbot."""

    def __init__(self, base_url="http://localhost:8000/chatbot/"):
        """Initialize the tester with the base URL for the API."""
        self.base_url = base_url
        self.direct_chat_url = f"{base_url}direct-chat/"
        self.conversations_url = f"{base_url}conversations/"
        self.results = {
            "response_times": [],
            "ratings": [],
            "cache_hits": 0,
            "total_queries": 0,
            "conversation_metrics": [],
            "technical_metrics": [],
            "legal_accuracy": [],
            "relevance_scores": [],
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
                     "Quelles sont les différences avec les associations locales?")
                ],
                "edge_cases": [
                    "What happens if I ask in English instead of French?",
                    "Je voudrais savoir quelque chose qui n'est pas lié aux associations",
                    "Est-ce que les associations peuvent faire du commerce?"
                ],
                "processing_test": [
                    "Peux-tu me donner une réponse détaillée sur la création, le financement, la gestion et la dissolution d'une association?"
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

    def test_direct_chat(self, query):
        """Test a single query using the direct chat endpoint."""
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

            # Check for cache hit
            if result.get("cached", False):
                self.results["cache_hits"] += 1

            # Store response for further analysis
            self.results["responses"].append({
                "query": query,
                "response": result.get("response", ""),
                "response_time": response_time,
                "relevant_chunks": result.get("relevant_chunks", []),
                "is_conversational": result.get("is_conversational", False)
            })

            # Evaluate response quality
            self._evaluate_response_quality(query, result.get("response", ""))

            return result, response_time
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None, response_time

    def test_conversation(self, prompts):
        """Test a multi-turn conversation with the chatbot."""
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

                # For follow-up questions, evaluate context retention
                if i > 0:
                    context_score = self._evaluate_context_retention(
                        conversation_metrics["turns"][i - 1]["prompt"],
                        conversation_metrics["turns"][i - 1]["response"],
                        prompt,
                        result.get("content", "")
                    )
                    conversation_metrics["context_retention_score"] = context_score

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
        """Evaluate the quality of a response."""
        # 1. Evaluate legal accuracy
        accuracy_score = self._evaluate_legal_accuracy(query, response)
        self.results["legal_accuracy"].append(accuracy_score)

        # 2. Evaluate relevance
        relevance_score = self._evaluate_relevance(query, response)
        self.results["relevance_scores"].append(relevance_score)

        # 3. Simulate user rating (in a real scenario, this would come from users)
        # Here we use a weighted combination of accuracy and relevance
        simulated_rating = min(5, (accuracy_score * 0.6 + relevance_score * 0.4) * 5)
        self.results["ratings"].append(simulated_rating)

    def _evaluate_legal_accuracy(self, query, response):
        """Evaluate the legal accuracy of a response (0-1 scale)."""
        # Basic implementation - checks for presence of article numbers
        # A more sophisticated implementation would validate actual content
        query_lower = query.lower()
        accuracy_score = 0.5  # Default middle score

        # Check if response mentions appropriate articles based on query topic
        for topic, keywords in self.legal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Get the expected articles for this topic
                expected_articles = self.legal_ground_truth.get(topic, "")

                # Check if these article numbers are mentioned in the response
                article_mentions = 0
                for article_num in expected_articles.split():
                    if article_num.isdigit() and article_num in response:
                        article_mentions += 1

                if article_mentions > 0:
                    # Calculate score based on how many expected articles were mentioned
                    accuracy_score = min(1.0, 0.5 + article_mentions * 0.1)
                break

        # Better score if response mentions "décret-loi n° 2011-88"
        if "décret-loi n° 2011-88" in response:
            accuracy_score = min(1.0, accuracy_score + 0.2)

        return accuracy_score

    def _evaluate_relevance(self, query, response):
        """Evaluate how relevant the response is to the query (0-1 scale)."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Basic relevance: word overlap
        overlap = len(query_words.intersection(response_words)) / max(1, len(query_words))

        # Check if response contains legal keywords relevant to the query
        query_lower = query.lower()
        keyword_relevance = 0

        for topic, keywords in self.legal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Count relevant keywords in response
                response_keyword_count = sum(1 for keyword in keywords if keyword in response.lower())
                keyword_relevance = min(1.0, response_keyword_count / max(3, len(keywords)))
                break

        # Combined relevance score (word overlap + keyword relevance)
        relevance_score = (overlap * 0.4) + (keyword_relevance * 0.6)
        return relevance_score

    def _evaluate_context_retention(self, prev_prompt, prev_response, curr_prompt, curr_response):
        """Evaluate how well the chatbot maintains context (0-1 scale)."""
        # Simple implementation: check if current response references content from previous exchange
        prev_keywords = set([word.lower() for word in prev_prompt.split() + prev_response.split()
                             if len(word) > 4 and word.lower() not in {"cette", "pour", "dans", "avec"}])

        # Count keywords from previous exchange appearing in current response
        matching_keywords = sum(1 for word in prev_keywords if word in curr_response.lower())

        # Calculate score based on keyword matches
        context_score = min(1.0, matching_keywords / max(5, len(prev_keywords) * 0.3))
        return context_score

    def run_performance_tests(self):
        """Run comprehensive performance tests on the chatbot."""
        print("Starting performance tests...")

        # 1. Test conversational queries
        print("\nTesting conversational queries...")
        for prompt in tqdm(self.test_prompts["conversational"]):
            self.test_direct_chat(prompt)

        # 2. Test legal domain queries
        print("\nTesting legal domain queries...")
        for prompt in tqdm(self.test_prompts["legal_domain"]):
            self.test_direct_chat(prompt)

        # 3. Test context retention through conversations
        print("\nTesting multi-turn conversations...")
        for prompt_pair in tqdm(self.test_prompts["context_testing"]):
            self.test_conversation(prompt_pair)

        # 4. Test edge cases
        print("\nTesting edge cases...")
        for prompt in tqdm(self.test_prompts["edge_cases"]):
            self.test_direct_chat(prompt)

        # 5. Test processing capability
        print("\nTesting processing capacity...")
        for prompt in tqdm(self.test_prompts["processing_test"]):
            self.test_direct_chat(prompt)

        # 6. Test load handling (concurrent requests)
        print("\nTesting load handling...")
        self._test_concurrent_load()

        # 7. Test GPU vs CPU performance (if applicable)
        self._test_cpu_vs_gpu_performance()

        # Generate and save report
        self.generate_report()

        print("\nPerformance tests completed!")
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
                        "success": result is not None
                    })
                except Exception as e:
                    print(f"Error in user {user_id}, query {i + 1}: {str(e)}")
                    user_results.append({
                        "user_id": user_id,
                        "query_num": i + 1,
                        "query": query,
                        "error": str(e),
                        "success": False
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
        avg_response_time = statistics.mean([r.get("response_time", 0) for r in all_results if "response_time" in r])

        # Save load test results
        self.results["technical_metrics"].append({
            "test_type": "concurrent_load",
            "concurrent_users": num_concurrent,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / max(1, total_requests),
            "avg_response_time": avg_response_time,
            "load_test_duration": load_test_duration,
            "requests_per_second": total_requests / max(1, load_test_duration)
        })

        return self.results["technical_metrics"][-1]

    def _test_cpu_vs_gpu_performance(self, sample_size=3):
        """Test performance difference between CPU and GPU (if applicable).
        This is a simulated test since we can't directly control hardware from the script.
        In a real scenario, you would need server-side configuration to switch between CPU/GPU."""

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

    def generate_report(self):
        """Generate a comprehensive report of the test results."""
        print("\nGenerating performance report...")

        # Create a results directory if it doesn't exist
        import os
        if not os.path.exists("results"):
            os.makedirs("results")

        # 1. Save raw results data
        with open("results/raw_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 2. Calculate summary metrics
        summary = self._calculate_summary_metrics()

        # 3. Generate charts
        self._generate_charts(summary)

        # 4. Generate HTML report
        self._generate_html_report(summary)

        print(f"Report generated. See 'results' directory for output files.")
        return summary

    def _calculate_summary_metrics(self):
        """Calculate summary metrics from the test results."""
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
                "rate": self.results["cache_hits"] / max(1, self.results["total_queries"])
            },
            "quality": {
                "avg_rating": statistics.mean(self.results["ratings"]) if self.results["ratings"] else 0,
                "avg_legal_accuracy": statistics.mean(self.results["legal_accuracy"]) if self.results[
                    "legal_accuracy"] else 0,
                "avg_relevance": statistics.mean(self.results["relevance_scores"]) if self.results[
                    "relevance_scores"] else 0
            },
            "conversation": {
                "count": len(self.results["conversation_metrics"]),
                "avg_context_retention": statistics.mean(
                    [cm["context_retention_score"] for cm in self.results["conversation_metrics"]])
                if self.results["conversation_metrics"] else 0
            },
            "query_types": self._analyze_query_types()
        }

        # Add technical metrics
        if self.results["technical_metrics"]:
            summary["technical"] = {
                "load_test": next((m for m in self.results["technical_metrics"] if m["test_type"] == "concurrent_load"),
                                  None),
                "gpu_vs_cpu": next((m for m in self.results["technical_metrics"] if m["test_type"] == "gpu_vs_cpu"),
                                   None)
            }

        # Save summary metrics
        with open("results/summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _analyze_query_types(self):
        """Analyze the distribution of query types."""
        if not self.results["responses"]:
            return {}

        # Count queries by category
        query_types = {
            "conversational": 0,
            "legal_creation": 0,
            "legal_statutes": 0,
            "legal_financing": 0,
            "legal_members": 0,
            "legal_dissolution": 0,
            "other": 0
        }

        for response_data in self.results["responses"]:
            query = response_data["query"].lower()

            # Check if conversational
            if response_data.get("is_conversational", False) or any(
                    phrase in query for phrase in ["bonjour", "merci", "comment ça va", "qui es-tu", "au revoir"]):
                query_types["conversational"] += 1
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

    def _generate_charts(self, summary):
        """Generate visualization charts from the test results."""
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

            # 2. Quality Metrics
            quality_metrics = ['Rating', 'Legal Accuracy', 'Relevance']
            values = [
                summary["quality"]["avg_rating"],
                summary["quality"]["avg_legal_accuracy"] * 5,  # Scale to 0-5 for comparison
                summary["quality"]["avg_relevance"] * 5  # Scale to 0-5 for comparison
            ]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(quality_metrics, values, alpha=0.7)
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
            plt.title('Quality Metrics (scale 0-5)')
            plt.ylabel('Score')
            plt.ylim(0, 5.5)
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig('results/quality_metrics.png')
            plt.close()

            # 3. Query Type Distribution
            if summary.get("query_types"):
                query_types = summary["query_types"]
                labels = list(query_types.keys())
                sizes = list(query_types.values())

                plt.figure(figsize=(10, 8))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
                plt.axis('equal')
                plt.title('Query Type Distribution')
                plt.savefig('results/query_type_distribution.png')
                plt.close()

            # 4. Cache Hit Rate
            plt.figure(figsize=(8, 8))
            labels = ['Cache Hits', 'Cache Misses']
            sizes = [summary["cache"]["hits"], summary["total_queries"] - summary["cache"]["hits"]]
            colors = ['#66b3ff', '#ff9999']
            explode = (0.1, 0)  # explode the 1st slice (Cache Hits)

            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Cache Hit Rate')
            plt.savefig('results/cache_hit_rate.png')
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

            print("Charts generated successfully")
        except Exception as e:
            print(f"Error generating charts: {str(e)}")

    def _generate_html_report(self, summary):
        """Generate an HTML report with the test results."""
        try:
            # Create a basic HTML template
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>Chatbot Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metric-box {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                    .chart {{ width: 48%; margin-bottom: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlight {{ background-color: #ffffcc; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Chatbot Performance Report</h1>
                    <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

                    <div class="metric-box">
                        <h2>Overall Performance Summary</h2>
                        <p>Total Queries: <strong>{summary["total_queries"]}</strong></p>
                        <p>Average Response Time: <strong>{summary["response_time"]["avg"]:.2f} seconds</strong></p>
                        <p>Cache Hit Rate: <strong>{summary["cache"]["rate"] * 100:.1f}%</strong></p>
                        <p>Average Quality Rating: <strong>{summary["quality"]["avg_rating"]:.2f}/5.0</strong></p>
                    </div>

                    <h2>Response Time Metrics</h2>
                    <div class="metric-box">
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value (seconds)</th>
                            </tr>
                            <tr>
                                <td>Minimum</td>
                                <td>{summary["response_time"]["min"]:.3f}</td>
                            </tr>
                            <tr>
                                <td>Maximum</td>
                                <td>{summary["response_time"]["max"]:.3f}</td>
                            </tr>
                            <tr>
                                <td>Average</td>
                                <td>{summary["response_time"]["avg"]:.3f}</td>
                            </tr>
                            <tr>
                                <td>Median</td>
                                <td>{summary["response_time"]["median"]:.3f}</td>
                            </tr>
                            <tr>
                                <td>90th Percentile</td>
                                <td>{summary["response_time"]["p90"]:.3f}</td>
                            </tr>
                            <tr>
                                <td>95th Percentile</td>
                                <td>{summary["response_time"]["p95"]:.3f}</td>
                            </tr>
                        </table>
                    </div>

                    <h2>Quality Metrics</h2>
                    <div class="metric-box">
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Scale</th>
                            </tr>
                            <tr>
                                <td>Average Rating</td>
                                <td>{summary["quality"]["avg_rating"]:.2f}</td>
                                <td>0-5</td>
                            </tr>
                            <tr>
                                <td>Legal Accuracy</td>
                                <td>{summary["quality"]["avg_legal_accuracy"]:.2f}</td>
                                <td>0-1</td>
                            </tr>
                            <tr>
                                <td>Relevance</td>
                                <td>{summary["quality"]["avg_relevance"]:.2f}</td>
                                <td>0-1</td>
                            </tr>
                            <tr>
                                <td>Context Retention (conversations)</td>
                                <td>{summary["conversation"]["avg_context_retention"]:.2f}</td>
                                <td>0-1</td>
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
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Concurrent Users</td>
                                <td>{load_test["concurrent_users"]}</td>
                            </tr>
                            <tr>
                                <td>Total Requests</td>
                                <td>{load_test["total_requests"]}</td>
                            </tr>
                            <tr>
                                <td>Successful Requests</td>
                                <td>{load_test["successful_requests"]}</td>
                            </tr>
                            <tr>
                                <td>Success Rate</td>
                                <td>{load_test["success_rate"] * 100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Average Response Time</td>
                                <td>{load_test["avg_response_time"]:.3f} seconds</td>
                            </tr>
                            <tr>
                                <td>Requests Per Second</td>
                                <td>{load_test["requests_per_second"]:.2f}</td>
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
                            </tr>
                            <tr>
                                <td>Average GPU Response Time</td>
                                <td>{gpu_vs_cpu["avg_gpu_time"]:.3f} seconds</td>
                            </tr>
                            <tr>
                                <td>Average CPU Response Time</td>
                                <td>{gpu_vs_cpu["avg_cpu_time"]:.3f} seconds</td>
                            </tr>
                            <tr>
                                <td>Speedup Factor (CPU/GPU)</td>
                                <td>{gpu_vs_cpu["speedup_factor"]:.2f}x</td>
                            </tr>
                        </table>
                    </div>
                """

            # Add visualization section
            html_content += """
                    <h2>Visualizations</h2>
                    <div class="chart-container">
                        <div class="chart">
                            <h3>Response Time Distribution</h3>
                            <img src="response_time_distribution.png" alt="Response Time Distribution" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Quality Metrics</h3>
                            <img src="quality_metrics.png" alt="Quality Metrics" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Query Type Distribution</h3>
                            <img src="query_type_distribution.png" alt="Query Type Distribution" style="width:100%;">
                        </div>
                        <div class="chart">
                            <h3>Cache Hit Rate</h3>
                            <img src="cache_hit_rate.png" alt="Cache Hit Rate" style="width:100%;">
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

            # Close tags and save the file
            html_content += """
                    </div>
                </div>
            </body>
            </html>
            """

            with open("results/performance_report.html", "w") as f:
                f.write(html_content)

            print("HTML report generated successfully")
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")


if __name__ == "__main__":

    tester = ChatbotTester(base_url="http://127.0.0.1:8000/chatbot/")

    # Run all tests
    results = tester.run_performance_tests()

    # Print summary
    print("\nTest Results Summary:")
    print(f"Total Queries: {results['total_queries']}")
    if results['response_times']:
        print(f"Average Response Time: {statistics.mean(results['response_times']):.2f} seconds")
    print(f"Cache Hit Rate: {results['cache_hits'] / max(1, results['total_queries']) * 100:.1f}%")
    if results['ratings']:
        print(f"Average Rating: {statistics.mean(results['ratings']):.2f}/5.0")
    print(f"See 'results' directory for full report and visualizations.")