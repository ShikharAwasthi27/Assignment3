"""
Agentic RAG System with LangGraph
Implements self-critique loop with Azure GPT-4 mini and MLflow observability
"""

import os
from typing import List, Dict, TypedDict, Literal
from openai import AzureOpenAI
from pinecone import Pinecone
import mlflow
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from datetime import datetime

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_GPT4_DEPLOYMENT = os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-kb")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")


# State definition for LangGraph
class AgentState(TypedDict):
    """State passed between nodes in the graph"""
    query: str
    retrieved_docs: List[Dict]
    initial_answer: str
    critique_result: str
    final_answer: str
    refinement_doc: Dict
    iteration: int


class AgenticRAGSystem:
    """Agentic RAG system with self-critique and refinement"""
    
    def __init__(self):
        # Initialize Azure OpenAI
        self.azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize Gemini for critique
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("agentic_rag_azure")
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with 4 nodes"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("llm_answer", self.llm_answer_node)
        workflow.add_node("self_critique", self.self_critique_node)
        workflow.add_node("refinement", self.refinement_node)
        
        # Define edges
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "llm_answer")
        workflow.add_edge("llm_answer", "self_critique")
        
        # Conditional edge based on critique
        workflow.add_conditional_edges(
            "self_critique",
            self.should_refine,
            {
                "refine": "refinement",
                "complete": END
            }
        )
        
        workflow.add_edge("refinement", END)
        
        return workflow.compile()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI"""
        response = self.azure_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    
    def retriever_node(self, state: AgentState) -> AgentState:
        """Node 1: Retrieve top-5 KB snippets"""
        query = state["query"]
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract documents
        retrieved_docs = []
        for match in results.matches:
            doc = {
                "doc_id": match.id,
                "score": match.score,
                "question": match.metadata.get("question"),
                "answer_snippet": match.metadata.get("answer_snippet"),
                "source": match.metadata.get("source")
            }
            retrieved_docs.append(doc)
        
        mlflow.log_param("num_retrieved_docs", len(retrieved_docs))
        mlflow.log_text(str(retrieved_docs), "retrieved_docs.json")
        
        state["retrieved_docs"] = retrieved_docs
        state["iteration"] = 1
        return state
    
    def llm_answer_node(self, state: AgentState) -> AgentState:
        """Node 2: Generate answer using Azure GPT-4 mini with citations"""
        query = state["query"]
        docs = state["retrieved_docs"]
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"[{doc['doc_id']}] {doc['answer_snippet']}"
            for doc in docs
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge base snippets.

Question: {query}

Knowledge Base Context:
{context}

Instructions:
- Provide a comprehensive answer based on the context above
- Cite sources using [KBxxx] format where relevant
- Be clear and concise
- If the context doesn't fully answer the question, do your best with available information

Answer:"""
        
        # Generate answer with temperature=0
        response = self.azure_client.chat.completions.create(
            model=AZURE_GPT4_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers with citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        mlflow.log_text(answer, f"initial_answer_iter_{state['iteration']}.txt")
        
        state["initial_answer"] = answer
        state["final_answer"] = answer
        return state
    
    def self_critique_node(self, state: AgentState) -> AgentState:
        """Node 3: Critique answer completeness using Gemini"""
        query = state["query"]
        answer = state["initial_answer"]
        
        critique_prompt = f"""You are an expert evaluator. Assess whether the following answer completely and adequately addresses the user's question.

Question: {query}

Answer: {answer}

Evaluation Criteria:
- Does the answer directly address the question?
- Is the information comprehensive enough?
- Are there obvious gaps or missing important details?
- Is the answer clear and well-structured?

Respond with ONLY ONE WORD:
- "COMPLETE" if the answer adequately addresses the question
- "REFINE" if the answer needs additional information or clarification

Your assessment:"""
        
        # Use Gemini for critique
        response = self.gemini_model.generate_content(critique_prompt)
        critique_result = response.text.strip().upper()
        
        # Ensure valid response
        if "COMPLETE" in critique_result:
            critique_result = "COMPLETE"
        elif "REFINE" in critique_result:
            critique_result = "REFINE"
        else:
            critique_result = "COMPLETE"
        
        mlflow.log_param(f"critique_result_iter_{state['iteration']}", critique_result)
        
        state["critique_result"] = critique_result
        return state
    
    def should_refine(self, state: AgentState) -> Literal["refine", "complete"]:
        """Decision function for conditional edge"""
        if state["critique_result"] == "REFINE":
            return "refine"
        return "complete"
    
    def refinement_node(self, state: AgentState) -> AgentState:
        """Node 4: Retrieve 1 additional snippet and regenerate answer"""
        query = state["query"]
        existing_doc_ids = {doc["doc_id"] for doc in state["retrieved_docs"]}
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Retrieve top-10 to find a new document
        results = self.index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        # Find first document not in existing set
        refinement_doc = None
        for match in results.matches:
            if match.id not in existing_doc_ids:
                refinement_doc = {
                    "doc_id": match.id,
                    "score": match.score,
                    "question": match.metadata.get("question"),
                    "answer_snippet": match.metadata.get("answer_snippet"),
                    "source": match.metadata.get("source")
                }
                break
        
        if refinement_doc:
            state["refinement_doc"] = refinement_doc
            
            # Add to retrieved docs
            all_docs = state["retrieved_docs"] + [refinement_doc]
            
            # Regenerate answer with expanded context
            context = "\n\n".join([
                f"[{doc['doc_id']}] {doc['answer_snippet']}"
                for doc in all_docs
            ])
            
            prompt = f"""You are a helpful assistant that answers questions based on the provided knowledge base snippets.

Question: {query}

Knowledge Base Context:
{context}

Instructions:
- Provide a comprehensive answer based on the context above
- Cite sources using [KBxxx] format where relevant
- Be clear and concise
- Incorporate all relevant information from the snippets

Answer:"""
            
            response = self.azure_client.chat.completions.create(
                model=AZURE_GPT4_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers with citations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            refined_answer = response.choices[0].message.content
            
            mlflow.log_text(str(refinement_doc), "refinement_doc.json")
            mlflow.log_text(refined_answer, "refined_answer.txt")
            
            state["final_answer"] = refined_answer
        else:
            mlflow.log_param("refinement_status", "no_new_doc_found")
        
        return state
    
    def run(self, query: str) -> Dict:
        """Execute the agentic RAG workflow"""
        with mlflow.start_run(run_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("query", query)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Initialize state
            initial_state = {
                "query": query,
                "retrieved_docs": [],
                "initial_answer": "",
                "critique_result": "",
                "final_answer": "",
                "refinement_doc": {},
                "iteration": 1
            }
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Log final results
            mlflow.log_text(final_state["final_answer"], "final_answer.txt")
            mlflow.log_metric("total_docs_used", len(final_state["retrieved_docs"]) + (1 if final_state.get("refinement_doc") else 0))
            
            return {
                "query": query,
                "answer": final_state["final_answer"],
                "critique_result": final_state["critique_result"],
                "num_docs_retrieved": len(final_state["retrieved_docs"]),
                "refinement_applied": bool(final_state.get("refinement_doc"))
            }


def main():
    """Main execution with sample queries"""
    print("=" * 80)
    print("AGENTIC RAG SYSTEM - AZURE + LANGGRAPH + MLFLOW")
    print("=" * 80)
    
    # Validate environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "GEMINI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Initialize system
    rag_system = AgenticRAGSystem()
    
    # Sample queries
    sample_queries = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?",
        "How do I version my APIs?",
        "What should I consider for error handling?"
    ]
    
    print("\nRunning sample queries...\n")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 80}")
        
        result = rag_system.run(query)
        
        print(f"\nCritique Result: {result['critique_result']}")
        print(f"Docs Retrieved: {result['num_docs_retrieved']}")
        print(f"Refinement Applied: {result['refinement_applied']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n{'=' * 80}")
    
    print("\n✓ All queries processed successfully")
    print(f"✓ MLflow logs available at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
