"""
Knowledge Base Indexing Script
Loads KB JSON, generates embeddings using Azure OpenAI, and indexes into Pinecone
"""

import json
import os
from typing import List, Dict
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-kb")

KB_FILE_PATH = "self_critique_loop_dataset.json"


class KnowledgeBaseIndexer:
    """Handles KB loading, embedding generation, and vector DB indexing"""
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.azure_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = None
        
    def load_kb_data(self, file_path: str) -> List[Dict]:
        """Load knowledge base from JSON file"""
        print(f"Loading KB from {file_path}...")
        with open(file_path, 'r') as f:
            kb_data = json.load(f)
        print(f"Loaded {len(kb_data)} KB entries")
        return kb_data
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI text-embedding-3-small"""
        response = self.azure_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    
    def create_pinecone_index(self, dimension: int = 1536):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            print("Index created successfully")
        else:
            print(f"Index {PINECONE_INDEX_NAME} already exists")
        
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
    
    def prepare_document_text(self, doc: Dict) -> str:
        """Prepare text for embedding from KB document"""
        return f"{doc['question']} {doc['answer_snippet']}"
    
    def index_documents(self, kb_data: List[Dict], batch_size: int = 100):
        """Generate embeddings and index documents into Pinecone"""
        print(f"\nIndexing {len(kb_data)} documents...")
        
        vectors_to_upsert = []
        
        for doc in tqdm(kb_data, desc="Generating embeddings"):
            # Prepare text and generate embedding
            text = self.prepare_document_text(doc)
            embedding = self.generate_embedding(text)
            
            # Prepare metadata
            metadata = {
                "doc_id": doc["doc_id"],
                "question": doc["question"],
                "answer_snippet": doc["answer_snippet"],
                "source": doc["source"],
                "confidence_indicator": doc["confidence_indicator"],
                "last_updated": doc["last_updated"]
            }
            
            # Add to batch
            vectors_to_upsert.append({
                "id": doc["doc_id"],
                "values": embedding,
                "metadata": metadata
            })
            
            # Upsert batch when reaching batch_size
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"\n✓ Successfully indexed {len(kb_data)} documents")
        
        # Print index stats
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")
    
    def verify_index(self, sample_query: str = "What are best practices for caching?"):
        """Verify indexing by performing a sample query"""
        print(f"\nVerifying index with sample query: '{sample_query}'")
        
        query_embedding = self.generate_embedding(sample_query)
        results = self.index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        print(f"\nTop 3 results:")
        for i, match in enumerate(results.matches, 1):
            print(f"\n{i}. [{match.id}] Score: {match.score:.4f}")
            print(f"   Question: {match.metadata.get('question')}")
            print(f"   Source: {match.metadata.get('source')}")


def main():
    """Main indexing pipeline"""
    print("=" * 80)
    print("KNOWLEDGE BASE INDEXING PIPELINE")
    print("=" * 80)
    
    # Validate environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "PINECONE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Initialize indexer
    indexer = KnowledgeBaseIndexer()
    
    # Load KB data
    kb_data = indexer.load_kb_data(KB_FILE_PATH)
    
    # Create Pinecone index
    indexer.create_pinecone_index(dimension=1536)
    
    # Index documents
    indexer.index_documents(kb_data)
    
    # Verify indexing
    indexer.verify_index()
    
    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
