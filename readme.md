1. Install dependencies:
pip install -r requirements.txt

2. Set environment variables:
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export AZURE_GPT4_DEPLOYMENT="gpt-4-mini"
export PINECONE_API_KEY="your-pinecone-key"
export GEMINI_API_KEY="your-gemini-key"

3. Index the knowledge base:
python index_kb.py

4. Run the Agentic RAG system:
python agentic_rag_azure.py

5. View MLflow results:
mlflow ui

