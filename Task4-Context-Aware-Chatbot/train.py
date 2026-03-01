import wikipedia
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

topics = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Neural network",
    "BERT model",
    "Large language model",
    "Computer vision",
]

print("Loading Wikipedia pages...")
docs = []
for topic in topics:
    try:
        page    = wikipedia.page(topic, auto_suggest=False)
        content = page.content[:3000]
        docs.append(Document(
            page_content=content,
            metadata={"source": topic}
        ))
        print(f"  Loaded: {topic}")
    except Exception as e:
        print(f"  Skipped: {topic} - {e}")

print(f"Total documents: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

print("Creating embeddings...")
embeddings  = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")
print("Vector store saved!")
