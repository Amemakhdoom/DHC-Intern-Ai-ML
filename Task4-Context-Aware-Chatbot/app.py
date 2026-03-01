import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

print("Loading model...")
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=150,
    device=0 if torch.cuda.is_available() else -1,
    do_sample=False,
    temperature=1.0,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

embeddings  = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory    = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)
print("Model loaded!")

def chat(message, history):
    result  = qa_chain.invoke({"question": message})
    answer  = result["answer"]
    sources = list(set([
        doc.metadata["source"]
        for doc in result["source_documents"]
    ]))
    return f"{answer}\n\nSources: {', '.join(sources)}"

demo = gr.ChatInterface(
    fn=chat,
    title="Context-Aware AI Chatbot",
    description="Powered by LangChain + RAG + Wikipedia | DevelopersHub Internship",
    examples=[
        "What is machine learning?",
        "How does deep learning work?",
        "What is BERT?",
        "What is computer vision?",
        "What is the difference between AI and ML?",
    ],
    theme=gr.themes.Soft()
)

demo.launch(share=True)
