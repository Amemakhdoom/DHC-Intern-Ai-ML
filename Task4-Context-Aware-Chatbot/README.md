# 🤖 Task 4 - Context-Aware Chatbot Using LangChain and RAG

## Objective
Build a conversational chatbot that remembers context and retrieves
external information during conversations.

## Project Structure
- app.py — Gradio deployment
- train.py — RAG pipeline building script
- requirements.txt — Required libraries
- vectorstore/ — FAISS vector database
- README.md — Documentation

## Knowledge Base
Wikipedia pages on:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Neural Networks
- BERT Model
- Large Language Models
- Computer Vision

## Architecture
- LangChain RAG Pipeline
- FAISS Vector Store for document search
- Sentence Transformers for embeddings
- TinyLlama-1.1B as LLM
- ConversationBufferMemory for context memory

## How to Run
pip install -r requirements.txt
python train.py
python app.py

## Skills Gained
- Conversational AI development
- Document embedding and vector search
- Retrieval-Augmented Generation RAG
- LLM integration and deployment
- Gradio deployment
