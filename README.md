# Autonomous Tier-1 Helpdesk System

This project is the Capstone for Week 2 of my AI Engineering curriculum. It implements a fully modular Hybrid Retrieval-Augmented Generation (RAG) pipeline for an IT Helpdesk.

## Architecture
The codebase has been refactored from standalone learning scripts into a professional, modular architecture:
* **`ingestion/`**: Handles PostgreSQL vector database connections and BM25 index loading.
* **`retrieval/`**: Manages query expansion, pgvector similarity search, BM25 keyword search, and Reciprocal Rank Fusion (RRF).
* **`agent/`**: Manages the LLM interaction using strict Pydantic schemas and OpenAI Structured Outputs.

## Features
* **Hybrid Search:** Combines dense vector embeddings with sparse keyword matching (BM25) for high-accuracy retrieval.
* **Hallucination Guardrails:** Forces the LLM to verify context and flag unanswerable queries to prevent fabricated answers.
* **CLI Application:** A clean `app.py` entry point that orchestrates the data flow.