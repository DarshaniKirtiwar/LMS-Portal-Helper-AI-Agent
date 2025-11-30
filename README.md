# LMS-Portal-Helper-AI-Agent
This is an integrated AI chatbot is designed to help students find the complex answers within seconds using simple prompts.
Project Overview**

The LMS Portal Helper AI Agent is designed to enhance the learning experience by providing instant, intelligent support to students navigating online course content. Instead of manually searching through long videos, PDFs, and modules, learners can ask the AI agent questions and receive accurate, context-aware answers within seconds. The system integrates course materials, external tools, and online search capabilities to deliver comprehensive assistance on demand. With features like personalized guidance, doubt clarification, real-time information retrieval, and adaptive learning support, the agent transforms traditional LMS platforms into interactive, learner-centric environments. This project aims to significantly improve study efficiency, reduce frustration, and increase overall engagement and course completion rates.


Problem Statement

Learners often waste significant time searching through lengthy videos, PDFs, and course modules to find specific answers, which slows down learning and reduces overall engagement. Traditional LMS platforms lack real-time, interactive support, leaving students stuck when instructors are unavailable. This leads to frustration, low productivity, and inconsistent understanding of key concepts. Many learners also struggle with technical jargon, making it difficult to interpret course material without additional guidance. Since similar questions are frequently asked by different learners, time is wasted repeating the same clarifications. Existing LMS tools do not adapt to individual learning speeds or styles, and they fail to provide personalized assistance. The absence of a unified system that combines course insights with external reliable information further limits learning depth. Without instant, accurate support, motivation drops and course completion rates suffer. An AI-driven LMS Helper can solve these gaps effectively.


Why agents? -- Why are agents the right solution to this problem

AI agents are ideal for this problem because they provide instant, context-aware assistance, reducing the time learners spend searching through course materials manually. Agents can read, interpret, and reference course content, enabling students to get accurate answers in seconds. Unlike static FAQs or traditional chatbots, agents can reason, plan, and use tools—such as web search or document retrieval—to give deeper, updated information. They also learn patterns from repeated queries, offering personalized support tailored to each learner’s pace and style. AI agents operate 24/7, ensuring continuous guidance without waiting for instructors. Their ability to unify internal course content with external knowledge sources creates a central learning assistant, greatly improving clarity, efficiency, and user engagement. Overall, agents transform passive content consumption into an interactive, dynamic learning experience.


"""
Study Buddy Agent
- Indexes course materials (txt, pdf, docx, pptx)
- Creates embeddings (sentence-transformers)
- Stores embeddings in FAISS (fast similarity search)
- Retrieves relevant chunks for a question
- Calls an LLM (pluggable) to generate final answer using retrieved context
"""

import os
import glob
import re
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# Document parsers
from PyPDF2 import PdfReader
import docx
from pptx import Presentation

# ---------------------------
# Helpers: file loaders
# ---------------------------

def load_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def load_pdf(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)

def load_pptx(path: str) -> str:
    prs = Presentation(path)
    slides_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                txt = shape.text.strip()
                if txt:
                    slides_text.append(txt)
    return "\n".join(slides_text)

LOADER_MAP = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
}

# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple chunking by characters (keeps words intact).
    chunk_size: approx characters per chunk
    """
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:length])
            break
        # try to break at last space before end to keep words whole
        split = text.rfind(' ', start, end)
        if split == -1:
            split = end
        chunk = text[start:split].strip()
        chunks.append(chunk)
        start = split - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------------------
# Indexing / Embeddings
# ---------------------------

class Indexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.dim = self.embed_model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []  # list of dicts parallel to vectors (file, chunk_text, start_pos)
    
    def build_index(self, texts: List[Tuple[str, dict]]):
        """
        texts: list of (chunk_text, metadata_dict)
        """
        vectors = []
        for chunk, meta in texts:
            emb = self.embed_model.encode(chunk, show_progress_bar=False)
            vectors.append(emb.astype('float32'))
            self.metadata.append({"text": chunk, **meta})
        if not vectors:
            raise ValueError("No chunks to index.")
        matrix = np.vstack(vectors)
        self.index = faiss.IndexFlatIP(self.dim)  # inner product for cosine after normalize
        faiss.normalize_L2(matrix)
        self.index.add(matrix)
    
    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def load(self, index_path: str, meta_path: str):
        import json
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        qvec = self.embed_model.encode(query_text).astype('float32')
        faiss.normalize_L2(qvec.reshape(1, -1))
        D, I = self.index.search(qvec.reshape(1, -1), top_k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(self.metadata[idx])
        return results

# ---------------------------
# LLM call (pluggable)
# ---------------------------

def call_llm(prompt: str) -> str:
    """
    Replace the body of this function with a call to Google AI Studio / ADK
    or any other model provider.
    Example (pseudocode for Google AI Studio):
        from google.ai import aiplatform
        response = aiplatform.predict_text(model="projects/.../locations/.../models/...", input=prompt)
        return response.text
    Or for OpenAI (if you choose to use it):
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        return resp["choices"][0]["message"]["content"]
    """
    # ==== Placeholder simple echo (for offline testing) ====
    # In practice, you MUST replace this with your Google AI Studio ADK invocation.
    return "LLM_PLACEHOLDER_RESPONSE: " + prompt[:200]  # remove or replace in production

# ---------------------------
# Agent logic
# ---------------------------

class StudyBuddyAgent:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def build_context_prompt(self, question: str, retrieved: List[dict]) -> str:
        # Build a prompt that instructs the LLM to answer using the retrieved context.
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(f"Source {i} (excerpt):\n{r['text']}\n")
        context = "\n---\n".join(context_parts)
        prompt = (
            "You are a helpful study assistant. Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, say you cannot find an authoritative answer and provide "
            "a best-effort response flagged as 'inferred'. Keep the answer concise and include references to "
            "source indices where appropriate.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n"
        )
        return prompt
    
    def answer(self, question: str, top_k: int = 5) -> dict:
        retrieved = self.indexer.query(question, top_k=top_k)
        prompt = self.build_context_prompt(question, retrieved)
        llm_response = call_llm(prompt)
        return {
            "question": question,
            "answer": llm_response,
            "sources": retrieved
        }

# ---------------------------
# Utility: index files from a folder
# ---------------------------

def index_folder(folder_path: str, indexer: Indexer, chunk_size: int = 800, overlap: int = 150):
    files = glob.glob(os.path.join(folder_path, "*"))
    all_chunks = []
    for f in files:
        ext = Path(f).suffix.lower()
        loader = LOADER_MAP.get(ext)
        if not loader:
            print(f"Skipping unsupported file: {f}")
            continue
        print(f"Loading: {f}")
        try:
            text = loader(f)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            continue
        if not text.strip():
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            meta = {"source": os.path.basename(f), "chunk_index": i}
            all_chunks.append((c, meta))
    print(f"Total chunks to index: {len(all_chunks)}")
    indexer.build_index(all_chunks)

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # 1) Index course materials (place pdf/docx/txt/pptx files inside ./course_materials)
    data_folder = "./course_materials"
    os.makedirs(data_folder, exist_ok=True)
    idx = Indexer(model_name="all-MiniLM-L6-v2")
    print("Indexing folder...")
    index_folder(data_folder, idx, chunk_size=800, overlap=150)
    print("Index built.")
    
    # Optional: persist index
    idx.save("faiss_index.bin", "metadata.json")
    
    # 2) Create agent and ask questions
    agent = StudyBuddyAgent(idx)
    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = agent.answer(q, top_k=5)
        print("\n--- Answer ---")
        print(out["answer"])
        print("\n--- Sources ---")
        for s in out["sources"]:
            print(f" - {s['source']} (chunk {s['chunk_index']})")


def call_llm(prompt: str) -> str:
    # Pseudocode - replace with actual Google AI Studio client code
    from google import ai  # imaginary import - replace with real client
    client = ai.TextModelClient(model="projects/PROJECT/locations/LOC/models/MODEL")
    response = client.predict(prompt=prompt, max_output_tokens=500, temperature=0.0)
    return response.text

