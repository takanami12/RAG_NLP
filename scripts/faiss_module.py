import os
import uuid
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from embedding_module import Embedding
from transformers import logging as hf_logging

load_dotenv()

hf_logging.set_verbosity_error()  # Suppress warnings from Hugging Face Transformers

class FAISSDBClient:
    def __init__(self, model_embedding, chunk_size=256, index_path="faiss_index"):
        self.embedding = Embedding(
            model_embedding=model_embedding,
            chunk_size=chunk_size
        )
        self.index_path = index_path
        self.index = None
        self.texts = []
        self.metadatas = []

        # Load index if exists
        if os.path.exists(index_path + ".index") and os.path.exists(index_path + ".pkl"):
            self.index = faiss.read_index(index_path + ".index")
            with open(index_path + ".pkl", "rb") as f:
                self.texts, self.metadatas = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.embedding.embedding_dim)

    def insert_with_text(self, text: str):
        try:
            vector = np.array([self.embedding.embedding(text)], dtype=np.float32)
            self.index.add(vector)
            metadata = {"id": str(uuid.uuid4()), "source": "script"}
            self.texts.append(text)
            self.metadatas.append(metadata)

            # Save index and metadata
            faiss.write_index(self.index, self.index_path + ".index")
            with open(self.index_path + ".pkl", "wb") as f:
                pickle.dump((self.texts, self.metadatas), f)

        except Exception as e:
            print("Insert error:", e)

    def query(self, query_text: str, top_k: int = 3):
        try:
            vector = np.array([self.embedding.embedding(query_text)], dtype=np.float32)
            scores, indices = self.index.search(vector, top_k)
            results = []
            metas = []
            for idx in indices[0]:
                if idx < len(self.texts):
                    results.append(self.texts[idx])
                    metas.append(self.metadatas[idx])
            return {
                "documents": results,
                "metadatas": metas,
            }
        except Exception as e:
            print("Query error:", e)
            return None