from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
import faiss
import os
import openai

openai.api_key = "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"

Settings.llm = None  # Không sử dụng LLM trong quá trình tạo index

# 1. Load dữ liệu từ thư mục văn bản (txt, md, json,...)
documents = SimpleDirectoryReader("../crawl").load_data()  # chứa tài liệu UET

# 2. Khởi tạo model embedding (E5 base)
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

# 3. Khởi tạo FAISS vector store
dimension = 768  # E5-base vector size
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 4. Tạo service context
Settings.embed_model = embed_model

# 5. Chia văn bản thành đoạn ngắn (chunk)
node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=40)

# 6. Tạo index vector
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    transformations=[node_parser]
)

# 7. Tạo công cụ truy vấn (query engine)
query_engine = index.as_query_engine(similarity_top_k=3)

# 8. Chat với RAG
while True:
    query = input("👤 Bạn hỏi: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = query_engine.query(query)
    print("🤖 Trả lời:", response)