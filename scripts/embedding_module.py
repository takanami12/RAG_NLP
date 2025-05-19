from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re


class Embedding:
    def __init__(self, model_embedding="intfloat/multilingual-e5-base", chunk_size=256):
        self.model_embedding = model_embedding
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_embedding)
        self.model = AutoModel.from_pretrained(model_embedding)
        self.model_max_len = self.tokenizer.model_max_length
        self.embedding_dim = self.model.config.hidden_size

    def chunk_text(self, text: str):
        def tokenize(text_chunk):
            return self.tokenizer.encode(text_chunk, add_special_tokens=False)
        
        def tokenize_and_check_length(text_chunk):
            tokens = self.tokenizer.encode(
                text_chunk, add_special_tokens=False
            )
            return tokens, len(tokens)

        def decode(tokens):
            return self.tokenizer.decode(tokens)

        def split_by_token(tokens, max_len):
            if len(tokens) <= max_len:
                return [tokens]
            # Chia thành nhiều đoạn liên tiếp (thay vì chia đôi đệ quy)
            return [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]

        def process_chunk(text_chunk):
            tokens, length = tokenize_and_check_length(text_chunk)
            print(f"[Chunk] Tokens: {length} | Text: {text_chunk[:50]}...")
            if len(tokens) <= self.chunk_size:
                return [text_chunk]
            elif len(tokens) <= self.chunk_size * 1.5:
                return [decode(t) for t in split_by_token(tokens, self.chunk_size)]
            else:
                subchunks = [s.strip() for s in re.split(r',|;|\\.|\\n', text_chunk) if s.strip()]
                results = []
                for s in subchunks:
                    subtokens = tokenize(s)
                    if len(subtokens) <= self.chunk_size:
                        results.append(s)
                    else:
                        token_chunks = split_by_token(subtokens, self.chunk_size)
                        results.extend([decode(t) for t in token_chunks])
                return results

        # Step 1: tách thô theo câu và dòng
        rough_chunks = re.split(r'\.\s+|\n+', text)
        rough_chunks = [chunk.strip() for chunk in rough_chunks if chunk.strip()]

        final_chunks = []
        for chunk in rough_chunks:
            final_chunks.extend(process_chunk(chunk))

        return final_chunks

    def embedding(self, text: str):
        chunks = self.chunk_text(text)
        embeddings = []
        for chunk in chunks:
            token_ids = self.tokenizer.encode(chunk, add_special_tokens=True)
            if len(token_ids) > self.model_max_len:
                split_ids = [token_ids[i: i + self.model_max_len] for i in range(0, len(token_ids), self.model_max_len)]
                chunk_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in split_ids]
            else:
                chunk_texts = [chunk]

            for c in chunk_texts:
                inputs = self.tokenizer(
                    c,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.model_max_len
                )
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(F.normalize(cls_embedding, p=2, dim=1).squeeze(0))

        if len(embeddings) == 1:
            return embeddings[0].tolist()
        else:
            mean_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
            return mean_embedding.tolist()