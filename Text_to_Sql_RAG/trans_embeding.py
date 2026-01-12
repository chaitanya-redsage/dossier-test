from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
sentences = ['Ho my name is Ananda.', 'I love programming in Python.']
embeddings = model.encode(sentences)
print(embeddings)
