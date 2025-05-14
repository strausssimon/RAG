from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

# 📁 Pfad zum Modell (GGUF)
model_path = "models/llama-3-8b-instruct.Q4_K_M.gguf"

# 🔁 LLM initialisieren
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_gpu_layers=35,  # oder 0 für reine CPU
    verbose=False,
)

# 📁 Dokumentenverzeichnis
doc_dir = Path("documents")
documents = []

for file in doc_dir.glob("*"):
    if file.suffix == ".txt":
        loader = TextLoader(str(file), encoding='utf-8')
        docs = loader.load()
        documents.extend(docs)

# 📄 Text in Chunks aufteilen
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 📌 Embeddings (lokal, z. B. BAAI/bge-small-en)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# 🧠 RAG QA-Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🎤 Frage-Antwort-Loop
while True:
    frage = input("\nFrage an das RAG-System (oder 'exit'): ")
    if frage.lower() in ["exit", "quit"]:
        break
    antwort = qa.run(frage)
    print("\n📘 Antwort:", antwort)