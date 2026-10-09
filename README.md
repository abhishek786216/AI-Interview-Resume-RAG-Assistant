## Live demo

Try the deployed application:

https://ai-interview-rag-client.onrender.com/

## Verify document loading

The loader uses LangChain's `Document` type and community loaders such as
`PyPDFLoader`, `TextLoader`, and `CSVLoader`. Install the dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Run the loader from the project root:

```powershell
python -m src.ingestion.loader
```

The command prints the number of LangChain documents loaded. A PDF can produce
multiple documents (one per page), while a CSV can produce one document per
row. Detailed activity is written to `logs/loader.log`, including each source
file, loader name, character count, skipped file, failure count, and the final
total.

## Split loaded documents into chunks

Run the complete load-and-split pipeline:

```powershell
python -m src.ingestion.splitter
```

The splitter uses LangChain's `RecursiveCharacterTextSplitter` with a
`chunk_size` of 1000 characters and a `chunk_overlap` of 200 characters.
Detailed chunk information is written to `logs/splitter.log`. Each output
document retains the source metadata and receives `chunk_index` and
`chunk_size` metadata.

## Create Hugging Face embeddings

The embedding step uses the Hugging Face
`sentence-transformers/all-MiniLM-L6-v2` model. Put your token in `.env` as
`HF_TOKEN=...`; the token is read from the environment and is never logged.
Install the additional dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m src.ingestion.embeddings
```

This runs loading, splitting, and embedding in order. The result is one
embedding vector per chunk. The embedding model writes its activity to
`logs/embeddings.log`, including the number of vectors and their dimensions.

## Store embeddings in FAISS

Build and persist a local FAISS index:

```powershell
python -m src.retrieval.vectorstore
```

The complete pipeline loads the files, creates chunks, generates embeddings,
and stores the vectors in `storage/faiss_index`. The index consists of
`index.faiss` and `index.pkl`; the latter contains the LangChain documents and
their metadata. Build details are written to `logs/vectorstore.log`. Only load
an index from this project-controlled directory because `index.pkl` is a
pickle file and is trusted by the reload helper.

To load the saved index in Python:

```python
from src.retrieval.vectorstore import load_vector_store

vector_store = load_vector_store()
results = vector_store.similarity_search("Python experience", k=3)
```

## Retrieve top-k chunks

Retrieve the most relevant chunks before connecting an LLM:

```powershell
python -m src.retrieval.retriever
```

Enter a question when prompted. The retriever loads the saved FAISS index,
embeds the query with the same Hugging Face model, and returns the top 3
matching chunks by default. Retrieval details, including rank, source, chunk
index, and chunk size, are written to `logs/retriever.log`.

You can also call it from Python:

```python
from src.retrieval.retriever import retrieve_documents

results = retrieve_documents("What Python experience is required?", k=3)
for result in results:
    print(result.page_content)
```

## Ask questions with the LLM

Add your Groq key to `.env`:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Then run the complete RAG application:

```powershell
python -m src.rag.qa
```

You can also run the file directly from any directory:

```powershell
python src\rag\qa.py
```

Type a question, for example:

```text
What Python experience is required?
```

The application retrieves the top 3 relevant chunks from FAISS, sends only
those chunks to the Groq model, and prints the answer with its sources. Type
`exit` to stop. The default model is `openai/gpt-oss-20b`; activity is
written to `logs/qa.log`. The API key is never logged. If `GROQ_MODEL` is
omitted from the project `.env`, the default is `openai/gpt-oss-20b`.

Questions can only be answered from files currently loaded into `data/`.
Place resume/project documents in `data/resume/` or `data/projects/`, rebuild
the FAISS index with `python -m src.retrieval.vectorstore`, and then ask the
question again.

## Interview question generation and evaluation

Generate one interview question from the retrieved resume/JD context:

```powershell
python -m src.interview.questions
```

In Python, evaluate a candidate answer against the same context:

```python
from src.interview.evaluator import evaluate_answer
from src.retrieval.retriever import retrieve_documents

context = retrieve_documents("candidate projects and technical decisions", k=5)
evaluation = evaluate_answer(
    "Why did you choose this technology?",
    "I chose it because it was easy to scale.",
    context,
)
print(evaluation.score)
print(evaluation.feedback)
```

Question-generation and evaluation activity is written to
`logs/questions.log` and `logs/evaluator.log`.

## Run the Streamlit app without LangGraph

The UI uses the LangChain components directly. LangGraph is not required:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose **Chat** to ask questions about the indexed documents. Choose
**Interview** to generate a question and evaluate your answer. Use
**Rebuild FAISS index** after adding or changing files under `data/`.

The project disables Streamlit's optional module watcher because the
`transformers` package contains vision modules that are not needed by this
text-embedding application. This prevents harmless `torchvision` watcher
tracebacks. Streamlit still runs normally; restart it after code changes.

## MERN-style realtime web app

The web version keeps the LangChain pipeline in Python and adds a MERN-style
web layer:

```text
React client → Express gateway → Python LangChain API → FAISS/Groq
```

Start the Python API:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```

Start the Express gateway in another terminal:

```powershell
cd server
npm install
npm start
```

Start the React client in a third terminal:

```powershell
cd client
npm install
npm run dev
```

Open `http://localhost:5173`. Users can upload resume, project, and job
description files at runtime. The first upload after starting the API clears
existing files from all three categories, so the FAISS index contains only
documents uploaded in the current user session. Uploading a resume or job
description then replaces the previous file in that category, while project
uploads are accumulated. Each upload immediately rebuilds the FAISS index, after
which Chat and Interview mode use only the newly uploaded documents.

If `python` opens the Microsoft Store or reports that Python is not installed,
install or select the global Python interpreter first. The repository currently
does not include an application entry point that calls `load_documents()`.