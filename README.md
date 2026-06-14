# Local RAG Enterprise PDF Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** application built with **Chainlit** and **LangChain**. This chatbot enables secure, contextual conversations over local PDF documents by combining localized semantic vector searches with high-throughput cloud inference.

---

## 🏗️ Core Architecture Flow

The backend processes incoming data and user interactions through a fully decoupled pipeline:

1. **Ingestion & Text Extraction:** Native binary reading and noise cleanup of documents.
2. **Chunking & Segmentation:** Splitting extensive text blocks into managed semantic slices to fit LLM context ceilings perfectly.
3. **Vector Embeddings:** Mapping text into high-dimensional geometric coordinates.
4. **Local Store Indexing:** Caching structural indices locally for fast vector similarity lookups ($k=3$).
5. **Contextual Inference:** Dynamically pairing matching document context, sliding conversational memory logs, and user prompts into a streamlined payload handled by high-performance inference models.

---

## 🛠️ Explaining the Deep Stack (Imported Tools)

To maintain clean, professional documentation, here is exactly what every imported tool handles inside your `chatpdf.py` file:

### Core Python Modules

* **`os`**: Manages operational environment routing, enabling the script to securely fetch your underlying system variables without exposing raw values in the codebase.
* **`re`**: Python's native Regular Expressions library. It cleans up extracted PDF strings by stripping out structural page numbers, image headers, and noisy formatting artifacts before indexing.
* **`logging`**: Overrides background engine diagnostics. It silences harmless operational verbosity from matrix computing engines (like FAISS) to keep your command-line output clean.

### Chainlit (Asynchronous Interface Engine)

* **`chainlit` (`cl`)**: An asynchronous Python framework used to build production-grade conversational UIs. It manages real-time UI interactions, streams data chunks smoothly, and handles cross-origin browser network events natively.

### LangChain Orchestration & Expression Framework

* **`ChatOpenAI`**: A modular communication class configured here to point directly to Groq Cloud's ultra-fast inference infrastructure using an OpenAI-compatible API gateway.
* **`PromptTemplate`**: Structures dynamic inputs. It acts as an architectural blueprint that enforces system behavior boundaries, mapping raw document extracts and chat history logs into a rigid format before sending it to the LLM.
* **`StrOutputParser`**: Extracts text from raw API response objects. It strips away complex token arrays and status metadata, outputting only the pure string answer for the frontend user interface.

### Document Processing & Mathematical Vectors

* **`RecursiveCharacterTextSplitter`**: Splitting text blindly by word count breaks sentences apart. This utility intelligently splits text by searching for natural line breaks, paragraph marks, and spaces, establishing a 1,000-character chunk matrix with a 200-character safety overlap to preserve contextual continuity.
* **`HuggingFaceEmbeddings`**: Downloads and interfaces with the underlying sentence-transformers engine. It converts plain sentences into dense 384-dimensional mathematical floating-point arrays representing semantic meaning.
* **`FAISS`**: *Facebook AI Similarity Search*. A high-efficiency, localized vector database that clusters and indexes your embeddings directly in systems memory for blazing-fast similarity searches.
* **`PdfReader`**: An implementation from `PyPDF2` that parses binary PDF file streams into raw, indexable string buffers.

---

## 🚀 Getting Started

### 1. Environmental Dependencies

Run the following package install via your environment terminal. This includes `sentence-transformers` to guarantee your local text vectorization runs out-of-the-box:

```bash
python -m pip install chainlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu PyPDF2 python-dotenv sentence-transformers

```

### 2. Guarding Secrets (`.env`)

Create an environment file named exactly `.env` in the root folder of your project to manage configurations cleanly:

```text
GROQ_API_KEY=gsk_your_private_api_key_here

```

> ⚠️ **Important:** Add `.env` to your `.gitignore` configuration file immediately to block credential scanning on git pushes.

---

## 🏃 Execution Workflow

Launch the application using your explicit solution target pathway:

```bash
chainlit run solutions/langchain/chatpdf.py --host 0.0.0.0 --port 8080

```

Once running, navigate to the local server port provided inside your terminal log. Drop an item (like an invoice or documentation sheet) using the **paperclip icon**, type your query simultaneously (e.g., *"summarize the content"*), and witness real-time streaming context generation!