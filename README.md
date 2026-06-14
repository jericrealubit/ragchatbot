# Local Hybrid RAG & Live Search Enterprise Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** application built with **Chainlit** and **LangChain**. This chatbot features a dynamic dual-routing architecture: it enables secure, contextual conversations over local PDF documents using localized semantic vector searches, and seamlessly falls back to real-time internet search when no document is active.

---

## 🚀 Live Application
Try the live system deployed on my domain here: [waai.au/chat](https://waai.au/chat)

<img width="50%" height="1810" alt="image" src="https://github.com/user-attachments/assets/781b8169-ee00-4070-a39e-f30747ebfaeb" />

---

## 🏗️ Core Architecture Flow

The application handles incoming user intents and document processing through two primary processing paths:

1. **Document RAG Path:** When a PDF is supplied, raw binary pages are extracted, cleaned, and segmented into 1,000-character semantic chunks. These blocks are vectorized using a local transformer model (`all-MiniLM-L6-v2`) and cached in an in-memory `FAISS` vector index for ultra-low latency similarity parsing ($k=3$).
2. **Live Web Search Path:** If no document is active in the user session memory, the application executes live background search queries via DuckDuckGo. Real-time web result snippets are scraped and dynamically injected straight into the LLM context layer.

All operations feature optimized streaming tokens over high-throughput **Groq Cloud API frameworks** (`llama-3.1-8b-instant`) with a rolling conversational turn tracker preserving contextual continuity.

---

## 🛠️ Explaining the Deep Stack (Imported Tools)

To maintain transparent, professional documentation, here is exactly what every imported tool handles inside your `chatpdf.py` execution layer:

### Core Python Modules

* **`os`**: Handles system operational environmental calls, pulling down host credentials safely without exposing private parameters inside the repository.
* **`re`**: Python's native Regular Expressions engine. It purges noisy extraction metadata (like layout page markings and image artifacts) from the text buffer prior to embedding.
* **`logging`**: Silences non-critical background runtime diagnostic logging from matrix math subsystems like FAISS to keep your application output clean.

### Chainlit (Asynchronous Core Framework)

* **`chainlit` (`cl`)**: An asynchronous python automation library for launching interactive chat interfaces. It natively manages user socket states, pushes token text blocks smoothly, and parses client-side interaction events.

### LangChain Expression Language (LCEL) & Utilities

* **`ChatOpenAI`**: A communication model layer mapped directly to Groq Cloud's ultra-low-latency endpoint infrastructure utilizing an OpenAI-compatible translation class.
* **`PromptTemplate`**: Structures input formatting rules. Enforces behavioral bounds, compiling your live search strings or vector context arrays cleanly alongside active history logs before execution.
* **`StrOutputParser`**: Extracts text strings from raw LLM object responses, filtering out performance metrics or structural tokens for clean UI presentation.
* **`DuckDuckGoSearchAPIWrapper`**: A zero-config network connector that acts as the real-time internet engine when the app runs without an active document database.

### Data Ingestion & Mathematical Vectors

* **`RecursiveCharacterTextSplitter`**: Intelligently splices long document texts by natural word bounds and paragraph breaks using a 1,000-character chunk boundary with a 200-character rolling overlap.
* **`HuggingFaceEmbeddings`**: Manages the underlying sentence-transformers engine to convert readable text strings into 384-dimensional semantic matrix arrays.
* **`FAISS`**: *Facebook AI Similarity Search*. An optimized structural calculation engine running locally in system memory to perform vector cosine similarity calculations.
* **`PdfReader`**: An implementation from `PyPDF2` that reads binary PDF files into raw string buffers.

---

## 🚀 Getting Started

### 1. Environmental Dependencies

Install all package dependencies directly into your active runtime environment using the python binary module command flag:

```bash
python -m pip install chainlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu PyPDF2 python-dotenv sentence-transformers duckduckgo-search ddgs

```

### 2. Guarding Secrets (`.env`)

Create an environment file named exactly `.env` at your root directory path to secure your infrastructure access tokens:

```text
GROQ_API_KEY=gsk_your_private_api_key_here

```

> ⚠️ **Security Policy Reminder:** Ensure your project's `.gitignore` contains a line explicitly omitting `.env` configurations from Git tracking layers.

---

## 🏃 Execution Workflow

Launch the live application streaming engine by target-pointing to your python solution structure:

```bash
chainlit run solutions/langchain/chatpdf.py --host 0.0.0.0 --port 8080

```

Once initialized, open the forwarded port interface link provided in your console logs. You can interact directly with the web search functionality, or load up an invoice, spreadsheet, or documentation guide using the attachment bar to lock down localized RAG operations.

---

## 📬 Contact & Collaboration

Are you an engineering hiring manager looking for a versatile Full-Stack Web Developer with practical AI implementation skills, or a business owner looking to deploy optimized, zero-overhead automated intelligence tools onto your internal systems? Let's connect!

* **Name:** Jeric Realubit
* **Role:** Full-Stack Web Developer & AI Solutions Engineer
* **Location:** Perth, Western Australia (Open to local, hybrid, and global remote opportunities)
* **LinkedIn:** [linkedin.com/in/jericrealubit](https://www.google.com/search?q=https://linkedin.com/in/jericrealubit)
* **Mobile:** [+61 491 098 073](https://www.google.com/search?q=tel:%2B61491098073)
* **GitHub:** [github.com/jericrealubit](https://www.google.com/search?q=https://github.com/jericrealubit)
