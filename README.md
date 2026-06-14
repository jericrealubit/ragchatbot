Here is a fresh, professional `README.md` tailored specifically for your project. It clearly explains the RAG (Retrieval-Augmented Generation) architecture you've built, documents your secure setup, and provides clear execution instructions for anyone running it locally or in a container.

---

# Local LLM Crash Course: Enterprise PDF Chatbot

A high-performance, lightweight **Retrieval-Augmented Generation (RAG)** conversational pipeline built with **Chainlit**, **LangChain (LCEL)**, and **FAISS**. This application securely reads local PDF documents, indexes their contents in-memory using localized vector embeddings, and leverages **Groq Cloud's Llama 3.1 architecture** for near-instantaneous context-aware streaming responses.

---

## 🏗️ Architecture Overview

The application processes user queries and document context through an asynchronous, decoupled pipeline:

1. **Document Ingestion:** Reads raw data natively via `PyPDF2`.
2. **Text Segmentation:** Chunks document string contexts down into 1,000-character blocks with a 200-character rolling overlap using `RecursiveCharacterTextSplitter`.
3. **Vector Vectorization:** Generates localized mathematical semantic matrices using the `all-MiniLM-L6-v2` HuggingFace engine.
4. **Local Vector Store:** Caches embedded documents into a local in-memory `FAISS` database structure for rapid proximity querying ($k=3$).
5. **Inference Execution:** Dynamically bundles user input, sliding conversation history logs, and extracted PDF reference blocks into a unified context window processed over high-throughput Groq Cloud inference endpoints.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Framework:** Chainlit (Async Web UI Workflow Engine)
* **Orchestration:** LangChain Expression Language (LCEL)
* **LLM Engine:** ChatOpenAI interface pointing to Groq Cloud API (`llama-3.1-8b-instant`)
* **Vector Embeddings:** HuggingFace Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Document Parsing:** PyPDF2

---

## 🚀 Getting Started

### 1. Installation

Clone this repository to your workspace or open it directly inside your GitHub Codespace environment, then install the required core packages:

```bash
pip install chainlit langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu PyPDF2 python-dotenv

```

### 2. Environment Configuration (`.env`)

To safeguard your infrastructure credentials, the application relies strictly on localized environment routing. Create a file named `.env` in the root folder of your project workspace:

```text
GROQ_API_KEY=gsk_your_actual_secret_groq_key_here

```

> ⚠️ **Security Compliance Note:** Never commit your `.env` file to a public repository. Ensure your project's `.gitignore` contains a line explicitly omitting `.env` configurations from Git tracking layers.

---

## 🏃 Running the Application

Launch the local streaming server by supplying the explicit source pathway to your main LangChain solution layout:

```bash
chainlit run solutions/langchain/chatpdf.py --host 0.0.0.0 --port 8080

```

Once executed, follow the local address terminal prompt or access your forwarded port via your browser to interact with the Chat UI interface.

---

## 💡 Usage Workflow

1. **Upload Document:** Click the **Paperclip / Attachment icon** inside the single-line input text field to select your target `.pdf` file.
2. **Simultaneous Prompts:** You can supply a textual command (e.g., *"summarize the content"* or *"how much to pay"*) directly alongside your initial file upload.
3. **Continuous Chatting:** Once the system reports that the file has been successfully indexed, the vector state remains loaded inside your session cache. You can continue asking deep follow-up questions sequentially.
4. **Context History Navigation:** The conversational engine actively utilizes a sliding memory window context to track previous statement turns seamlessly.