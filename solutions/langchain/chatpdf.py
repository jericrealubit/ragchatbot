import os
import re
import logging
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# 1. NEW IMPORT: Grab the DuckDuckGo Search utility
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Load environment variables securely
load_dotenv()
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Initialize Groq Cloud LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# 2. INITIALIZE THE SEARCH ENGINE
search_engine = DuckDuckGoSearchAPIWrapper(max_results=3)

# Base Prompt Template for PDF interactions
template = """
You are a helpful AI assistant. Use the following PDF Reference Context and Chat History to answer the user's question accurately.
If you do not know the answer based on the context, politely state that you don't know. Keep your answer direct, clear, and concise.

PDF Reference Context:
{pdf_context}

Recent Chat History:
{chat_history}

User Question: {instruction}
Assistant Answer:"""
prompt = PromptTemplate(template=template, input_variables=["pdf_context", "chat_history", "instruction"])

# 3. INTERNET FALLBACK PROMPT TEMPLATE
internet_template = """
You are a helpful AI assistant with real-time internet access capabilities.
Use the Live Search Results and Chat History below to provide a precise, clear, and accurate answer to the user's question.

Live Search Results:
{search_context}

Recent Chat History:
{chat_history}

User Question: {instruction}
Assistant Answer:"""
internet_prompt = PromptTemplate(template=internet_template, input_variables=["search_context", "chat_history", "instruction"])


@cl.on_chat_start
async def on_chat_start():
    await cl.ChatSettings([
        cl.input_widget.Select(id="file_upload_status", label="File Upload Tray", values=["Enabled"], initial_value="Enabled")
    ]).send()
    cl.user_session.set("chain", None)
    cl.user_session.set("history_logs", [])
    await cl.Message(content="Welcome! Upload a PDF to start a document chat, or ask anything to search the web live!").send()


async def execute_query(user_query: str):
    """Streams standard document RAG queries, with an automatic internet fallback if not found"""
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")
    if not chain:
        return

    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    # 1. First, try to get the answer from the PDF context
    pdf_response_chunks = []
    res = cl.Message(content="")

    async for chunk in chain.astream({
        "instruction": user_query,
        "chat_history": formatted_history
    }):
        pdf_response_chunks.append(chunk)
        await res.stream_token(chunk)

    await res.send()
    full_pdf_response = "".join(pdf_response_chunks)

    # 2. Check if the LLM failed to find it in the PDF context
    negative_phrases = ["don't know", "dont know", "not include", "not mentioned", "not found in context", "no information"]
    if any(phrase in full_pdf_response.lower() for phrase in negative_phrases):

        # Update the UI to let the user know it's pivoting to the web
        status_msg = cl.Message(content="Context not found in PDF. Searching the internet live...")
        await status_msg.send()

        # Run the background web search pass
        try:
            live_search_results = search_engine.run(user_query)
        except Exception:
            live_search_results = "Unable to retrieve live results right now."

        # Clear the previous "I don't know" message block and stream the real answer
        fallback_res = cl.Message(content="")
        fallback_chain = internet_prompt | llm | StrOutputParser()

        async for chunk in fallback_chain.astream({
            "instruction": user_query,
            "search_context": live_search_results,
            "chat_history": formatted_history
        }):
            await fallback_res.stream_token(chunk)

        await fallback_res.send()
        await status_msg.remove() # Clean up the status message

        # Save the actual internet answer to history logs
        history.append(f"User: {user_query}")
        history.append(f"Assistant: {fallback_res.content}")
        cl.user_session.set("history_logs", history)
    else:
        # Save the valid PDF answer to history logs
        history.append(f"User: {user_query}")
        history.append(f"Assistant: {full_pdf_response}")
        cl.user_session.set("history_logs", history)


@cl.on_message
async def on_message(message: cl.Message):
    text_prompt = message.content.strip()

    # SECTION A: Document Upload Logic
    if message.elements:
        pdf_files = [el for el in message.elements if el.mime == "application/pdf" or el.name.endswith('.pdf')]
        if pdf_files:
            pdf_file = pdf_files[0]
            status_msg = cl.Message(content=f"Analyzing and indexing `{pdf_file.name}`...")
            await status_msg.send()

            reader = PdfReader(pdf_file.path)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text: pdf_text += text + "\n"

            pdf_text = re.sub(r'\[Image \d+\]', '', pdf_text)
            pdf_text = re.sub(r'--- PAGE \d+ ---', '', pdf_text)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([pdf_text])

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            chain = (
                {
                    "pdf_context": (lambda x: x["instruction"]) | retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                    "chat_history": lambda x: x["chat_history"],
                    "instruction": lambda x: x["instruction"]
                }
                | prompt | llm | StrOutputParser()
            )
            cl.user_session.set("chain", chain)
            await cl.sleep(0.1)

            status_msg.content = f"`{pdf_file.name}` successfully processed!"
            await status_msg.update()

            if text_prompt:
                await execute_query(text_prompt)
            else:
                await cl.Message(content="Document ready! What would you like to know about it?").send()
            return

    # SECTION B: Multi-Route Execution Processing Flow
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")
    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    if text_prompt:
        if chain:
            # Route 1: PDF Document is active -> Query vector storage context layers
            await execute_query(text_prompt)
        else:
            # Route 2: Standalone conversation -> Trigger Internet Search Fallback
            res = cl.Message(content="")

            # Run the background web scraping pass asynchronously
            try:
                live_search_results = search_engine.run(text_prompt)
            except Exception:
                live_search_results = "Unable to retrieve live results right now. Rely on core parameters."

            # Construct dynamic LCEL stream pipeline passing scraped parameters
            fallback_chain = internet_prompt | llm | StrOutputParser()

            async for chunk in fallback_chain.astream({
                "instruction": text_prompt,
                "search_context": live_search_results,
                "chat_history": formatted_history
            }):
                await res.stream_token(chunk)

            await res.send()

            # Sync context exchanges into session-wide history trackers
            history.append(f"User: {text_prompt}")
            history.append(f"Assistant: {res.content}")
            cl.user_session.set("history_logs", history)
