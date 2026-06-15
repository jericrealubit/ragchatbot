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

# 1. UPGRADED IMPORT: Switch from DuckDuckGo to true Google index parsing
from langchain_community.utilities import SerpAPIWrapper

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

# 2. INITIALIZE NATIVE GOOGLE SEARCH
# SerpAPI automatically uses Google Search under the hood
search_engine = SerpAPIWrapper()

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

# INTERNET FALLBACK PROMPT TEMPLATE
internet_template = """
You are a helpful AI assistant with real-time Google internet access capabilities.
Use the Live Google Search Results and Chat History below to provide a precise, clear, and accurate answer to the user's question.

Live Google Search Results:
{search_context}

Recent Chat History:
{chat_history}

User Question: {instruction}
Assistant Answer:"""
internet_prompt = PromptTemplate(template=internet_template, input_variables=["search_context", "chat_history", "instruction"])


def get_search_parameters(user_query: str):
    """Helper utility to localize queries and configure time parameters."""
    search_query = user_query

    # Localize weather anomalies automatically
    if "weather" in user_query.lower() and "perth" not in user_query.lower():
        search_query = f"{user_query} Perth Western Australia"

    return search_query


async def fetch_structured_web_results(query: str):
    """Queries Google's live engine via SerpAPI, returning matching titles, snippets, and structural reference links."""
    try:
        # Fetch Google's official organic response matrix
        raw_response = search_engine.results(query)
        organic_results = raw_response.get("organic_results", [])[:3] # Cap at top 3 Google results
    except Exception:
        organic_results = []

    if organic_results:
        context_pieces = []
        ui_cards = ["\n\n### 🌐 Google Search Results Found:"]

        for idx, result in enumerate(organic_results, 1):
            title = result.get("title", "No Title")
            url = result.get("link", "#")
            snippet = result.get("snippet", "")

            # Formulate structural logs for LLM comprehension
            context_pieces.append(f"Source [{idx}]: {title}\nURL: {url}\nSnippet: {snippet}\n")
            # Build interactive markdown components for the Chainlit viewport
            ui_cards.append(f"{idx}. **[{title}]({url})**\n   _{snippet}_\n")

        return "\n".join(context_pieces), "\n".join(ui_cards)

    return "No current live search data found.", ""


async def execute_query(user_query: str):
    """Streams standard document RAG queries with an automatic Google engine fallback."""
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")
    if not chain:
        return

    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    pdf_response_chunks = []
    res = cl.Message(content="")

    async for chunk in chain.astream({"instruction": user_query, "chat_history": formatted_history}):
        pdf_response_chunks.append(chunk)
        await res.stream_token(chunk)

    await res.send()
    full_pdf_response = "".join(pdf_response_chunks)

    negative_phrases = ["don't know", "dont know", "not include", "not mentioned", "not found in context", "no information"]
    if any(phrase in full_pdf_response.lower() for phrase in negative_phrases):
        status_msg = cl.Message(content="Context not found in PDF. Consulting Google Search...")
        await status_msg.send()

        search_query = get_search_parameters(user_query)
        live_search_context, ui_references_output = await fetch_structured_web_results(search_query)

        fallback_res = cl.Message(content="")
        fallback_chain = internet_prompt | llm | StrOutputParser()

        async for chunk in fallback_chain.astream({
            "instruction": user_query,
            "search_context": live_search_context,
            "chat_history": formatted_history
        }):
            await fallback_res.stream_token(chunk)

        if ui_references_output:
            fallback_res.content += ui_references_output

        await fallback_res.send()
        await status_msg.remove()

        history.append(f"User: {user_query}")
        history.append(f"Assistant: {fallback_res.content}")
    else:
        history.append(f"User: {user_query}")
        history.append(f"Assistant: {full_pdf_response}")

    cl.user_session.set("history_logs", history)


@cl.on_chat_start
async def on_chat_start():
    await cl.ChatSettings([
        cl.input_widget.Select(id="file_upload_status", label="File Upload Tray", values=["Enabled"], initial_value="Enabled")
    ]).send()
    cl.user_session.set("chain", None)
    cl.user_session.set("history_logs", [])
    await cl.Message(content="Welcome! Upload a PDF to start a document chat, or ask anything to search Google live!").send()


@cl.on_message
async def on_message(message: cl.Message):
    text_prompt = message.content.strip()

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
                if text:
                    pdf_text += text + "\n"

            pdf_text = re.sub(r'\[Image \d+\]', '', pdf_text)
            pdf_text = re.sub(r'--- PAGE \d+ ---', '', pdf_text)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([pdf_text])

            # 🛠️ NEW DEFENSIVE GUARD: Prevent FAISS IndexError if text extraction returns nothing
            if not docs or not pdf_text.strip():
                status_msg.content = f"⚠️ Processing failed! `{pdf_file.name}` appears to be an empty document or a scanned image with no extractable embedded text."
                await status_msg.update()
                return

            # This line will now only execute safely if documents actually exist!
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

    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")
    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    if text_prompt:
        if chain:
            await execute_query(text_prompt)
        else:
            res = cl.Message(content="")

            # Query Google's index live
            search_query = get_search_parameters(text_prompt)
            live_search_context, ui_references_output = await fetch_structured_web_results(search_query)

            fallback_chain = internet_prompt | llm | StrOutputParser()

            async for chunk in fallback_chain.astream({
                "instruction": text_prompt,
                "search_context": live_search_context,
                "chat_history": formatted_history
            }):
                await res.stream_token(chunk)

            if ui_references_output:
                res.content += ui_references_output

            await res.send()

            history.append(f"User: {text_prompt}")
            history.append(f"Assistant: {res.content}")
            cl.user_session.set("history_logs", history)
