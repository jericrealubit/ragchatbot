import os
import re
import time
import logging
import httpx
import chainlit as cl
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq

# 1. UPGRADED IMPORT: Switch from DuckDuckGo to true Google index parsing
from langchain_community.utilities import SerpAPIWrapper

# Load environment variables securely
load_dotenv()
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Initialize Groq Cloud LLM
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.2,
    # streaming=True is handled safely out of the box here
)

# 2. INITIALIZE NATIVE GOOGLE SEARCH (lazy — avoids crashing the app at import
# time if SERPAPI_API_KEY is missing; PDF RAG still works without it)
_search_engine = None


def _get_search_engine():
    global _search_engine
    if _search_engine is None:
        if not os.environ.get("SERPAPI_API_KEY"):
            return None
        _search_engine = SerpAPIWrapper()
    return _search_engine


# 3. WA AI DIGITAL FACTS — fetched from the site's own /llms.txt, which waai.au
# generates from the same data its pages are built from, so the bot can't drift
# from the site. Cached for hours; when the fetch fails, the last good copy is
# kept (or, before the first success, questions fall through to live search)
# and the fetch is retried a few minutes later rather than on every message.
WAAI_FACTS_URL = os.environ.get("WAAI_FACTS_URL", "https://waai.au/llms.txt")
WAAI_FACTS_TTL = 6 * 60 * 60
WAAI_FACTS_RETRY = 5 * 60
NOT_IN_SITE_FACTS = "NOT_IN_SITE_FACTS"
_site_facts = ""
_site_facts_next_fetch = 0.0


async def get_site_facts() -> str:
    global _site_facts, _site_facts_next_fetch
    if time.monotonic() < _site_facts_next_fetch:
        return _site_facts
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.get(WAAI_FACTS_URL)
            response.raise_for_status()
        _site_facts = response.text.strip()
        _site_facts_next_fetch = time.monotonic() + WAAI_FACTS_TTL
    except Exception:
        logging.exception("Could not fetch site facts from %s", WAAI_FACTS_URL)
        _site_facts_next_fetch = time.monotonic() + WAAI_FACTS_RETRY
    return _site_facts


# Site facts prompt. The facts come first so the long, unchanging part of the
# prompt is a stable prefix across questions.
site_template = """
You are the assistant on the WA AI Digital website (waai.au). These are the business's facts, generated from the website's own content:

{site_facts}

Answer the user's question using ONLY the facts above. Quote prices exactly as written and include the relevant page link when there is one. Never invent services, prices, clients, timelines or results that are not in the facts.

If the question is not about WA AI Digital (its services, pricing, process, projects, people or contact details), or the facts do not answer it, reply with exactly NOT_IN_SITE_FACTS and nothing else.

Respond in natural, clean conversational prose or basic markdown bullet points only. Do not output JSON, XML tags or tool calls.

Recent Chat History:
{chat_history}

User Question: {instruction}
Assistant Answer:"""
site_prompt = PromptTemplate(template=site_template, input_variables=["site_facts", "chat_history", "instruction"])
# A lookup over a short document doesn't need gpt-oss's default reasoning
# budget, and every token here counts against the key's per-minute cap.
site_chain = site_prompt | llm.bind(reasoning_effort="low") | StrOutputParser()


async def answer_from_site(user_query: str, formatted_history: str) -> str | None:
    """Answers from the WA AI Digital facts, or returns None when they don't cover the question."""
    site_facts = await get_site_facts()
    if not site_facts:
        return None
    try:
        answer = await site_chain.ainvoke({
            "site_facts": site_facts,
            "chat_history": formatted_history,
            "instruction": user_query,
        })
    except Exception:
        logging.exception("Site facts answer failed")
        return None
    answer = (answer or "").strip()
    if not answer or NOT_IN_SITE_FACTS in answer:
        return None
    return answer


# Base Prompt Template for PDF interactions
template = """
You are a helpful AI assistant. Use the following PDF Reference Context and Chat History to answer the user's question accurately.
If you do not know the answer based on the context, politely state that you don't know. Keep your answer direct, clear, and concise.

CRITICAL FORMATTING RULE:
Respond using natural, clean conversational prose or basic markdown bullet points only.
Do NOT output raw JSON blocks, XML tags, nested array brackets, or code-like schemas, as this will trigger a backend API error.

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

CRITICAL MODEL CONSTRAINT: Do NOT attempt to call, invoke, or format any internal or external tools, functions, or special Harmony format tags (such as <|call|>, <|search|>, or JSON tool blocks). You are operating in a text-generation sandbox. Output your entire response as standard conversational text and clean Markdown only.

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
    engine = _get_search_engine()
    if engine is None:
        return "Live web search is disabled (SERPAPI_API_KEY not set).", ""
    try:
        # Fetch Google's official organic response matrix
        raw_response = engine.results(query)
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

            # ✅ FIXED: Plain text formatting for the LLM to prevent Groq tool-choice bugs
            context_pieces.append(f"Search Result {idx} Title: {title}\nSummary: {snippet}\n")

            # Keep interactive markdown components intact for the Chainlit UI viewport
            ui_cards.append(f"{idx}. **[{title}]({url})**\n   _{snippet}_\n")

        return "\n".join(context_pieces), "\n".join(ui_cards)

    return "No current live search data found.", ""


async def stream_answer(chain, inputs: dict, res: cl.Message) -> str:
    """Streams a chain into a Chainlit message, guaranteeing the message never ends up blank."""
    chunks = []
    try:
        async for chunk in chain.astream(inputs):
            chunks.append(chunk)
            await res.stream_token(chunk)

        # Reasoning models (gpt-oss) occasionally return an empty final answer; retry once without streaming
        if not "".join(chunks).strip():
            answer = await chain.ainvoke(inputs)
            if answer and answer.strip():
                chunks = [answer]
                res.content = answer
    except Exception as e:
        logging.exception("LLM call failed")
        res.content = f"⚠️ Sorry, the model failed to respond: {e}"
        return ""

    answer = "".join(chunks)
    if not answer.strip():
        res.content = "⚠️ Sorry, the model returned an empty response. Please try again."
    return answer


async def execute_query(user_query: str):
    """Streams standard document RAG queries with an automatic Google engine fallback."""
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs") or []
    if not chain:
        return

    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    res = cl.Message(content="")
    full_pdf_response = await stream_answer(chain, {"instruction": user_query, "chat_history": formatted_history}, res)
    await res.send()

    if not full_pdf_response.strip():
        return

    negative_phrases = ["don't know", "dont know", "not include", "not mentioned", "not found in context", "no information"]
    if any(phrase in full_pdf_response.lower() for phrase in negative_phrases):
        status_msg = cl.Message(content="Context not found in PDF. Consulting Google Search...")
        await status_msg.send()

        search_query = get_search_parameters(user_query)
        live_search_context, ui_references_output = await fetch_structured_web_results(search_query)

        fallback_res = cl.Message(content="")
        fallback_chain = internet_prompt | llm | StrOutputParser()

        await stream_answer(fallback_chain, {
            "instruction": user_query,
            "search_context": live_search_context,
            "chat_history": formatted_history
        }, fallback_res)

        if ui_references_output:
            fallback_res.content += ui_references_output

        # send(), not update(): if nothing streamed (an error, or the empty-answer
        # retry), the UI has never seen this message and drops an update to it.
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
    await cl.Message(content="Welcome! Ask me about WA AI Digital's services, pricing and projects. You can also upload a PDF to chat with it, or ask anything else and I'll search Google live.").send()


@cl.on_message
async def on_message(message: cl.Message):
    text_prompt = message.content.strip()

    if message.elements:
        pdf_files = [el for el in message.elements if el.mime == "application/pdf" or (el.name or "").endswith('.pdf')]
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
    history = cl.user_session.get("history_logs") or []
    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    if text_prompt:
        if chain:
            await execute_query(text_prompt)
        else:
            # Questions about the business are answered from the site's own
            # facts; anything those don't cover falls through to live search.
            site_answer = await answer_from_site(text_prompt, formatted_history)
            if site_answer:
                await cl.Message(content=site_answer).send()
                history.append(f"User: {text_prompt}")
                history.append(f"Assistant: {site_answer}")
                cl.user_session.set("history_logs", history)
                return

            res = cl.Message(content="")

            # Query Google's index live
            search_query = get_search_parameters(text_prompt)
            live_search_context, ui_references_output = await fetch_structured_web_results(search_query)

            fallback_chain = internet_prompt | llm | StrOutputParser()

            await stream_answer(fallback_chain, {
                "instruction": text_prompt,
                "search_context": live_search_context,
                "chat_history": formatted_history
            }, res)

            if ui_references_output:
                res.content += ui_references_output

            # send(), not update() — see execute_query.
            await res.send()

            history.append(f"User: {text_prompt}")
            history.append(f"Assistant: {res.content}")
            cl.user_session.set("history_logs", history)
