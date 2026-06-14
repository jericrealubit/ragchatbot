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

# Load environment variables securely
load_dotenv()

# Suppress harmless background loading notifications from FAISS
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Initialize Groq Cloud LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# Expanded Template to handle both Document Context and Conversation History
# Clean, polished template structure for the LLM context
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


@cl.on_chat_start
async def on_chat_start():
    # Force Chainlit to ensure the attachment icon displays in older builds
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="file_upload_status",
            label="File Upload Tray",
            values=["Enabled"],
            initial_value="Enabled"
        )
    ]).send()

    # Initialize both the processing chain and an empty history array in the session
    cl.user_session.set("chain", None)
    cl.user_session.set("history_logs", [])

    await cl.Message(
        content="Welcome! Click the paperclip icon in the chat box below to upload your PDF and ask questions."
    ).send()


async def execute_query(user_query: str):
    """Streams responses from the chain while appending strings directly to the history state"""
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")

    if not chain:
        return

    # Formulate a structured string block of the recent conversation history
    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    res = cl.Message(content="")

    # Stream the tokens smoothly from Groq while mapping context variables explicitly
    async for chunk in chain.astream({
        "instruction": user_query,
        "chat_history": formatted_history
    }):
        await res.stream_token(chunk)

    await res.send()

    # Commit the exchange to session memory so the next message can reference it
    history.append(f"User: {user_query}")
    history.append(f"Assistant: {res.content}")
    cl.user_session.set("history_logs", history)


# ADD A GENERAL FALLBACK TEMPLATE (No PDF context)
general_template = """
You are a helpful AI assistant. Answer the user's question directly, clearly, and concisely.
Recent Chat History:
{chat_history}

User Question: {instruction}
Assistant Answer:"""

general_prompt = PromptTemplate(template=general_template, input_variables=["chat_history", "instruction"])

# ... (Keep your existing @cl.on_chat_start and execute_query helper function the same)


@cl.on_message
async def on_message(message: cl.Message):
    text_prompt = message.content.strip()

    # 1. Catching File Uploads: Process document additions
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

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Chain setup mapping document dependencies
            chain = (
                {
                    "pdf_context": (lambda x: x["instruction"]) | retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                    "chat_history": lambda x: x["chat_history"],
                    "instruction": lambda x: x["instruction"]
                }
                | prompt
                | llm
                | StrOutputParser()
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

    # 2. Dynamic Processing Flow
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history_logs")
    formatted_history = "\n".join(history[-6:]) if history else "No previous history."

    if text_prompt:
        if chain:
            # A PDF exists! Run the standard RAG pipeline route
            await execute_query(text_prompt)
        else:
            # NO PDF uploaded! Fallback to direct conversational LLM mode
            res = cl.Message(content="")

            # Formulate standard LCEL chain expression on-the-fly for clean streaming
            fallback_chain = general_prompt | llm | StrOutputParser()

            async for chunk in fallback_chain.astream({
                "instruction": text_prompt,
                "chat_history": formatted_history
            }):
                await res.stream_token(chunk)

            await res.send()

            # Save the exchange to your ongoing conversation history tracker
            history.append(f"User: {text_prompt}")
            history.append(f"Assistant: {res.content}")
            cl.user_session.set("history_logs", history)
