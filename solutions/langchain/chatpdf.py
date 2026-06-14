import os
import re
import logging
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Suppress harmless background loading notifications from FAISS matrix engine
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Initialize Groq Cloud LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY", ""),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# Clean, structured instructions template for the model context
template = """
[INST] <<SYS>>
You are a helpful assistant. Use the following pieces of PDF Reference Context to answer the user's question. 
If you don't know the answer, say you don't know. Keep your answer precise and concise.

PDF Reference Context:
{pdf_context}
<</SYS>>
User: {instruction} [/INST]"""

prompt = PromptTemplate(template=template, input_variables=["pdf_context", "instruction"])


@cl.on_chat_start
async def on_chat_start():
    # Welcome message on launch. Chatbox stays perfectly clean and 1-line thin!
    await cl.Message(
        content="Welcome! Click the paperclip/upload icon in the chat box below to attach your PDF and start chatting."
    ).send()


@cl.on_file_upload(accept=["application/pdf"])
async def on_file_upload(files: list[cl.File]):
    """This handles the file when uploaded via the attachment icon in the chat box"""
    pdf_file = files[0]
    
    msg = cl.Message(content=f"Analyzing and indexing `{pdf_file.name}`...")
    await msg.send()

    # 1. Read entire PDF file text
    reader = PdfReader(pdf_file.path)
    pdf_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

    # 1b. CLEANUP: Strip out structural layout noise like "[Image 1]" or bracketed artifacts
    pdf_text = re.sub(r'\[Image \d+\]', '', pdf_text)
    pdf_text = re.sub(r'--- PAGE \d+ ---', '', pdf_text)

    # 2. Chop cleaned text into bite-sized 1,000-character pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([pdf_text])

    # 3. Create local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Generate an in-memory database of your document chunks using FAISS
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 5. Build stable LCEL chain mapping variables explicitly without dictionary leakage
    chain = (
        {
            "pdf_context": (lambda x: x["instruction"]) | retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "instruction": lambda x: x["instruction"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Store the active document chain in the session
    cl.user_session.set("chain", chain)

    msg.content = f"`{pdf_file.name}` successfully processed! Feel free to type your questions now."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    # Safety fallback if they message before uploading anything
    if not chain:
        await cl.Message(content="Please upload a PDF document using the attach icon first!").send()
        return

    res = cl.Message(content="")

    # Stream tokens cleanly with no cluttered background tracking cards
    async for chunk in chain.astream({"instruction": message.content}):
        await res.stream_token(chunk)

    await res.send()