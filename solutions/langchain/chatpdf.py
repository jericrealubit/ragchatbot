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

# Clean instruction template for model context
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
    # Force Chainlit to ensure the attachment icon displays in older builds
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="file_upload_status",
            label="File Upload Tray",
            values=["Enabled"],
            initial_value="Enabled"
        )
    ]).send()
    
    cl.user_session.set("chain", None)
    await cl.Message(
        content="Welcome! Click the paperclip icon in the chat box below to upload your PDF and ask questions."
    ).send()


async def execute_query(user_query: str):
    """Helper function to cleanly stream the answer from the active chain architecture"""
    chain = cl.user_session.get("chain")
    if not chain:
        return

    res = cl.Message(content="")
    async for chunk in chain.astream({"instruction": user_query}):
        await res.stream_token(chunk)
    await res.send()


@cl.on_message
async def on_message(message: cl.Message):
    # Capture the textual instruction from the input window explicitly
    text_prompt = message.content.strip()

    # 1. Catching File Uploads: Check if a document is attached to this current action
    if message.elements:
        pdf_files = [el for el in message.elements if el.mime == "application/pdf" or el.name.endswith('.pdf')]
        
        if pdf_files:
            pdf_file = pdf_files[0]
            status_msg = cl.Message(content=f"Analyzing and indexing `{pdf_file.name}`...")
            await status_msg.send()

            # Read raw PDF text content
            reader = PdfReader(pdf_file.path)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"

            # CLEANUP: Strip out structural layout artifacts
            pdf_text = re.sub(r'\[Image \d+\]', '', pdf_text)
            pdf_text = re.sub(r'--- PAGE \d+ ---', '', pdf_text)
            
            # Segment content out to fit context windows seamlessly
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([pdf_text])

            # Process embeddings using FAISS engine
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Rebuild stable LCEL Chain mappings without context collision
            chain = (
                {
                    "pdf_context": (lambda x: x["instruction"]) | retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                    "instruction": lambda x: x["instruction"]
                }
                | prompt 
                | llm 
                | StrOutputParser()
            )

            cl.user_session.set("chain", chain)
            
            # Allow the session memory to register the chain before moving forward
            await cl.sleep(0.1) 
            
            # Fixed update syntax for older Chainlit versions
            status_msg.content = f"`{pdf_file.name}` successfully processed!"
            await status_msg.update()

            # Execute the attached text query instantly
            if text_prompt:
                await execute_query(text_prompt)
            else:
                await cl.Message(content="Document ready! What would you like to know about it?").send()
            return

    # 2. Standard follow-up question processing flow (no file attached)
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Please upload a PDF document using the attachment icon first!").send()
        return

    if text_prompt:
        await execute_query(text_prompt)