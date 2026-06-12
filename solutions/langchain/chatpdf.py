import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from PyPDF2 import PdfReader

# Initialize Groq Cloud LLM
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY", "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"),
    model_name="llama-3.1-8b-instant", 
    temperature=0.2
)

# Template adapted for highly accurate RAG injection
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
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload the Prison Officer Booklet PDF to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

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

    # 2. Chop text into bite-sized 1,000-character pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([pdf_text])

    # 3. Create a totally free, local embedding model (runs completely on your CPU machine)
    # This transforms document text strings into searchable vector numbers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Generate an in-memory database of your document chunks
    vector_store = Chroma.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 closest chunks

    # 5. Build the production chain mapping retrieved context cleanly
    chain = (
        {
            "pdf_context": lambda inputs: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(inputs["instruction"])]),
            "instruction": lambda inputs: inputs["instruction"]
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )

    cl.user_session.set("chain", chain)
    
    msg.content = f"`{pdf_file.name}` successfully processed via Local Vector Index! Ask me about the salary tiers now."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = cl.Message(content="")

    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)

    # Stream the highly accurate, down-sampled token block securely
    async for chunk in chain.astream(
        {"instruction": message.content},
        config={"callbacks": [cb]}
    ):
        await res.stream_token(chunk)

    await res.send()
