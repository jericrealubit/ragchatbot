# 🚀 Enterprise PDF Chatbot (Hybrid RAG & Live Search)

Welcome! This system is a high-performance intelligence layer powered by **Chainlit** and **LangChain Engine Expressions (LCEL)**. It features an adaptive, dual-route architecture designed to pull down target information smoothly depending on user context:

1. **Document RAG Mode:** When a PDF is uploaded, text chunks are converted into local geometric vectors using a localized embedding transformer model (`all-MiniLM-L6-v2`) and indexed via an in-memory `FAISS` database for contextual query matching ($k=3$).
2. **Live Web Search Mode:** If no document is active in the session cache, the application seamlessly adapts, utilizing a real-time internet search utility to scrape relevant web contexts on-the-fly.

All inference requests are handled via high-throughput **Groq Cloud API architectures** running optimized `Llama 3.1` model instances with dynamic rolling conversational history log tracking.

---

## 💡 How to Use

* **Option A: General & Live Internet Queries** Simply type any general knowledge or real-time query straight into the message box (e.g., *"What is the weather like in Perth today?"* or *"Who won the match last night?"*). The assistant will access the live web to formulate a precise answer.

* **Option B: Secure PDF Context Analysis**
  Click the 📎 **Paperclip / Attachment icon** in the input tray to submit a target `.pdf` file. You can attach a question directly alongside your upload (e.g., *"Extract the total balance due"*). Once indexed, you can continue asking sequential follow-up questions normally.

---

## 📬 Contact & Collaboration

Are you an engineering hiring manager looking for a versatile Full-Stack Web Developer with practical AI implementation skills, or a business owner looking to deploy optimized, zero-overhead automated intelligence tools onto your internal systems? Let's connect!

* **Name:** Jeric Realubit
* **Role:** Full-Stack Web Developer & AI Solutions Engineer
* **Location:** Perth, Western Australia (Open to local, hybrid, and global remote opportunities)
* **LinkedIn:** [linkedin.com/in/jericrealubit](https://linkedin.com/in/jericrealubit)
* **Mobile:** [+61 491 098 073](tel:+61491098073)
* **GitHub:** [github.com/jericrealubit](https://github.com/jericrealubit)