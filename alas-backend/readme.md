# 🧠 ALAS Backend

The **ALAS Backend** is a FastAPI-based server that powers an AI-driven adaptive learning and assessment system. It supports multiple user roles (Admin, Teacher, Student), intelligent chat interaction, and an adaptive JEE quiz engine using LLMs and vector search.

---

## 📁 Project Structure

```
alas-backend/
├── admin/                     # Admin-related logic
├── auth/                      # Auth system and role-based access
│   ├── auth.py
│   ├── deps.py
│   ├── models.py
│   └── schemas.py
├── chatbot/                   # Chatbot logic
│   ├── history/               # Chat logs
│   │   └── chat_history.json
│   ├── chat.py                # FAISS + RAG + LangChain chatbot
│   └── views.py               # Helper functions for chat handling
├── core/
│   └── config.py              # Centralized environment/config manager
├── db/                        # FAISS index, embeddings, question CSV
│   ├── faiss_index.bin
│   ├── text_chunks.pkl
│   └── corrected_questions.csv
├── ques_ans/              # Adaptive quiz engine
│   ├── quiz.py
│   └── quizagent.py
├── routes/                    # API routers
│   ├── admin.py
│   ├── student.py
│   └── teacher.py
├── main.py                    # Entry point
└── .env                       # Environment variables
```

---

## 🚀 Features

- ✅ **JWT Authentication** with role-based access (Admin / Teacher / Student)
- 🤖 **AI-Powered Chatbot** with FAISS vector search + LangChain + RAG
- 📚 **Adaptive JEE Quiz System** using LLM feedback and user profiling
- 🔐 **Secure Config Management** using `.env` and a centralized config
- 🔄 **Chat History Tracking** stored per-user for contextual conversations

---

## 🔧 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/alas-backend.git
   cd alas-backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables in `.env`**
   Example:
   ```env
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   MODEL=llama-3.3-70b-versatile
   RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
   CHAT_HISTORY_FILE=app/chatbot/history/chat_history.json
   FAISS_INDEX=app/db/faiss_index.bin
   TEXT_CHUNKS=app/db/text_chunks.pkl
   QUESTIONS=app/db/corrected_questions.csv
   ```

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

---

## 🧐 Tech Stack

- **FastAPI** – High-performance API framework
- **LangChain + Groq + HuggingFace** – Chatbot with RAG + embeddings
- **FAISS** – Vector search index for similarity search
- **CrossEncoder** – Semantic reranking
- **Pandas / NumPy** – Data management and embeddings
- **Pydantic** – Request validation
- **dotenv** – Environment configuration
- **Multiprocessing / Async** – Efficient request handling

---

## 📬 API Overview

| Endpoint                    | Method | Description                                 |
|----------------------------|--------|---------------------------------------------|
| `/auth/login`              | POST   | Authenticate user and get JWT               |
| `/admin/dashboard`         | GET    | Admin-only dashboard                        |
| `/query`                   | POST   | Ask chatbot a question                      |
| `/history/{user_id}`       | GET    | Get previous chat messages                  |
| `/quiz/start`              | POST   | Start adaptive quiz                         |
| `/quiz/answer`             | POST   | Submit answer and get next question         |

---

## 🛡️ Security & Best Practices

- API keys and sensitive info are stored securely via `.env` + `config.py`.
- JWT tokens validate user roles before accessing endpoints.
- Chat and quiz history are stored per-user in isolated files.
- Multiprocessing-safe design to avoid resource tracking issues.

---

## 👥 Contributing

PRs are welcome! Please make sure to:
- Follow the existing folder structure
- Add descriptive commit messages
- Run `black` or `ruff` for code formatting

---

## 📄 License

This project is under the **MIT License**.

---

## 💬 Contact

For queries, contributions, or bugs, reach out to [your-email@example.com](mailto:your-email@example.com)

---

