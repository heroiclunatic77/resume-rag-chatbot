import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Chat with Hrishikesh",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chat with My Resume")
st.write(
    "Ask anything about my experience, projects, technical skills, or background."
)

# ==========================================================
# LOAD VECTOR STORE (Built in Colab)
# ==========================================================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"  # MUST match Colab
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ==========================================================
# LOAD GROQ CLIENT
# ==========================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def ask_llm(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional resume assistant for Hrishikesh Keswani. "
                    "Only answer using the provided context. "
                    "If the answer is not present in the context, say: "
                    "'That information is not available in the resume.'"
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# ==========================================================
# USER INPUT
# ==========================================================

query = st.text_input("Enter your question:")

if query:

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    # Get LLM response
    answer = ask_llm(prompt)

    # Display answer
    st.subheader("Answer")
    st.write(answer)

    # # Display sources
    # st.subheader("Sources")
    # for doc in docs:
    #     st.write(doc.metadata)
