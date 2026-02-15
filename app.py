import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



st.set_page_config(
    page_title="Chat with Hrishikesh",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chat with My Resume")
st.write(
    "Ask anything about my experience, projects, technical skills, or background."
)



@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"  # MUST match your Colab build
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})



client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_llm(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional assistant providing information about Hrishikesh's background and experience. "
                    "You must answer strictly using the provided context.\n"
                    "Do not fabricate, assume, or add external knowledge.\n\n"
                    "Formatting Rules:\n"
                    "- Write in clear professional paragraph format.\n"
                    "- Highlight measurable impact when available.\n"
                    "- Keep answers concise (3 to 6 sentences unless more detail is required).\n"
                    "- Avoid repeating similar information.\n\n"
                    "If the answer is not explicitly supported by the context, respond exactly with:\n"
                    "'That information is not available in the resume.'"
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content



query = st.text_input("Enter your question:")

if query:

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    if docs:

        # Build context
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


        st.subheader("Sources")

        sections = {
            doc.metadata.get("section")
            for doc in docs
            if doc.metadata.get("section")
        }

        for section in sorted(sections):
            st.write(f"- {section.capitalize()}")



