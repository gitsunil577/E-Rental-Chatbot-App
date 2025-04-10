import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login


# Set your HuggingFace token (should be set in .env or system environment)
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate.from_template(custom_prompt_template)
    return prompt

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={
            "max_length": 512
        }
    )
    return llm

def main():
    st.title("E-Rental Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask me something...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        Context: {context}
        Question: {question}
        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            # Optional: display source docs in expandable section
            with st.expander("Sources"):
                for i, doc in enumerate(source_documents):
                    st.markdown(f"**Document {i+1}:**\n```\n{doc.page_content}\n```")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
