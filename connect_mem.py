import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Hugging Face API Token
HF_TOKEN = "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"  # ⚠️ Keep secure in production!

# ✅ LLM: Use a real text-generation model
HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"  # Or try "tiiuae/falcon-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2"

# ✅ Load LLM from HuggingFace
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=2048,  # ⬅️ Set high limit (Zephyr supports large output)
        huggingfacehub_api_token=HF_TOKEN,
    )
    return llm

# ✅ Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Context: {context}
Question: {question}
Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ✅ Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ✅ Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# ✅ User query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# ✅ Print result
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
