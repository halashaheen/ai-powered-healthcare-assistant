import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# LangChain and FAISS imports
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline




# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # Token-free model from original script
RETRIEVAL_TOP_K = 2
MAX_NEW_TOKENS = 256

CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store from disk."""
    st.write("Initializing Embeddings and Loading Vector Store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        st.success("Vector store loaded successfully.")
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}. Ensure '{DB_FAISS_PATH}' exists and `create_memory_for_llm.py` has been run.")
        return None


@st.cache_resource
def initialize_llm():
    """Initialize the token-free language model using HuggingFacePipeline."""
    st.write(f"Loading LLM: {LLM_MODEL}...")
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        
        # Create text generation pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Wrap in LangChain HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        st.success("LLM loaded successfully.")
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None


def set_custom_prompt():
    """Create the custom prompt template."""
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )


@st.cache_resource
def create_rag_chain(_db, _llm):
    """Create and cache the RetrievalQA chain."""
    st.write("Creating RAG chain...")
    db= _db
    llm= _llm
    
    if db is None or llm is None:
        return None
    
    prompt = set_custom_prompt()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_TOP_K}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    st.success("RAG chain created successfully.")
    return qa_chain


def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
    st.title("👨‍⚕️ Medical RAG Chatbot")
    st.markdown("Ask questions based on the embedded medical knowledge (uses **Flan-T5-Base**, no API key needed).")
    
    # 1. Initialize and cache resources
    vectorstore = get_vectorstore()
    llm = initialize_llm()
    qa_chain = create_rag_chain(vectorstore, llm)
    
    if qa_chain is None:
        st.warning("Chatbot is not ready. Please resolve the errors above (e.g., missing vector store or model loading failure).")
        return
    
    # 2. Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 3. Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # 4. Handle user input
    prompt = st.chat_input("Pass your prompt here...")

    if prompt:
        # Display user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Get response from the RAG chain
        with st.spinner("Thinking..."):
            try:
                # Invoke the chain
                response = qa_chain.invoke({'query': prompt})
                
                result = response["result"]
                source_documents = response.get("source_documents", [])
                
                # Format the response to show sources clearly
                result_to_show = f"**Answer:** {result}"
                
                if source_documents:
                    source_text = "\n\n**Sources Retrieved:**\n"
                    # Display the content of the retrieved documents
                    for i, doc in enumerate(source_documents):
                        metadata = doc.metadata.get('source', 'No source metadata')
                        source_text += f"**[{i+1}]** (Source: `{metadata}`)\n{doc.page_content[:150]}...\n\n"
                    
                    result_to_show += source_text

                # Display assistant message
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                
                # Append to session state
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                error_message = f"An error occurred during query processing: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({'role': 'assistant', 'content': error_message})


if __name__ == "__main__":
    main()