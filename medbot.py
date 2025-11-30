"""
Medical RAG Chatbot 
"""

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model (No token required - completely free)
LLM_MODEL = "google/flan-t5-base"  # 250MB, fast, no authentication needed

# RAG Settings
RETRIEVAL_TOP_K = 2
MAX_NEW_TOKENS = 256

# Custom RAG Prompt Template
CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def load_vector_store():
    """Load the FAISS vector store from disk."""
    print(f"Loading vector store from: {DB_FAISS_PATH}")
    
    # Initialize embeddings (must match create_memory_for_llm.py)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    # Load vector store
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print("✓ Vector store loaded successfully")
    return db

def initialize_llm():
    """Initialize the language model (no token required)."""
    print(f"Loading LLM: {LLM_MODEL}")
    print("(First run will download ~250MB, subsequent runs use cache)")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    
    # Create text generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Wrap in LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("✓ LLM loaded successfully")
    return llm


def create_rag_chain(db, llm):
    """Create the RAG chain with custom prompt."""
    print("Creating RAG chain...")
    
    # Create custom prompt
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
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
    
    print("✓ RAG chain created successfully")
    return qa_chain


def chat_loop(qa_chain):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Medical RAG Chatbot")
    print("=" * 60)
    print("Ask medical questions based on the encyclopedia.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("=" * 60 + "\n")
    
    while True:
        # Get user input
        user_question = input("You: ").strip()
        
        # Check for exit commands
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Medical RAG Chatbot. Goodbye!")
            break
        
        # Skip empty questions
        if not user_question:
            continue
        
        try:
            # Get response from RAG chain
            print("\nChatbot: ", end="", flush=True)
            response = qa_chain.invoke({"query": user_question})
            
            # Extract and print answer
            answer = response['result']
            print(answer)
            
            # Show number of sources
            if response.get('source_documents'):
                num_sources = len(response['source_documents'])
                print(f"\n[Retrieved from {num_sources} source(s)]")
            
            print()  # Empty line for readability
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try rephrasing your question.\n")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Medical RAG Chatbot - Initializing")
    print("=" * 60)
    
    try:
        # Step 1: Load vector store
        db = load_vector_store()
        
        # Step 2: Initialize LLM
        llm = initialize_llm()
        
        # Step 3: Create RAG chain
        qa_chain = create_rag_chain(db, llm)
        
        print("\n" + "=" * 60)
        print("✓ Chatbot is ready!")
        print("=" * 60)
        
        # Step 4: Start chat loop
        chat_loop(qa_chain)
        
    except KeyboardInterrupt:
        print("\n\nChatbot interrupted. Goodbye!")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've run 'create_memory_for_llm.py' first!")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()