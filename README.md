# README: Setting Up the Environment using Pipenv

## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install langchain-text-splitters sentence-transformers transformers langchain-core
pipenv install streamlit


Run python files:
pipenv run python memory_for_llm.py
pipenv run python medbot.py

Run App:
pipenv run streamlit run  medbot_ui.py




