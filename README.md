---
title: RAG With Phi 2 And LangChain
emoji: 👀
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.14.0
app_file: app.py
pinned: false
---

# Retrieval Augmented Generation with Phi-2: Question Answering
  ### This demo uses the Phi-2 language model and Retrieval Augmented Generation (RAG). It allows you to upload a txt file and ask the model questions related to the content of that file.
  ### If you don't have one, there is a txt file already loaded, the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the Phi-2 model is not aware of it.
  The context size of the Phi-2 model is 2048 tokens, so even this medium size wikipedia page (11.5k tokens) does not fit in the context window.
  Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt.

  The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.
  
  This demo was built using the Hugging Face `transformers` library, `langchain`, and `gradio`.
# Demo
  The demo has been depolyed to the following HuggingFace space.
  
  https://huggingface.co/spaces/rasyosef/RAG-with-Phi-2-and-LangChain
