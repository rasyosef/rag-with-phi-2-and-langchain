import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import TextIteratorStreamer
from threading import Thread

# Prompt template
template = """Instruction:
You are an AI assistant for answering questions about the provided context.
You are given the following extracted parts of a long document and a question. Provide a detailed answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
=======
{context}
=======
Question: {question}
Output:\n"""

QA_PROMPT = PromptTemplate(
  template=template,
  input_variables=["question", "context"]
)

# Returns a faiss vector store given a txt file
def prepare_vector_store(filename):
  # Load data
  loader = UnstructuredFileLoader(filename)
  raw_documents = loader.load()

  # Split the text
  text_splitter = CharacterTextSplitter(
      separator="\n\n",
      chunk_size=400,
      chunk_overlap=100,
      length_function=len
  )

  documents = text_splitter.split_documents(raw_documents)

  # Creating a vectorstore
  embeddings = HuggingFaceEmbeddings()
  vectorstore = FAISS.from_documents(documents, embeddings)

  return vectorstore

# Load Phi-2 model from hugging face hub
model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)

streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)
phi2 = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto",
    streamer=streamer
  ) # GPU
hf_model = HuggingFacePipeline(pipeline=phi2)

# Retrieveal QA chian
def get_retrieval_qa_chain(filename):
  llm = hf_model
  retriever = VectorStoreRetriever(
    vectorstore=prepare_vector_store(filename)
  )
  model = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=retriever,
      chain_type_kwargs={"prompt": QA_PROMPT},
  )
  return model

# Question Answering Chain
qa_chain = get_retrieval_qa_chain(filename="Oppenheimer-movie-wiki.txt")

# Generates response using the question answering chain defined earlier
def generate(question, answer):
  query = f"{question}"

  thread = Thread(target=qa_chain.invoke, kwargs={"input": {"query": query}})
  thread.start()

  response = ""
  for token in streamer:
    response += token
    yield response

# replaces the retreiver in the question answering chain whenever a new file is uploaded
def upload_file(qa_chain):
  def uploader(file):
    qa_chain.retriever = VectorStoreRetriever(
      vectorstore=prepare_vector_store(file)
    )
    return file
  return uploader

with gr.Blocks() as demo:
  gr.Markdown("""
  # RAG-Phi-2 Question Answering demo
  ### This demo uses the Phi-2 language model and Retrieval Augmented Generation (RAG) to allow you to upload a txt file and ask the model questions related to the content of that file.
  ### If you don't have one, there is a txt file already loaded, the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the Phi-2 model is not aware of it.
  The context size of the Phi-2 model is 2048 tokens, so even this medium size wikipedia page (11.5k tokens) does not fit in the context window.
  Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt.
  The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.
  """)

  file_output = gr.File()
  upload_button = gr.UploadButton(
      label="Click to upload a text file",
      file_types=["text"],
      file_count="single"
  )
  upload_button.upload(upload_file(qa_chain), upload_button, file_output)
 
  with gr.Row():
    with gr.Column():
      ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=3)
    with gr.Column():
      ans = gr.Textbox(label="Answer", lines=4)
  with gr.Row():
    with gr.Column():
      btn = gr.Button("Submit")
    with gr.Column():
      clear = gr.ClearButton([ques, ans])
  btn.click(fn=generate, inputs=[ques, ans], outputs=[ans])
  examples = gr.Examples(
        examples=[
            "Who portrayed J. Robert Oppenheimer in the new Oppenheimer movie?",
            "In the plot of the movie, why did Lewis Strauss resent Robert Oppenheimer?"
        ],
        inputs=[ques],
    )

demo.queue().launch()