import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Prompt template
template = """Instruction:
You are an AI assistant for answering questions about the provided context.
You are given the following extracted parts of a long document and a question. Provide a detailed answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
=======
{context}
=======
Chat History:

{question}
Output:"""

QA_PROMPT = PromptTemplate(
  template=template,
  input_variables=["question", "context"]
)

# Returns a faiss vector store given a txt file
def prepare_vector_store(filename):
  # Load data
  loader = UnstructuredFileLoader(filename)
  raw_documents = loader.load()
  print(raw_documents[:1000])

  # Split the text
  text_splitter = CharacterTextSplitter(
      separator="\n\n",
      chunk_size=400,
      chunk_overlap=100,
      length_function=len
  )

  documents = text_splitter.split_documents(raw_documents)
  print(documents[:3])

  # Creating a vectorstore
  embeddings = HuggingFaceEmbeddings()
  vectorstore = FAISS.from_documents(documents, embeddings)
  print(embeddings, vectorstore)

  return vectorstore

# Load Phi-2 model from hugging face hub
model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
phi2 = pipeline("text-generation", tokenizer=tokenizer, model=model, max_new_tokens=128, device_map="auto") # GPU

phi2.tokenizer.pad_token_id = phi2.model.config.eos_token_id
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
      chain_type_kwargs={"prompt": QA_PROMPT, "verbose": True},
      verbose=True,
  )
  print(filename)
  return model

# Question Answering Chain
qa_chain = get_retrieval_qa_chain(filename="Oppenheimer-movie-wiki.txt")

# Generates response using the question answering chain defined earlier
def generate(question, chat_history):
  query = ""
  for req, res in chat_history:
    query += f"User: {req}\n"
    query += f"Assistant: {res}\n"
  query += f"User: {question}"

  result = qa_chain.invoke({"query": query})
  response = result["result"].strip()
  response = response.split("\n\n")[0].strip()

  if "User:" in response:
    response = response.split("User:")[0].strip()
  if "INPUT:" in response:
    response = response.split("INPUT:")[0].strip()
  if "Assistant:" in response:
    response = response.split("Assistant:")[1].strip()

  chat_history.append((question, response))

  return "", chat_history

# replaces the retreiver in the question answering chain whenever a new file is uploaded
def upload_file(qa_chain):
  def uploader(file):
    print(file)
    qa_chain.retriever = VectorStoreRetriever(
      vectorstore=prepare_vector_store(file)
    )
    return file
  return uploader

with gr.Blocks() as demo:
  gr.Markdown("""
  # RAG-Phi-2 Chatbot demo
  ### This chatbot uses the Phi-2 language model and retrieval augmented generation to allow you to add domain-specific knowledge by uploading a txt file.
  """)

  file_output = gr.File(label="txt file")
  upload_button = gr.UploadButton(
      label="Click to upload a txt file",
      file_types=["text"],
      file_count="single"
  )
  upload_button.upload(upload_file(qa_chain), upload_button, file_output)

  gr.Markdown("""
  ### Upload a txt file that contains the text data that you would like to augment the model with.
  If you don't have one, there is a default text data already loaded, the new Oppenheimer movie's wikipedia page.
  """)

  chatbot = gr.Chatbot(label="RAG Phi-2 Chatbot")
  msg = gr.Textbox(label="Message", placeholder="Enter text here")

  clear = gr.ClearButton([msg, chatbot])
  msg.submit(fn=generate, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()