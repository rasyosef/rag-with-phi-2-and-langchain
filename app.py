import gradio as gr

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

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

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

# Load Phi-2 model from hugging face hub
model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True
)

# sentence transformers to be used in vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-base-v4",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


# Returns a faiss vector store retriever given a txt file
def prepare_vector_store_retriever(filename):
    # Load data
    loader = UnstructuredFileLoader(filename)
    raw_documents = loader.load()

    # Split the text
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=800, chunk_overlap=0, length_function=len
    )

    documents = text_splitter.split_documents(raw_documents)

    # Creating a vectorstore
    vectorstore = FAISS.from_documents(
        documents, embeddings, distance_strategy=DistanceStrategy.DOT_PRODUCT
    )

    return VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": 2})


# Retrieveal QA chian
def get_retrieval_qa_chain(text_file, hf_model):
    retriever = default_retriever
    if text_file != default_text_file:
        retriever = prepare_vector_store_retriever(text_file)

    chain = RetrievalQA.from_chain_type(
        llm=hf_model,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return chain


# Generates response using the question answering chain defined earlier
def generate(question, answer, text_file, max_new_tokens):
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )
    phi2_pipeline = pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device_map="auto",
        streamer=streamer,
    )

    hf_model = HuggingFacePipeline(pipeline=phi2_pipeline)
    qa_chain = get_retrieval_qa_chain(text_file, hf_model)

    query = f"{question}"

    thread = Thread(target=qa_chain.invoke, kwargs={"input": {"query": query}})
    thread.start()

    response = ""
    for token in streamer:
        response += token
        yield response.strip()


# replaces the retreiver in the question answering chain whenever a new file is uploaded
def upload_file(file):
    return file, file


with gr.Blocks() as demo:
    gr.Markdown(
        """
  # Retrieval Augmented Generation with Phi-2: Question Answering demo
  ### This demo uses the Phi-2 language model and Retrieval Augmented Generation (RAG). It allows you to upload a txt file and ask the model questions related to the content of that file.
  ### If you don't have one, there is a txt file already loaded, the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the Phi-2 model is not aware of it.
  The context size of the Phi-2 model is 2048 tokens, so even this medium size wikipedia page (11.5k tokens) does not fit in the context window.
  Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt.
  The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.
  """
    )

    default_text_file = "Oppenheimer-movie-wiki.txt"
    default_retriever = prepare_vector_store_retriever(default_text_file)

    text_file = gr.State(default_text_file)

    gr.Markdown(
        "## Upload a txt file or Use the Default 'Oppenheimer-movie-wiki.txt' that has already been loaded"
    )

    file_name = gr.Textbox(
        label="Loaded text file", value=default_text_file, lines=1, interactive=False
    )
    upload_button = gr.UploadButton(
        label="Click to upload a text file", file_types=["text"], file_count="single"
    )
    upload_button.upload(upload_file, upload_button, [file_name, text_file])

    gr.Markdown("## Enter your question")
    tokens_slider = gr.Slider(
        8,
        256,
        value=64,
        label="Maximum new tokens",
        info="A larger `max_new_tokens` parameter value gives you longer text responses but at the cost of a slower response time.",
    )

    with gr.Row():
        with gr.Column():
            ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=3)
        with gr.Column():
            ans = gr.Textbox(label="Answer", lines=4, interactive=False)
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
        with gr.Column():
            clear = gr.ClearButton([ques, ans])

    btn.click(fn=generate, inputs=[ques, ans, text_file, tokens_slider], outputs=[ans])
    examples = gr.Examples(
        examples=[
            "Who portrayed J. Robert Oppenheimer in the new Oppenheimer movie?",
            "In the plot of the movie, why did Lewis Strauss resent Robert Oppenheimer?",
        ],
        inputs=[ques],
    )

demo.queue().launch()
