from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import
from langchain.document_loaders import PyPDFLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import gradio as gr


FILEPATH = r"C:\Users\ah735\Downloads\BigDataApplicationsandTools.pdf"
LOCAL_MODEL = "llama2"
EMBEDDING = "nomic-embed-text"

loader = PyPDFLoader(FILEPATH)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)


embedding = OllamaEmbeddings(model=EMBEDDING)


vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding
)

llm = Ollama(
    base_url="http://localhost:11434",
    model=LOCAL_MODEL,
    verbose=True,
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]
    )
)

retriever = vectorstore.as_retriever()


template = """ You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}

    User: {question}
    Chatbot:
    """
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)


memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
    }
)


def chat_with_bot(user_input, chat_history=[]):
    
    response = qa_chain.invoke({"query": user_input})
    
   
    chat_history.append((user_input, response['result']))
    
    
    return chat_history, chat_history


with gr.Blocks(css=".chat {max-height: 500px; overflow-y: auto;}") as interface:
    gr.Markdown("<h1 style='text-align: center;'>_VOIS Bot</h1>")
    gr.Markdown("Ask questions related to Big Data applications and tools.")
    
    chatbot = gr.Chatbot(label="Chat History", height=500)
    with gr.Row():
        with gr.Column(scale=6):
            user_input = gr.Textbox(show_label=False, placeholder="Type your question here...")
        with gr.Column(scale=1):
            send_button = gr.Button("Send")

    def respond(user_input, chat_history):
        chat_history, chat = chat_with_bot(user_input, chat_history)
        return chat, gr.update(value="", interactive=True)

    user_input.submit(respond, [user_input, chatbot], [chatbot, user_input])
    send_button.click(respond, [user_input, chatbot], [chatbot, user_input])


interface.launch()