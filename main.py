from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
import torch

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from flask import Flask, request, jsonify
from quart import Quart, request, jsonify
from llama_index.core.memory import ChatMemoryBuffer

## create embeddings

documents = SimpleDirectoryReader('/Users/sachinmishra/Desktop/VoiceMLPizza/mistral-pizza/Data').load_data()

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


embed_model = LangchainEmbedding(
HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Initialize Flask application
app = Quart(__name__)

# Initialize ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Define route to handle POST requests
@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve JSON data from the POST request
    data = request.json

    # Extract necessary information from the JSON data
    user_message = data.get('user_message', "")

    # Initialize chat engine with provided system prompt and memory
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=("""You are a helpful pizza-bot. Follow this flow for taking the orders from customers.
        First greet the customer. Then ask for orders of pizza from customers & try to customize their order if they ask for customization & finally confirm their order."""
                      "Hi"
                      ),
    )

    # Generate response using chat engine
    response = chat_engine.chat(user_message)
    print(response)

    # Return response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)