from ingestion import get_index
from flask import Flask, request, jsonify
from llama_index.core.memory import ChatMemoryBuffer
import uvicorn

# Initialize Flask application
app = Flask(__name__)

path = "/Users/sachinmishra/Desktop/VoiceMLPizza/mistral-pizza/pizza_knowledge_base.pdf"
index = get_index(path)

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
                      ),
    )

    # Generate response using chat engine
    response = chat_engine.chat(user_message)

    # Return response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    
    uvicorn.run(app, host="0.0.0.0", port=5000, debug=True)