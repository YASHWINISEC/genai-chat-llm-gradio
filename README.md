## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### DESIGN STEPS:

#### STEP 1:
Install Libraries: Use pip to install gradio, transformers, and accelerate.
#### STEP 2:
Import Gradio: Import gradio as gr for the user interface.
#### STEP 3:
Import Transformers: Import AutoTokenizer and AutoModelForCausalLM for the LLM.
#### STEP 4:
Import PyTorch: Import torch for deep learning operations.
#### STEP 5:
Load Tokenizer: Initialize the tokenizer for your chosen model
#### STEP 6:
Load Model: Load the pre-trained LLM model (e.g., tiiuae/falcon-rw-1b) and move it to a GPU if available.
#### STEP 7:
Set Model to Eval Mode: Put the model in evaluation mode to disable training-specific behaviors.
#### STEP 8:
Define Respond Function: Create a function that takes a message and chat history, tokenizes the message, generates a response using the model, and updates the chat history.

### PROGRAM:
```
!pip install gradio transformers accelerate -q

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b").to("cuda")
model.eval()

def respond(message, chat_history):
    chat_history = chat_history or []
    chat_history.append(("User", message))
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split(message, 1)[-1].strip() 
    chat_history.append(("AI", response))
    return chat_history, chat_history


with gr.Blocks() as demo:
    gr.Markdown("Chat with Local LLM ")
    chatbot_ui = gr.Chatbot(label="Chatbot", type='tuples') 
    txt = gr.Textbox(label="Type your message here...", placeholder="Ask anything...")
    state = gr.State([])
    txt.submit(respond, [txt, state], [chatbot_ui, state])
demo.launch()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/29363766-a24b-461c-9bf6-1cbe88fea34d)

### RESULT:
Therefore the code is excuted successfully.
