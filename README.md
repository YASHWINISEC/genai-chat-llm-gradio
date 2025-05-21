## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### DESIGN STEPS:

#### STEP 1:
Install Libraries: Install the necessary Python packages: diffusers, transformers, gradio, torch, torchvision, and accelerate using pip.
#### STEP 2:
Import Modules: Import torch, StableDiffusionPipeline from diffusers, and gradio as gr.
#### STEP 3:
Define Model and Device: Specify the pre-trained model ID (runwayml/stable-diffusion-v1-5) and determine the computation device (cuda if a GPU is available, otherwise cpu).
#### STEP 4:
Load Stable Diffusion Pipeline: Load the pre-trained Stable Diffusion model using StableDiffusionPipeline.from_pretrained(), specifying the model ID, data type based on the device, and enabling safe tensors. Move the pipeline to the selected device.
#### STEP 5:
Enable Attention Slicing: Optimize memory usage by enabling attention slicing.
#### STEP 6:
Define Image Generation Function: Create a function generate_image that takes a prompt as input, uses the loaded pipe to generate an image with specified height and width, and returns the generated image.
#### STEP 7:
Create Gradio Interface: Define a Gradio Interface with the generate_image function as the core, a text input for the prompt, and an image output to display the result. Set a title and description for the interface.
#### STEP 8:
Launch Gradio App: Launch the Gradio interface, making it publicly accessible via a shareable link.

### PROGRAM:
```
!pip install diffusers transformers gradio torch torchvision accelerate safetensors

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
).to(device)

pipe.enable_attention_slicing()

def generate_image(prompt):
    image = pipe(prompt, height=512, width=512).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your image prompt", placeholder="e.g., A castle in the clouds at sunset"),
    outputs=gr.Image(label="Generated Image"),
    title="AI Image Generator",
    description="Generate images from your imagination using Stable Diffusion v1.5 + Gradio UI"
)
demo.launch(share=True)
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/19cd717d-79f7-4f3e-837d-79f3b011b4d4)

### RESULT:
Therefore the code is excuted successfully.
