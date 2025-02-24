import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion pipeline (This doesn't require a GPU, but it may take time on CPU)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Streamlit UI
st.title("🖼️ Free AI Image Generator")
st.write("Enter a prompt to generate an image using Stable Diffusion.")

# Text input for user prompt
prompt = st.text_input("Enter your prompt:", placeholder="A futuristic cityscape at sunset")

# Generate the image when button is pressed
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating... Please wait."):
            try:
                # Generate the image using the model
                image = pipe(prompt).images[0]
                st.image(image, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"🚨 Error: {e}")
    else:
        st.warning("⚠️ Please enter a prompt.")
