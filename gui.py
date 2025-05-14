"""
Gradio UI for impressionist art generation using a VAE model.

To use: 
1. $ pip install gradio pillow
2. and run this script.
3. go to link http://127.0.0.1:7860 unless you specify a different port
"""

import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

# model part
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from run import VAE
model = VAE(latent_dim=256)
state_dict = torch.load("./vae_impressionism.pt",map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval().to(DEVICE)


# I/O part
IMSIZE = 512
preprocess = transforms.Compose([
    transforms.Resize((IMSIZE, IMSIZE), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
])
postprocess = transforms.ToPILImage() 



# inference wrapper
@torch.inference_mode()
def monetify(pil_img: Image.Image) -> Image.Image:
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    recon, _, _ = model(x)
    return postprocess(recon.squeeze(0).cpu())

#gradio UI
demo = gr.Interface(
    fn=monetify,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Image(type="pil", label="Monet-style Impressionist Painting"),
    title="Impressionist Art Generator",
    description="Drag & drop any image and get a Monet-style impressionist painting!",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=False)   # share=True for a public link