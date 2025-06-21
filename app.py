import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# === Generator Class (must match your training script) ===
class Generator(nn.Module):
    def __init__(self, nz=100, nc=1, ngf=64, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, nz)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded = self.label_embed(labels)
        z = z + embedded
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

# === Load model ===
@st.cache_resource
def load_model():
    model = Generator().cpu()
    model.load_state_dict(torch.load("dcgan_generator.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === UI ===
st.title("🖊️ MNIST Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate 5 Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        generated = model(z, labels).cpu()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(generated[i][0].numpy(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
