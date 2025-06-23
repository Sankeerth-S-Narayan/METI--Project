import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# === Generator Class (Fully Connected, Matches Training Code) ===
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_embed(labels)
        x = torch.cat([z, label_input], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# === Load model ===
@st.cache_resource
def load_model():
    model = Generator().cpu()
    model.load_state_dict(torch.load("models/dcgan_generator.pth", map_location="cpu"))
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
