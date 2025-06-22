import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# 1️⃣ CVAE Model Definition (should match your training code)
latent_dim = 20
num_classes = 10
img_dim = 28 * 28

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(img_dim + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, img_dim)

    def encode(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 2️⃣ Load the trained model
model = CVAE().to(device)
model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
model.eval()

# -----------------------
# 3️⃣ Streamlit App UI
st.title("Generate Handwritten Digits using CVAE")
st.write("Choose a digit (0-9), and this app will generate 5 **unique** handwritten samples conditioned on that digit.")

chosen_digit = st.selectbox("Choose a digit to generate:", list(range(10)))

if st.button("Generate 5 Images"):
    with torch.no_grad():
        y = F.one_hot(torch.full((5,), chosen_digit), num_classes).float().to(device)
        z = torch.randn(5, latent_dim).to(device)  # 5 DIFFERENT latent vectors
        generated = model.decode(z, y).cpu().view(-1, 28, 28).numpy()

        # Display the generated images side-by-side
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axs[i].imshow(generated[i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Digit: {chosen_digit}")

        st.pyplot(fig)
