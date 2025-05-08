import os
import argparse
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

######### VAE MODEL ##########
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # encoder:some convolutional layers that shrink image and extract features
        # conv layers to get features
        # input: 3x512x512
        # output: 256x32x32
        # 3 channels to 32 channels, 4x4 kernel, stride 2, padding 1
        # 512/2 = 256, 256/2 = 128, 128/2 = 64, 64/2 = 32   
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Flatten()
        )
        # two linear layers to get mean and logvar of gaussian
        # fully connected layers to get mean and logvar
        self.fc_mu     = nn.Linear(256*32*32, latent_dim)
        self.fc_logvar = nn.Linear(256*32*32, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256*32*32)

        # decoder part: deconv to bring back to image
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 32, 32)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,  3, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        # sample from latent
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode, sample, decode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        # h: encoded features (batch, 256*32*32)
        mu = self.fc_mu(h)
        # mu: mean of latent (batch, latent_dim)
        lv = self.fc_logvar(h)
        # lv: logvar of latent (batch, latent_dim)
        z = self.reparameterize(mu, lv)
        # z: sampled latent (batch, latent_dim)
        d = self.fc_decode(z)
        d = d.view(-1, 256, 32, 32)
        # d: decoded features (batch, 256, 32, 32)
        out = self.decoder(d)
        # out: reconstructed image (batch, channels, height, width)
        return out, mu, lv





#3##########LOSS FUNCTOIN###########
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    #1. reconstruction_loss: how well the decoder output matches the input
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #2. kl_divergence: how close the latent distribution q(z|x) is to the prior N(0,1)
    return recon_loss + kld






############ LOADING DATSET###########
class ImpressionismDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = [os.path.join(root_dir, f)
                      for f in os.listdir(root_dir)
                      if f.lower().endswith(('png','jpg','jpeg'))]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img






######### TRAINNING #########
def train_model(args, device):
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    dataset = ImpressionismDataset(args.data_dir, transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)
    model = VAE(latent_dim=args.latent_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float('inf')
    loss_history = []
    os.makedirs(args.output_dir, exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for imgs in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total/len(dataset)
        loss_history.append(avg)
        print(f"Epoch {ep}/{args.epochs}  Avg Loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), args.model_path)
            print(f"  >> Saved best model (loss {best_loss:.4f})")
            
    # Plot and save training loss curve
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()
    print("Training complete.")







####### STYLIZING IMAGES###########
# loads input images, applies the trained VAE model to stylize them, and saves the output images
def stylize_images(args, device):
    model = VAE(latent_dim=args.latent_dim).to(device)
    # load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform_in = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    
    # PIL library to handle image loading and saving    
    to_pil = transforms.ToPILImage()
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('png','jpg','jpeg')): continue
        img = Image.open(os.path.join(args.input_dir, fname)).convert('RGB')
        x = transform_in(img).unsqueeze(0).to(device)
        with torch.no_grad(): recon,_,_ = model(x)
        out = recon.squeeze(0).cpu()
        pil = to_pil(out)
        pil.save(os.path.join(args.output_dir, fname))
        print(f"Stylized: {fname}")
    print("All images stylized.")







############# MAIN FUNCTION ############3
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',      choices=['train','stylize'], required=True)
    p.add_argument('--data_dir',  type=str, default='data/wikiart/Impressionism')
    p.add_argument('--input_dir', type=str, default='input')
    p.add_argument('--output_dir',type=str, default='output')
    p.add_argument('--model_path',type=str, default='vae_impressionism.pt')
    p.add_argument('--epochs',    type=int, default=100)
    p.add_argument('--batch_size',type=int, default=32)
    p.add_argument('--latent_dim',type=int, default=256)
    p.add_argument('--lr',        type=float, default=1e-4)
    args = p.parse_args()
    
    # device
    if torch.backends.mps.is_available(): device = torch.device('mps')
    elif torch.cuda.is_available():      device = torch.device('cuda')
    else:                                device = torch.device('cpu')
    print(f"using device: {device}")
    
    # run
    if args.mode == 'train':
        train_model(args, device)
    else:
        stylize_images(args, device)

if __name__ == '__main__':
    main()
