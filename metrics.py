import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from networks import define_network
from torch import nn

# Import Config - Update if needed
try:
    from Config import Config  # If this raises ImportError, check your file name (it could be config.py)
except ImportError:
    from config import Config

class AesFA_test(nn.Module):
    def __init__(self, config):
        super(AesFA_test, self).__init__()

        # Ensure the config has necessary attributes
        if not hasattr(config, 'alpha_in'):
            raise AttributeError("Config object is missing attribute 'alpha_in'. Check if you're passing the correct config instance.")

        self.netE = define_network(net_type="Encoder", config=config)
        self.netS = define_network(net_type="Encoder", config=config)
        self.netG = define_network(net_type="Generator", config=config)

        self.ssim = SSIM(data_range=1.0)
        self.lpips = LPIPS(net_type='vgg')

    def forward(self, real_A, real_B):
        with torch.no_grad():
            content_A = self.netE.forward_test(real_A, 'content')
            style_B = self.netS.forward_test(real_B, 'style')
            trs_AtoB = self.netG.forward_test(content_A, style_B)

        return trs_AtoB

    def compute_metrics(self, real_A, real_B, fake_B):
        """ Computes LPIPS and SSIM metrics """

        # Normalize the inputs to [-1, 1] for LPIPS
        def normalize(tensor):
            return torch.clamp(tensor, min=-1, max=1)  # Ensures values stay in range [-1,1]

        fake_B_norm = normalize(fake_B)
        real_B_norm = normalize(real_B)

        ssim_score = self.ssim(fake_B, real_B)
        lpips_score = self.lpips(fake_B_norm, real_B_norm)  # LPIPS requires normalized input

        return {
            'SSIM': ssim_score.item(),
            'LPIPS': lpips_score.item()
        }

# Example usage
if __name__ == "__main__":
    config = Config()  # Create an instance of Config

    # Define device (fixes 'Config' object has no attribute 'device' error)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AesFA_test(config).to(device)

    # Dummy tensors for testing
    real_A = torch.randn(1, 3, 256, 256).to(device)  # Content image
    real_B = torch.randn(1, 3, 256, 256).to(device)  # Style image

    # Normalize inputs to [0, 1] before passing to the model
    real_A = torch.clamp(real_A, min=-1, max=1)  # Scale to [-1, 1]
    real_B = torch.clamp(real_B, min=-1, max=1)  # Scale to [-1, 1]

    fake_B = model(real_A, real_B)

    metrics = model.compute_metrics(real_A, real_B, fake_B)
    print(metrics)
