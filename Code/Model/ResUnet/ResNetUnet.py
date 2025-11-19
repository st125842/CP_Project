import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Creates a Residual Block as seen in the diagram.
    The shortcut connection is taken *after* the first convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        # First convolution (3x3), which might change the number of channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Second convolution (3x3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # First conv + relu
        x1 = self.relu(self.conv1(x))
        
        # Shortcut connection is taken from the output of the first conv
        shortcut = x1
        
        # Second conv
        x2 = self.conv2(x1)
        
        # Element-wise Add
        x_out = torch.add(x2, shortcut)
        x_out = self.relu(x_out)
        return x_out

class ResUNet(nn.Module):
    """
    Builds the Res-U-Net model based on the diagram.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(ResUNet, self).__init__()
        
        # === Encoder (Contracting Path) ===
        
        # Initial 3x3 Conv from 3 -> 64 channels
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.init_relu = nn.ReLU()
        
        # L1 (64 filters)
        self.enc1_res = ResBlock(64, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # L2 (128 filters)
        self.enc2_res = ResBlock(64, 128)
        
        # L3 Bottleneck (256 filters)
        self.enc3_res = ResBlock(128, 256)
        
        # === Decoder (Expanding Path) ===
        
        # L2 (128 filters)
        # 2x2 Deconvolution (Upsampling)
        self.dec2_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Residual Block (128 from upsample + 128 from skip = 256)
        self.dec2_res = ResBlock(256, 128)
        
        # L1 (64 filters)
        # 2x2 Deconvolution (Upsampling)
        self.dec1_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Residual Block (64 from upsample + 64 from skip = 128)
        self.dec1_res = ResBlock(128, 64)
        
        # === Output ===
        
        # Final 3x3 Conv (64 -> 64)
        self.final_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU()
        
        # Final 3x3 Conv (64 -> out_channels)
        self.final_conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.final_act = nn.Tanh() # Sigmoid for output pixels between 0 and 1

    def forward(self, x):
        # === Encoder ===
        c0 = self.init_relu(self.init_conv(x))
        
        # L1
        c1 = self.enc1_res(c0)
        p1 = self.pool(c1)
        
        # L2
        c2 = self.enc2_res(p1)
        p2 = self.pool(c2)
        
        # L3 (Bottleneck)
        b3 = self.enc3_res(p2)
        
        # === Decoder ===
        
        # L2
        d2_up = self.dec2_up(b3)
        # Copy and Concatenate (dim=1 is the channel dimension)
        d2_cat = torch.cat([d2_up, c2], dim=1) 
        c4 = self.dec2_res(d2_cat)
        
        # L1
        d1_up = self.dec1_up(c4)
        # Copy and Concatenate
        d1_cat = torch.cat([d1_up, c1], dim=1)
        c5 = self.dec1_res(d1_cat)
        
        # === Output ===
        c6 = self.final_relu(self.final_conv1(c5))
        outputs = self.final_act(self.final_conv2(c6))
        
        return outputs

# --- Example Usage ---

# Create the model
# (PyTorch expects channels-first format: Batch, Channels, Height, Width)
model = ResUNet(in_channels=3, out_channels=3)

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 256, 256) # 1 image, 3 channels, 256x256 pixels

# Pass the input through the model
output = model(dummy_input)

print(f"Model: Res-U-Net")
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# For a Keras-like model summary, you can use the `torchinfo` library
# Install with: pip install torchinfo
#
# from torchinfo import summary
# summary(model, input_size=(1, 3, 256, 256))