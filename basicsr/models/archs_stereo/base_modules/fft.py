import torch
from torch import nn


class FFTFrequencySplit(nn.Module):
    def __init__(self, cutoff_ratio=0.5):
        """
        FrequencySplitNN using FFT to split into low-frequency and high-frequency components.

        Args:
            cutoff_ratio (float): Ratio for frequency cutoff, determining how much of the
                                  low-frequency and high-frequency components to keep.
        """
        super(FFTFrequencySplit, self).__init__()
        self.cutoff_ratio = cutoff_ratio
        # self.ratio_compute = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
        # )

    def forward(self, x):
        """
        Forward pass for splitting features into high- and low-frequency components using FFT.

        Args:
            x (torch.Tensor): Input feature maps of shape (batch_size, in_channels, height, width).

        Returns:
            low_freq (torch.Tensor): Low-frequency component of the features.
            high_freq (torch.Tensor): High-frequency component of the features.
        """
        # Apply FFT to convert feature maps to the frequency domain
        fft_features = torch.fft.fft2(x, norm="ortho")

        # Shift the zero-frequency component to the center
        fft_features_shifted = torch.fft.fftshift(fft_features)

        # Get dimensions of the feature map
        batch_size, channels, height, width = x.shape

        # Create a mask to separate low and high frequencies
        mask = torch.zeros_like(fft_features_shifted)
        cutoff_h = int(self.cutoff_ratio * height / 2)
        cutoff_w = int(self.cutoff_ratio * width / 2)

        # Create low-frequency mask (center of the frequency spectrum)
        mask[:, :, height // 2 - cutoff_h:height // 2 + cutoff_h, width // 2 - cutoff_w:width // 2 + cutoff_w] = 1

        # Separate low and high frequencies
        low_freq_fft = fft_features_shifted * mask  # Low-frequency component
        # high_freq_fft = fft_features_shifted * (1 - mask)  # High-frequency component

        # Shift back the frequencies to the original ordering
        low_freq_fft = torch.fft.ifftshift(low_freq_fft)
        # high_freq_fft = torch.fft.ifftshift(high_freq_fft)

        # Apply inverse FFT to convert back to the spatial domain
        low_freq = torch.fft.ifft2(low_freq_fft, norm="ortho").real
        # high_freq = torch.fft.ifft2(high_freq_fft, norm="ortho").real
        high_freq = x- low_freq
        return low_freq, high_freq