import numpy as np

__all__ = [
    "RandomAmplitudeScaling",
    "RandomChannelDropout",
    "RandomChannelMask",
    "RandomGaussianNoise",
    "RandomTimeMask",
    "RandomTimeShift",
]


class RandomGaussianNoise:
    def __init__(self, noise_std=0.01, p=0.5):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=signal.shape)
        return signal + noise


class RandomAmplitudeScaling:
    def __init__(self, min_scale=0.9, max_scale=1.1, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return signal * scale


class RandomTimeShift:
    def __init__(self, max_shift, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(signal, shift, axis=-1)


class RandomTimeMask:
    def __init__(self, mask_ratio=0.1, p=0.5):
        self.mask_ratio = mask_ratio
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        signal = signal.copy()
        T = signal.shape[-1]
        mask_length = int(T * self.mask_ratio)
        if mask_length == 0:
            return signal
        start = np.random.randint(0, T - mask_length + 1)
        signal[..., start : start + mask_length] = 0
        return signal


class RandomChannelMask:
    def __init__(self, mask_probability=0.1, p=0.5):
        self.mask_probability = mask_probability
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        signal = signal.copy()
        C = signal.shape[0]
        for ch in range(C):
            if np.random.rand() < self.mask_probability:
                signal[ch] = 0
        return signal


class RandomChannelDropout:
    def __init__(self, num_channels=5, p=0.5):
        self.num_channels = num_channels
        self.p = p

    def __call__(self, signal):
        if np.random.rand() >= self.p:
            return signal
        signal = signal.copy()
        C = signal.shape[0]
        num_channels = min(self.num_channels, C)
        channels = np.random.choice(C, num_channels, replace=False)
        signal[channels] = 0
        return signal
