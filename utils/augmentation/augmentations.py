import torchvision.transforms.v2 as T


class AugmentationRawEEG:
    def __init__(self):

        self.transform = T.Compose(
            [
                T.RandomGaussianNoise(noise_std=0.01, p=0.5),
                T.RandomAmplitudeScaling(min_scale=0.9, max_scale=1.1, p=0.5),
                T.RandomTimeShift(max_shift=0.5, p=0.3),
                T.RandomTimeMask(mask_ratio=0.1, p=0.3),
            ]
        )

    def __call__(self, eeg):
        return self.transform(eeg)
