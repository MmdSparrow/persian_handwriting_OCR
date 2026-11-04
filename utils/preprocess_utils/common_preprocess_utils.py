import mne

__all__ = [
    "convert_data_to_mne_raw_array_by_montage_name",
]


def convert_data_to_mne_raw_array_by_montage_name(data, sfreq, montage_name, ch_names=None):
    montage = mne.channels.make_standard_montage(montage_name)
    if ch_names is None:
        ch_names = montage.ch_names
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
        verbose=False,
    )
    eeg = mne.io.RawArray(data, info, verbose=False)
    eeg.set_montage(montage, verbose=False)
    return eeg
