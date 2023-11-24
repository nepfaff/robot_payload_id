import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import iirfilter, sosfiltfilt, sosfreqz


def filter_time_series_data(
    data: np.ndarray,
    order: int,
    cutoff_freq_hz: float,
    fs_hz: float,
    visualize: bool = False,
) -> np.ndarray:
    """Low-pass filters the time series data using a Butterworth filter (applied once
    forward and once backward to avoid phase shift).

    Args:
        data (np.ndarray): The data to filter. Filtering will occur along the first
        axis.
        order (int): The order of the Butterworth filter (the combined filter will have
        twice the order).
        cutoff_freq_hz (float): The cutoff frequency of the filter in Hz.
        fs_hz (float): The sampling frequency of the data in Hz.
        visualize (bool): Whether to plot the frequency response of the filter.

    Returns:
        np.ndarray: The filtered data.
    """
    sos = iirfilter(
        N=order,
        Wn=cutoff_freq_hz,
        fs=fs_hz,
        btype="lowpass",
        ftype="butter",
        output="sos",
    )

    if visualize:
        w, h = sosfreqz(sos=sos, worN=10000, fs=fs_hz)
        plt.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
        plt.title("Butterworth filter frequency response")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.margins(0, 0.1)
        plt.grid(which="both", axis="both")
        plt.axvline(cutoff_freq_hz, color="green")
        plt.show()

    filtered_data = sosfiltfilt(sos, data, axis=0).copy()
    assert not np.any(
        np.isnan(filtered_data)
    ), "Filtering failed. Try again with a different filter order."

    return filtered_data
