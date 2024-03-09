import logging

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import iirfilter, sosfiltfilt, sosfreqz

from .dataclasses import JointData


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


def process_joint_data(
    joint_data: JointData,
    num_endpoints_to_remove: int = 1,
    compute_velocities: bool = True,
    filter_positions: bool = False,
    pos_filter_order: int = 20,
    pos_cutoff_freq_hz: float = 30.0,
    vel_filter_order: int = 20,
    vel_cutoff_freq_hz: float = 2.0,
    acc_filter_order: int = 20,
    acc_cutoff_freq_hz: float = 2.0,
    torque_filter_order: int = 12,
    torque_cutoff_freq_hz: float = 1.6,
) -> JointData:
    """
    Process joint data by removing endpoints, filtering velocities, estimating
    accelerations, and filtering torques.

    Args:
        joint_data (JointData): The joint data to process.
        num_endpoints_to_remove (int, optional): The number of endpoints to remove from
            the beginning and end of the trajectory. This is useful as the sample times
            are not always increasing with the same period at the beginning and end of
            the trajectory.
        compute_velocities (bool, optional): Whether to compute velocities from the
            positions rather than taking the ones in `joint_data`.
        filter_positions (bool, optional): Whether to filter the joint positions.
        pos_filter_order (int, optional): The order of the filter for the joint
            positions.
        pos_cutoff_freq_hz (float, optional): The cutoff frequency of the filter for the
            joint positions.
        vel_filter_order (int, optional): The order of the filter for the joint
            velocities.
        vel_cutoff_freq_hz (float, optional): The cutoff frequency of the filter for the
            joint velocities.
        acc_filter_order (int, optional): The order of the filter for the joint
            accelerations.
        acc_cutoff_freq_hz (float, optional): The cutoff frequency of the filter for the
            joint accelerations.
        torque_filter_order (int, optional): The order of the filter for the joint
            torques.
        torque_cutoff_freq_hz (float, optional): The cutoff frequency of the filter for
            the joint torques.

    Returns:
        JointData: The processed joint data.
    """
    # Remove beginning and end points to ensure sample times are increasing with the
    # same period throughout to enable differentiation
    joint_positions = joint_data.joint_positions[
        num_endpoints_to_remove:-num_endpoints_to_remove
    ]
    if not compute_velocities:
        joint_velocities = joint_data.joint_velocities[
            num_endpoints_to_remove:-num_endpoints_to_remove
        ]
    joint_torques = joint_data.joint_torques[
        num_endpoints_to_remove:-num_endpoints_to_remove
    ]
    sample_times_s = joint_data.sample_times_s[
        num_endpoints_to_remove:-num_endpoints_to_remove
    ]

    # Process the joint data
    sample_period = sample_times_s[1] - sample_times_s[0]
    logging.info(f"Sample period: {sample_period} seconds.")
    sample_freq = 1.0 / sample_period

    if filter_positions:
        # Filter position data
        filtered_joint_positions = filter_time_series_data(
            data=joint_positions,
            order=pos_filter_order,
            cutoff_freq_hz=pos_cutoff_freq_hz,
            fs_hz=sample_freq,
        )
    else:
        filtered_joint_positions = joint_positions

    if compute_velocities:
        # Estimate velocities using finite differences
        joint_velocities = np.gradient(filtered_joint_positions, sample_period, axis=0)

    # Filter velocity data
    filtered_velocity_data = filter_time_series_data(
        data=joint_velocities,
        order=vel_filter_order,
        cutoff_freq_hz=vel_cutoff_freq_hz,
        fs_hz=sample_freq,
    )

    # Estimate accelerations using finite differences
    joint_accelerations = np.gradient(filtered_velocity_data, sample_period, axis=0)
    filtered_joint_accelerations = filter_time_series_data(
        data=joint_accelerations,
        order=acc_filter_order,
        cutoff_freq_hz=acc_cutoff_freq_hz,
        fs_hz=sample_freq,
    )

    # Filter torque data
    filtered_joint_torques = filter_time_series_data(
        data=joint_torques,
        order=torque_filter_order,
        cutoff_freq_hz=torque_cutoff_freq_hz,
        fs_hz=sample_freq,
    )

    processed_joint_data = JointData(
        joint_positions=joint_positions,
        joint_velocities=filtered_velocity_data,
        joint_accelerations=filtered_joint_accelerations,
        joint_torques=filtered_joint_torques,
        sample_times_s=sample_times_s,
    )
    return processed_joint_data
