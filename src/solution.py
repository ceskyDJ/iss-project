import shutil
import time
from typing import List

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
from matplotlib import ticker
from matplotlib.figure import Figure
from scipy.signal import spectrogram, chirp, buttord, butter, lfilter, tf2zpk, freqz


def graph(fun: callable, output_file: str, x_axis, y_axis, x_label: str = None, y_label: str = None, title: str = None):
    plt.figure()
    fun(x_axis, y_axis)

    axes: matplotlib.axes.Axes
    axes = plt.gca()
    if x_label:
        axes.set_xlabel(x_label)
    if y_label:
        axes.set_ylabel(y_label)
    if title:
        axes.set_title(title)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot(output_file: str, x_axis, y_axis, x_label: str = None, y_label: str = None, title: str = None):
    graph(plt.plot, output_file, x_axis, y_axis, x_label, y_label, title)


def stem(output_file: str, x_axis, y_axis, x_label: str = None, y_label: str = None, title: str = None):
    graph(plt.stem, output_file, x_axis, y_axis, x_label, y_label, title)


def dft(signal: np.array) -> np.array:
    # Prepare base matrix
    # noinspection PyPep8Naming
    N = signal.size
    w = np.exp(-2j * np.pi / N)

    base_matrix = []
    for i in range(0, N):
        row = []
        for j in range(0, N):
            row.append(np.power(w, i * j))
        base_matrix.append(row)

    base_matrix = np.array(base_matrix)

    return np.dot(base_matrix, signal)


def nice_time(nanoseconds: int) -> str:
    units = ["s", "ms", "μs", "ns"]
    big_units = ["h", "min"]

    exp = 9
    i = 0
    while nanoseconds / 10 ** exp < 1:
        exp -= 3
        i += 1

    if nanoseconds / 10 ** 9 < 60:
        return str(np.round(nanoseconds / 10 ** exp, 2)) + f" {units[i]}"
    else:
        seconds = nanoseconds / 10 ** 9

    exp = 2
    i = 0
    while seconds / 60 ** exp < 1:
        exp -= 1
        i += 1

    return str(np.round(seconds / 60 ** exp, 2)) + f" {big_units[i]}"


def process(src_file: str, audio_dir: str, img_dir: str) -> None:
    # Task 1
    print("Task 1\n======")

    # Load WAV
    signal: np.ndarray
    sample_rate: int
    signal, sample_rate = sf.read(src_file)

    # Samples and length
    samples = signal.size
    length = samples / sample_rate
    time_axis = np.arange(samples) / sample_rate

    # Minimum and maximum values
    # noinspection PyArgumentList
    min_val = signal.min()
    # noinspection PyArgumentList
    max_val = signal.max()

    # Print counted information
    print(f"Min value: {min_val}\nMax value: {max_val}")
    print(f"Sample rate (Hz): {sample_rate}")
    print(f"Samples: {samples}\nLength (s): {length}")

    # Draw input signal
    plot(f"{img_dir}/01-input-signal.pdf", time_axis, signal, "Čas $[s]$")

    # Task 2
    frame_size = 1024
    frame_overlap = 512

    # Signal concentration
    mid_val = np.mean(signal)
    signal_centered = np.subtract(signal, mid_val)

    # Signal normalization
    signal_abs = np.abs(signal_centered)
    abs_max_val = np.max(signal_abs)
    signal_norm = np.divide(signal_centered, abs_max_val)

    # Divide into frames
    frames = []
    for frame_num in range(round(samples / frame_overlap)):
        frame = np.array(signal_norm[frame_num * frame_overlap:frame_num * frame_overlap + frame_size])
        frame = np.pad(frame, (0, frame_size - frame.size), 'constant')
        frames.append(frame)
    frames_cols = np.column_stack(frames)

    # Draw all frames
    i = 0
    for frame in frames:
        plt.figure()

        frame_samples = frame.size
        time_axis = np.arange(frame_samples) / sample_rate

        plt.plot(time_axis, frame)

        axes: matplotlib.axes.Axes
        axes = plt.gca()
        axes.set_xlabel("Čas $[s]$")

        plt.savefig(f"{img_dir}/frames/frame-{i}.pdf", bbox_inches="tight", pad_inches=0)
        plt.close()
        i += 1

    # Copy the best frame
    # Chosen manually
    shutil.copy(f"{img_dir}/frames/frame-42.pdf", f"{img_dir}/02-nice-frame.pdf")

    # Task 3
    print("\nTask 3\n======")

    my_dft_start = time.time_ns()
    nice_frame_my_dft = dft(frames_cols[:, 43])
    my_dft_end = time.time_ns()
    print("My DFT time: " + nice_time(my_dft_end - my_dft_start))

    fft_start = time.time_ns()
    nice_frame_fft = np.fft.fft(frames_cols[:, 43])
    fft_end = time.time_ns()
    print("FFT time: " + nice_time(fft_end - fft_start))

    freq = (np.arange(frame_size) / frame_size) * sample_rate

    fig: Figure
    axes: List[matplotlib.axes.Axes]
    fig, axes = plt.subplots(2, 1)
    used_samples = frame_size // 2

    axes[0].plot(freq[:used_samples], np.abs(nice_frame_my_dft[:used_samples]))
    axes[0].set_title("DFT pomocí vlastní funkce")
    axes[0].set_xlabel("Frekvence $[Hz]$")

    axes[1].plot(freq[:used_samples], np.abs(nice_frame_fft[:used_samples]))
    axes[1].set_title("DFT pomocí np.fft.fft()")
    axes[1].set_xlabel("Frekvence $[Hz]$")

    fig.tight_layout()
    fig.savefig(f"{img_dir}/03-dft-module.pdf", bbox_inches="tight", pad_inches=0)

    # Task 4
    freq, time_axis, sgr = spectrogram(signal, sample_rate, nperseg=frame_size, noverlap=frame_overlap)
    sgr_log = 10 * np.log10(sgr + 1e-20)

    plt.figure(figsize=(9, 5))

    plt.pcolormesh(time_axis, freq, sgr_log)
    plt.gca().set_xlabel('Čas $[s]$')
    plt.gca().set_ylabel('Frekvence $[Hz]$')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu $[dB]$', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{img_dir}/04-spectrogram.pdf", bbox_inches="tight", pad_inches=0)

    # Task 5
    print("\nTask 5\n======")
    # Semiautomatically from the nice frame
    interesting_samples = 256

    modules = np.abs(nice_frame_fft[:interesting_samples])

    peaks, _ = scipy.signal.find_peaks(np.abs(nice_frame_fft[:interesting_samples]), height=5)
    peaks_freq = (peaks / frame_size) * sample_rate

    plot(f"{img_dir}/05-nice-frame-dft.pdf", freq[:interesting_samples], modules, "Frekvence $[Hz]$")

    print("Peak values:")
    i = 0
    for peak in peaks:
        axes[0].annotate(f"{i}", (peaks_freq[i], modules[peak]), color="red")

        print(f" - {i}: {peaks_freq[i]}")
        i += 1

    # Manually from spectrogram
    fig: Figure
    axes: List[matplotlib.axes.Axes]
    fig, axes = plt.subplots(4, 1, figsize=(9, 5))

    freq, time_axis, sgr = spectrogram(signal, sample_rate, nperseg=frame_size, noverlap=frame_overlap)
    sgr_log = 10 * np.log10(sgr + 1e-20)

    axes[0].pcolormesh(time_axis, freq, sgr_log)
    axes[0].set_xlabel('Čas $[s]$')
    axes[0].set_ylabel('Frekvence $[Hz]$')
    axes[0].set_ylim(bottom=600, top=700)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(25))

    axes[1].pcolormesh(time_axis, freq, sgr_log)
    axes[1].set_xlabel('Čas $[s]$')
    axes[1].set_ylabel('Frekvence $[Hz]$')
    axes[1].set_ylim(bottom=1300, top=1400)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(25))

    axes[2].pcolormesh(time_axis, freq, sgr_log)
    axes[2].set_xlabel('Čas $[s]$')
    axes[2].set_ylabel('Frekvence $[Hz]$')
    axes[2].set_ylim(bottom=1950, top=2050)
    axes[2].yaxis.set_major_locator(ticker.MultipleLocator(25))

    axes[3].pcolormesh(time_axis, freq, sgr_log)
    axes[3].set_xlabel('Čas $[s]$')
    axes[3].set_ylabel('Frekvence $[Hz]$')
    axes[3].set_ylim(bottom=2650, top=2750)
    axes[3].yaxis.set_major_locator(ticker.MultipleLocator(25))

    fig.tight_layout()
    fig.savefig(f"{img_dir}/05-detailed-spectrogram.pdf", bbox_inches="tight", pad_inches=0)

    cos_frequencies = [675, 1350, 2025, 2700]
    print("Cosine frequencies:\n"
          "- f1 = 675 (1x)\n"
          "- f2 = 1350 (2x)\n"
          "- f3 = 2025 (3x)\n"
          "- f4 = 2700 (4x)")

    # Task 6
    fig: Figure
    axes: List[matplotlib.axes.Axes]
    fig, axes = plt.subplots(len(cos_frequencies), 1, figsize=(9, 10))

    cos_signal = []
    i = 0
    for cos_freq in cos_frequencies:
        time_axis = np.linspace(0, length, samples)
        gen_signal = chirp(time_axis, f0=cos_freq, f1=cos_freq, t1=length, method='linear')

        if len(cos_signal) == 0:
            cos_signal = gen_signal
        else:
            cos_signal = np.add(cos_signal, gen_signal)

        axes[i].plot(time_axis, gen_signal)
        axes[i].set_title(f"Cos signál s frekvencí ${cos_freq} Hz$")
        axes[i].set_xlabel("Čas $[s]$")

        i += 1
    cos_signal = np.array(cos_signal)

    # Spectrogram
    fig.tight_layout()
    fig.savefig(f"{img_dir}/06-cos-signals-divided.pdf", bbox_inches="tight", pad_inches=0)

    freq, time_axis, sgr = spectrogram(cos_signal, sample_rate, nperseg=frame_size, noverlap=frame_overlap)
    sgr_log = 10 * np.log10(sgr + 1e-20)

    plt.figure(figsize=(9, 5))

    plt.pcolormesh(time_axis, freq, sgr_log)
    plt.gca().set_xlabel('Čas $[s]$')
    plt.gca().set_ylabel('Frekvence $[Hz]$')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu $[dB]$', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{img_dir}/06-cos-signals-spectrum.pdf", bbox_inches="tight", pad_inches=0)

    # Save WAV
    sf.write(f"{audio_dir}/4cos.wav", cos_signal, sample_rate)

    # Task 7
    print("\nTask 7\n======")

    nyquist_freq = sample_rate / 2

    # Create stop-band filters
    filters = []
    for cos_freq in cos_frequencies:
        lowest_ord, wn = buttord(
            wp=[(cos_freq - 50) / nyquist_freq, (cos_freq + 50) / nyquist_freq],
            ws=[(cos_freq - 15) / nyquist_freq, (cos_freq + 15) / nyquist_freq],
            gpass=3, gstop=40
        )

        b, a = butter(lowest_ord, wn, 'bandstop', output='ba')
        result_filter = (b, a)
        filters.append(result_filter)
    filters = np.array(filters)

    print("Coefficients of filters:")
    print(filters)

    # Impulse response
    imp_res_samples = 64

    fig: Figure
    axes: List[matplotlib.axes.Axes]
    fig, axes = plt.subplots(len(filters), 1, figsize=(10, 12))

    i = 0
    for (b, a) in filters:
        unit_pulse = [1, *np.zeros(imp_res_samples - 1)]
        imp_res = lfilter(b, a, unit_pulse)

        axes[i].stem(np.arange(imp_res_samples), imp_res, basefmt=' ', use_line_collection=True)

        axes[i].set_xlabel('$n$')
        axes[i].set_title(f"Filtr pro frekvenci ${cos_frequencies[i]}\\ Hz$")
        axes[i].grid(alpha=0.5, linestyle='--')

        i += 1

    fig.tight_layout()
    fig.savefig(f"{img_dir}/07-impulse-responses.pdf", bbox_inches="tight", pad_inches=0)

    # Task 8
    print("\nTask 8\n======")

    fig: Figure
    axes: List[matplotlib.axes.Axes]
    fig, axes = plt.subplots(len(filters), 1, figsize=(4, 3.5 * len(filters)))

    i = 0
    for (b, a) in filters:
        zeros, poles, _ = tf2zpk(b, a)

        ang = np.linspace(0, 2 * np.pi, 100)
        axes[i].plot(np.cos(ang), np.sin(ang))

        axes[i].scatter(np.real(zeros), np.imag(zeros), marker="o", facecolors="none", edgecolors="r", label="nuly")
        axes[i].scatter(np.real(poles), np.imag(poles), marker="x", color="g", label="póly")

        axes[i].set_xlabel("Realná složka")
        axes[i].set_ylabel("Imaginarní složka")
        axes[i].grid(alpha=0.5, linestyle="--")
        axes[i].legend(loc="upper right")

        is_stable = (poles.size == 0) or np.all(np.abs(poles) < 1)

        print(f"Filter for {cos_frequencies[i]} Hz:")
        print(f"- Zeros: {zeros}")
        print(f"- Poles: {poles}")
        print(" - Filter is" + (" not" if not is_stable else "") + " stable")
        i += 1

    fig.tight_layout()
    fig.savefig(f"{img_dir}/08-zeros-poles.pdf", bbox_inches="tight", pad_inches=0)

    # Task 9
    fig: Figure
    fig, axes = plt.subplots(len(filters), 2, figsize=(8, 3 * len(filters)))

    i = 0
    for (b, a) in filters:
        freq, freq_res = freqz(b, a)

        axes[i][0].plot(freq / 2 / np.pi * sample_rate, np.abs(freq_res))
        axes[i][0].set_xlabel("Frekvence $[Hz]$")
        axes[i][0].set_title(f"Modul frekv. char. filtru pro ${cos_frequencies[i]}\\ Hz$")

        axes[i][1].plot(freq / 2 / np.pi * sample_rate, np.angle(freq_res))
        axes[i][1].set_xlabel("Frekvence $[Hz]$")
        axes[i][1].set_title(f"Arg. frekv. char. filtru pro ${cos_frequencies[i]}\\ Hz$")

        axes[i][0].grid(alpha=0.5, linestyle='--')
        axes[i][1].grid(alpha=0.5, linestyle='--')

        i += 1

    fig.tight_layout()
    fig.savefig(f"{img_dir}/09-freq-character.pdf", bbox_inches="tight", pad_inches=0)

    # Task 10
    filtered_signal = signal
    time_axis = np.arange(samples) / sample_rate

    # Apply filters
    for (b, a) in filters:
        filtered_signal = lfilter(b, a, filtered_signal)

    plot(f"{img_dir}/10-filtered-signal.pdf", time_axis, filtered_signal, "Čas $[s]$")

    # Save filtered signal
    sf.write(f"{audio_dir}/clean_bandstop.wav", filtered_signal, sample_rate)

    # Draw final spectrogram
    freq, time_axis, sgr = spectrogram(filtered_signal, sample_rate, nperseg=frame_size, noverlap=frame_overlap)
    sgr_log = 10 * np.log10(sgr + 1e-20)

    plt.figure(figsize=(9, 5))

    plt.pcolormesh(time_axis, freq, sgr_log)
    plt.gca().set_xlabel('Čas $[s]$')
    plt.gca().set_ylabel('Frekvence $[Hz]$')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu $[dB]$', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f"{img_dir}/10-final-spectrogram.pdf", bbox_inches="tight", pad_inches=0)


def main() -> None:
    process("../audio/xsmahe01.wav", "../audio", "../img")


if __name__ == "__main__":
    main()
