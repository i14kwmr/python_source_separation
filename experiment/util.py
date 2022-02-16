import wave as wave

import librosa
import numpy as np


def resample_signals(conv_data, sample_rate, sro):
    print(f"sample rate: {sample_rate}")
    print(f"resample rate: {sample_rate * (1+sro)}")

    n_channels = conv_data.shape[0]
    max_samples = conv_data.shape[1]

    _conv_data = []
    for i in range(n_channels):
        if i == 0:
            fs_mic = sample_rate
        else:
            fs_mic = sample_rate * (1 + sro)

        resr_data = librosa.resample(
            conv_data[i, :],
            sample_rate,
            fs_mic,
            res_type="kaiser_best",
        )
        _conv_data.append(resr_data)

        if len(resr_data) < max_samples:
            max_samples = len(resr_data)

    for i in range(n_channels):
        _conv_data[i] = _conv_data[i][:max_samples]

    conv_data = np.stack(_conv_data, axis=0)  # nsrc x nsamples

    # [TODO] 信号長が短くなる理由の調査．FFTで切り捨てが原因？

    return conv_data


# 2バイトに変換してファイルに保存
# signal: time-domain 1d array (float)
# file_name: 出力先のファイル名
# sample_rate: サンプリングレート
def write_file_from_time_signal(signal, file_name, sample_rate):
    # 2バイトのデータに変換
    signal = signal.astype(np.int16)

    # waveファイルに書き込む
    wave_out = wave.open(file_name, "w")

    # モノラル:1、ステレオ:2
    wave_out.setnchannels(1)

    # サンプルサイズ2byte
    wave_out.setsampwidth(2)

    # サンプリング周波数
    wave_out.setframerate(sample_rate)

    # データを書き込み
    wave_out.writeframes(signal)

    # ファイルを閉じる
    wave_out.close()


# SNRをはかる
# desired: 目的音、Lt
# out:　雑音除去後の信号 Lt
def calculate_snr(desired, out):
    wave_length = np.minimum(np.shape(desired)[0], np.shape(out)[0])

    # 消し残った雑音
    desired = desired[:wave_length]
    out = out[:wave_length]
    noise = desired - out
    snr = 10.0 * np.log10(np.sum(np.square(desired)) / np.sum(np.square(noise)))

    return snr
