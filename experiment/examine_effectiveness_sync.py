# 順列計算に使用
import argparse
import time

import scipy.signal as sp

from util import *
from util_bss import *
from util_sim import *

parser = argparse.ArgumentParser()
parser.add_argument("sro", type=float, default=0.0)  # 単位はppm
args = parser.parse_args()


def examine_effectiveness_sync():
    # 乱数の種を初期化
    np.random.seed(0)

    # 畳み込みに用いる音声波形
    clean_wave_files = [
        "../CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav",
        "../CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0002.wav",
    ]

    # 音源数
    n_sources = len(clean_wave_files)

    # 長さを調べる
    n_samples = 0
    # ファイルを読み込む
    for clean_wave_file in clean_wave_files:
        wav = wave.open(clean_wave_file)
        if n_samples < wav.getnframes():
            n_samples = wav.getnframes()
        wav.close()

    clean_data = np.zeros([n_sources, n_samples])

    # ファイルを読み込む
    s = 0
    for clean_wave_file in clean_wave_files:
        wav = wave.open(clean_wave_file)
        data = wav.readframes(wav.getnframes())
        data = np.frombuffer(data, dtype=np.int16)
        data = data / np.iinfo(np.int16).max
        clean_data[s, : wav.getnframes()] = data
        wav.close()
        s = s + 1

    sample_rate = 16000  # サンプリング周波数
    N = 1024  # フレームサイズ
    Nk = int(N / 2 + 1)  # 周波数の数

    # シミュレーションのパラメータ
    (
        multi_conv_data,
        multi_conv_data_left_no_noise,
        multi_conv_data_right_no_noise,
    ) = simulate(clean_data, sample_rate)

    # リサンプリング
    sro = args.sro * 1e-6  # 0
    multi_conv_data = resample_signals(multi_conv_data, sample_rate, sro)
    multi_conv_data_left_no_noise = resample_signals(
        multi_conv_data_left_no_noise, sample_rate, sro
    )
    multi_conv_data_right_no_noise = resample_signals(
        multi_conv_data_right_no_noise, sample_rate, sro
    )

    # 畳み込んだ波形をファイルに書き込む
    write_file_from_time_signal(
        multi_conv_data_left_no_noise[0, :] * np.iinfo(np.int16).max / 20.0,
        "./ica_left_clean.wav",
        sample_rate,
    )

    # 畳み込んだ波形をファイルに書き込む
    write_file_from_time_signal(
        multi_conv_data_right_no_noise[0, :] * np.iinfo(np.int16).max / 20.0,
        "./ica_right_clean.wav",
        sample_rate,
    )

    # 畳み込んだ波形をファイルに書き込む
    write_file_from_time_signal(
        multi_conv_data[0, :] * np.iinfo(np.int16).max / 20.0,
        "./ica_in_left.wav",
        sample_rate,
    )
    write_file_from_time_signal(
        multi_conv_data[0, :] * np.iinfo(np.int16).max / 20.0,
        "./ica_in_right.wav",
        sample_rate,
    )

    # 短時間フーリエ変換を行う
    f, t, stft_data = sp.stft(multi_conv_data, fs=sample_rate, window="hann", nperseg=N)

    # ICAの繰り返し回数
    n_ica_iterations = 50

    # ICAの分離フィルタを初期化
    Wica = np.zeros(shape=(Nk, n_sources, n_sources), dtype=complex)

    Wica = Wica + np.eye(n_sources)[None, ...]

    Wiva = Wica.copy()
    Wiva_ip = Wica.copy()

    start_time = time.time()
    # 自然勾配法に基づくIVA実行コード（引数に与える関数を変更するだけ)
    Wiva, s_iva, cost_buff_iva = execute_natural_gradient_ica(
        stft_data,
        Wiva,
        phi_func=phi_multivariate_laplacian,
        contrast_func=contrast_multivariate_laplacian,
        mu=0.1,
        n_ica_iterations=n_ica_iterations,
        is_use_non_holonomic=False,
    )
    y_iva = projection_back(s_iva, Wiva)
    iva_time = time.time()

    # IP法に基づくIVA実行コード（引数に与える関数を変更するだけ)
    Wiva_ip, s_iva_ip, cost_buff_iva_ip = execute_ip_multivariate_laplacian_iva(
        stft_data, Wiva_ip, n_iterations=n_ica_iterations
    )
    y_iva_ip = projection_back(s_iva_ip, Wiva_ip)
    iva_ip_time = time.time()

    Wica, s_ica, cost_buff_ica = execute_natural_gradient_ica(
        stft_data,
        Wica,
        mu=0.1,
        n_ica_iterations=n_ica_iterations,
        is_use_non_holonomic=False,
    )
    permutation_index_result = solver_inter_frequency_permutation(s_ica)
    y_ica = projection_back(s_ica, Wica)

    # パーミュテーションを解く
    for k in range(Nk):
        y_ica[:, :, k, :] = y_ica[:, permutation_index_result[k], k, :]

    ica_time = time.time()

    t, y_ica = sp.istft(y_ica[0, ...], fs=sample_rate, window="hann", nperseg=N)
    t, y_iva = sp.istft(y_iva[0, ...], fs=sample_rate, window="hann", nperseg=N)
    t, y_iva_ip = sp.istft(y_iva_ip[0, ...], fs=sample_rate, window="hann", nperseg=N)

    snr_pre = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], multi_conv_data[0, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], multi_conv_data[0, ...])
    snr_pre /= 2.0

    snr_ica_post1 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_ica[0, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_ica[1, ...])
    snr_ica_post2 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_ica[1, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_ica[0, ...])

    snr_ica_post = np.maximum(snr_ica_post1, snr_ica_post2)
    snr_ica_post /= 2.0

    snr_iva_post1 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_iva[0, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_iva[1, ...])
    snr_iva_post2 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_iva[1, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_iva[0, ...])

    snr_iva_post = np.maximum(snr_iva_post1, snr_iva_post2)
    snr_iva_post /= 2.0

    snr_iva_ip_post1 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_iva_ip[0, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_iva_ip[1, ...])
    snr_iva_ip_post2 = calculate_snr(
        multi_conv_data_left_no_noise[0, ...], y_iva_ip[1, ...]
    ) + calculate_snr(multi_conv_data_right_no_noise[0, ...], y_iva_ip[0, ...])

    snr_iva_ip_post = np.maximum(snr_iva_ip_post1, snr_iva_ip_post2)
    snr_iva_ip_post /= 2.0

    write_file_from_time_signal(
        y_ica[0, ...] * np.iinfo(np.int16).max / 20.0, "./ica_1.wav", sample_rate
    )
    write_file_from_time_signal(
        y_ica[1, ...] * np.iinfo(np.int16).max / 20.0, "./ica_2.wav", sample_rate
    )

    write_file_from_time_signal(
        y_iva[0, ...] * np.iinfo(np.int16).max / 20.0, "./iva_1.wav", sample_rate
    )
    write_file_from_time_signal(
        y_iva[1, ...] * np.iinfo(np.int16).max / 20.0, "./iva_2.wav", sample_rate
    )

    write_file_from_time_signal(
        y_iva_ip[0, ...] * np.iinfo(np.int16).max / 20.0, "./iva_ip_1.wav", sample_rate
    )
    write_file_from_time_signal(
        y_iva_ip[1, ...] * np.iinfo(np.int16).max / 20.0, "./iva_ip_2.wav", sample_rate
    )

    print("method:    ", "NG-ICA", "NG-IVA", "AuxIVA")
    print(
        "処理時間[sec]: {:.2f}  {:.2f}  {:.2f}".format(
            ica_time - iva_ip_time, iva_ip_time - iva_time, iva_time - start_time
        )
    )
    print(
        "Δsnr [dB]: {:.2f}  {:.2f}  {:.2f}".format(
            snr_ica_post - snr_pre, snr_iva_post - snr_pre, snr_iva_ip_post - snr_pre
        )
    )

    # コストの値を表示
    # for t in range(n_ica_iterations):
    #    print(t,cost_buff_ica[t],cost_buff_iva[t],cost_buff_iva_ip[t])


if __name__ == "__main__":
    examine_effectiveness_sync()
