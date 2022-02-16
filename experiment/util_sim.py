import numpy as np
import pyroomacoustics as pa


def simulate(clean_data, sample_rate):
    # シミュレーションで用いる音源数
    n_sim_sources = 2

    # 音声と雑音との比率 [dB]
    SNR = 90.0

    # 部屋の大きさ
    room_dim = np.r_[9.0, 7.0, 4.0]

    # マイクロホンアレイを置く部屋の場所
    mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1

    # マイクロホンアレイのマイク配置
    mic_directions = np.array(
        [[np.pi / 2.0, theta / 180.0 * np.pi] for theta in np.arange(180, 361, 180)]
    )

    distance = 0.01
    mic_alignments = np.zeros((3, mic_directions.shape[0]), dtype=mic_directions.dtype)
    mic_alignments[0, :] = np.cos(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
    mic_alignments[1, :] = np.sin(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
    mic_alignments[2, :] = np.cos(mic_directions[:, 0])
    mic_alignments *= distance

    # マイクロホン数
    n_channels = np.shape(mic_alignments)[1]

    # マイクロホンアレイの座標
    R = mic_alignments + mic_array_loc[:, None]

    is_use_reverb = True

    if is_use_reverb == False:
        # 部屋を生成する
        room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
        room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
        room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

    else:

        rt60 = 0.2
        e_absorption, max_order = pa.inverse_sabine(rt60, room_dim)
        room = pa.ShoeBox(
            room_dim,
            fs=sample_rate,
            max_order=max_order,
            materials=pa.Material(e_absorption),
        )
        room_no_noise_left = pa.ShoeBox(
            room_dim,
            fs=sample_rate,
            max_order=max_order,
            materials=pa.Material(e_absorption),
        )
        room_no_noise_right = pa.ShoeBox(
            room_dim,
            fs=sample_rate,
            max_order=max_order,
            materials=pa.Material(e_absorption),
        )

    # 用いるマイクロホンアレイの情報を設定する
    room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
    room_no_noise_left.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
    room_no_noise_right.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

    # 音源の場所
    doas = np.array([[np.pi / 2.0, np.pi], [np.pi / 2.0, 0]])

    # 音源とマイクロホンの距離
    distance = 1.0

    source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
    source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
    source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
    source_locations[2, :] = np.cos(doas[:, 0])
    source_locations *= distance
    source_locations += mic_array_loc[:, None]

    # 各音源をシミュレーションに追加する
    for s in range(n_sim_sources):
        clean_data[s] /= np.std(clean_data[s])
        room.add_source(source_locations[:, s], signal=clean_data[s])
        if s == 0:
            room_no_noise_left.add_source(source_locations[:, s], signal=clean_data[s])
        if s == 1:
            room_no_noise_right.add_source(source_locations[:, s], signal=clean_data[s])

    # シミュレーションを回す
    room.simulate(snr=SNR)
    room_no_noise_left.simulate(snr=90)
    room_no_noise_right.simulate(snr=90)

    # 畳み込んだ波形を取得する(チャンネル、サンプル）
    multi_conv_data = room.mic_array.signals
    multi_conv_data_left_no_noise = room_no_noise_left.mic_array.signals
    multi_conv_data_right_no_noise = room_no_noise_right.mic_array.signals
    # print(f"multi_conv_data.shape: {multi_conv_data.shape}")
    # print(f"multi_conv_data_left_no_noise.shape: {multi_conv_data_left_no_noise.shape}")
    # print(f"multi_conv_data_right_no_noise.shape: {multi_conv_data_right_no_noise.shape}")

    return (
        multi_conv_data,
        multi_conv_data_left_no_noise,
        multi_conv_data_right_no_noise,
    )
