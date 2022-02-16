import itertools

import numpy as np


# コントラスト関数の微分（球対称多次元ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def phi_multivariate_laplacian(s_hat):

    power = np.square(np.abs(s_hat))
    norm = np.sqrt(np.sum(power, axis=1, keepdims=True))

    phi = s_hat / np.maximum(norm, 1.0e-18)
    return phi


# コントラスト関数の微分（球対称ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def phi_laplacian(s_hat):

    norm = np.abs(s_hat)
    phi = s_hat / np.maximum(norm, 1.0e-18)
    return phi


# コントラスト関数（球対称ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def contrast_laplacian(s_hat):

    norm = 2.0 * np.abs(s_hat)

    return norm


# コントラスト関数（球対称多次元ラプラス分布を仮定）
# s_hat: 分離信号(M, Nk, Lt)
def contrast_multivariate_laplacian(s_hat):
    power = np.square(np.abs(s_hat))
    norm = 2.0 * np.sqrt(np.sum(power, axis=1, keepdims=True))

    return norm


# ICAによる分離フィルタ更新
# x:入力信号( M, Nk, Lt)
# W: 分離フィルタ(Nk,M,M)
# mu: 更新係数
# n_ica_iterations: 繰り返しステップ数
# phi_func: コントラスト関数の微分を与える関数
# contrast_func: コントラスト関数
# is_use_non_holonomic: True (非ホロノミック拘束を用いる） False (用いない）
# return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff ICAのコスト (T)
def execute_natural_gradient_ica(
    x,
    W,
    phi_func=phi_laplacian,
    contrast_func=contrast_laplacian,
    mu=1.0,
    n_ica_iterations=20,
    is_use_non_holonomic=True,
):

    # マイクロホン数を取得する
    M = np.shape(x)[0]

    cost_buff = []
    for t in range(n_ica_iterations):
        # 音源分離信号を得る
        s_hat = np.einsum("kmn,nkt->mkt", W, x)

        # コントラスト関数を計算
        G = contrast_func(s_hat)

        # コスト計算
        cost = np.sum(np.mean(G, axis=-1)) - np.sum(
            2.0 * np.log(np.abs(np.linalg.det(W)))
        )
        cost_buff.append(cost)

        # コンストラクト関数の微分を取得
        phi = phi_func(s_hat)

        phi_s = np.einsum("mkt,nkt->ktmn", phi, np.conjugate(s_hat))
        phi_s = np.mean(phi_s, axis=1)

        I = np.eye(M, M)
        if is_use_non_holonomic == False:
            deltaW = np.einsum("kmi,kin->kmn", I[None, ...] - phi_s, W)
        else:
            mask = (np.ones((M, M)) - I)[None, ...]
            deltaW = np.einsum("kmi,kin->kmn", np.multiply(mask, -phi_s), W)

        # フィルタを更新する
        W = W + mu * deltaW

    # 最後に出力信号を分離
    s_hat = np.einsum("kmn,nkt->mkt", W, x)

    return (W, s_hat, cost_buff)


# IP法による分離フィルタ更新
# x:入力信号( M, Nk, Lt)
# W: 分離フィルタ(Nk,M,M)
# n_iterations: 繰り返しステップ数
# return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff コスト (T)
def execute_ip_multivariate_laplacian_iva(x, W, n_iterations=20):

    # マイクロホン数を取得する
    M = np.shape(x)[0]

    cost_buff = []
    for t in range(n_iterations):

        # 音源分離信号を得る
        s_hat = np.einsum("kmn,nkt->mkt", W, x)

        # 補助変数を更新する
        v = np.sqrt(np.sum(np.square(np.abs(s_hat)), axis=1))

        # コントラスト関数を計算
        G = contrast_multivariate_laplacian(s_hat)

        # コスト計算
        cost = np.sum(np.mean(G, axis=-1)) - np.sum(
            2.0 * np.log(np.abs(np.linalg.det(W)))
        )
        cost_buff.append(cost)

        # IP法による更新
        Q = np.einsum(
            "st,mkt,nkt->tksmn", 1.0 / np.maximum(v, 1.0e-18), x, np.conjugate(x)
        )
        Q = np.average(Q, axis=0)

        for source_index in range(M):
            WQ = np.einsum("kmi,kin->kmn", W, Q[:, source_index, :, :])
            invWQ = np.linalg.pinv(WQ)
            W[:, source_index, :] = np.conjugate(invWQ[:, :, source_index])
            wVw = np.einsum(
                "km,kmn,kn->k",
                W[:, source_index, :],
                Q[:, source_index, :, :],
                np.conjugate(W[:, source_index, :]),
            )
            wVw = np.sqrt(np.abs(wVw))
            W[:, source_index, :] = W[:, source_index, :] / np.maximum(
                wVw[:, None], 1.0e-18
            )

    s_hat = np.einsum("kmn,nkt->mkt", W, x)

    return (W, s_hat, cost_buff)


# 周波数間の振幅相関に基づくパーミュテーション解法
# s_hat: M,Nk,Lt
# return permutation_index_result：周波数毎のパーミュテーション解
def solver_inter_frequency_permutation(s_hat):
    n_sources = np.shape(s_hat)[0]
    n_freqs = np.shape(s_hat)[1]
    n_frames = np.shape(s_hat)[2]

    s_hat_abs = np.abs(s_hat)

    norm_amp = np.sqrt(np.sum(np.square(s_hat_abs), axis=0, keepdims=True))
    s_hat_abs = s_hat_abs / np.maximum(norm_amp, 1.0e-18)

    spectral_similarity = np.einsum("mkt,nkt->k", s_hat_abs, s_hat_abs)

    frequency_order = np.argsort(spectral_similarity)

    # 音源間の相関が最も低い周波数からパーミュテーションを解く
    is_first = True
    permutations = list(itertools.permutations(range(n_sources)))
    permutation_index_result = {}

    for freq in frequency_order:

        if is_first == True:
            is_first = False

            # 初期値を設定する
            accumurate_s_abs = s_hat_abs[:, frequency_order[0], :]
            permutation_index_result[freq] = range(n_sources)
        else:
            max_correlation = 0
            max_correlation_perm = None
            for perm in permutations:
                s_hat_abs_temp = s_hat_abs[list(perm), freq, :]
                correlation = np.sum(accumurate_s_abs * s_hat_abs_temp)

                if max_correlation_perm is None:
                    max_correlation_perm = list(perm)
                    max_correlation = correlation
                elif max_correlation < correlation:
                    max_correlation = correlation
                    max_correlation_perm = list(perm)
            permutation_index_result[freq] = max_correlation_perm
            accumurate_s_abs += s_hat_abs[max_correlation_perm, freq, :]

    return permutation_index_result


# プロジェクションバックで最終的な出力信号を求める
# s_hat: M,Nk,Lt
# W: 分離フィルタ(Nk,M,M)
# retunr c_hat: マイクロホン位置での分離結果(M,M,Nk,Lt)
def projection_back(s_hat, W):

    # ステアリングベクトルを推定
    A = np.linalg.pinv(W)
    c_hat = np.einsum("kmi,ikt->mikt", A, s_hat)
    return c_hat
