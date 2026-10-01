#!/usr/bin/env python3
"""
Mitsuba(mitsuba/bsdfs/nn.h)を使わずに、学習済みNBRDFの見た目を手早く確認するための
簡易プレビューレンダラー。

- nn.h の forward() と同じ計算(fc1->relu->fc2->relu->fc3->exp-1->relu)をnumpyで再現
- 球体 + 3灯の簡易ライティング(サンプル用の direct-lighting のみ、GIなし)
- sample_scene.xml と同じカメラ配置(origin=(0,0,-2), target=(0,0,1), fov=30)

Usage: python3 preview_render.py [weight_prefix] [out.png]
  weight_prefix: 既定は "_" (このディレクトリの _fc1.npy 等を読む)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coords


def load_weights(prefix):
    fc1 = np.load(prefix + 'fc1.npy')  # (6, 21)
    fc2 = np.load(prefix + 'fc2.npy')  # (21, 21)
    fc3 = np.load(prefix + 'fc3.npy')  # (21, 3)
    b1 = np.load(prefix + 'b1.npy')    # (21,)
    b2 = np.load(prefix + 'b2.npy')    # (21,)
    b3 = np.load(prefix + 'b3.npy')    # (3,)
    return fc1, fc2, fc3, b1, b2, b3


def nbrdf_forward(rvectors, weights, clamp=2000.0):
    """rvectors: (6, N) = [hx,hy,hz,dx,dy,dz]. nn.h と同じ計算。

    この小さなMLP(exp出力層)は、学習データがまばらな入力領域(グレージング角
    付近など)で桁違いに発散した値を出すことがある(実測MERLデータのBRDF最大値は
    alum-bronzeで~1668程度)。これはコードのバグではなくモデル自体の外挿不安定性
    なので、レンダリング時は実測データの値域を大きく超える異常値を安全にクランプする
    (Mitsuba側のnn.hは無条件でこれをクランプしないため、実際にレンダリングすると
    同様のfireflyが出る可能性がある)。
    """
    fc1, fc2, fc3, b1, b2, b3 = weights
    x = rvectors.T                                   # (N,6)
    a1 = np.maximum(x @ fc1 + b1, 0.0)                # (N,21)
    a2 = np.maximum(a1 @ fc2 + b2, 0.0)                # (N,21)
    pre = np.clip(a2 @ fc3 + b3, None, 20.0)           # exp(20)~5e8, オーバーフロー防止
    a3 = np.maximum(np.exp(pre) - 1.0, 0.0)            # (N,3)
    return np.clip(a3, 0.0, clamp)


def brdf_to_radiance_factor(rvectors, brdf):
    """train_NBRDF_pytorch.py の brdf_to_rgb と同じ (BRDF * cos_theta_i)。"""
    hx, hy, hz, dx, dy, dz = rvectors
    theta_h = np.arctan2(np.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = np.arctan2(dy, dx)
    cos_theta_i = (np.cos(theta_d) * np.cos(theta_h)
                   - np.sin(theta_d) * np.cos(phi_d) * np.sin(theta_h))
    cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)
    return brdf * cos_theta_i[:, None]


def to_local(v, tangent, bitangent, normal):
    return np.stack([
        np.einsum('ij,ij->i', v, tangent),
        np.einsum('ij,ij->i', v, bitangent),
        np.einsum('ij,ij->i', v, normal),
    ], axis=0)  # (3,N)


def render(weights, width=384, height=384, fov_deg=30.0, exposure=1.0):
    origin = np.array([0.0, 0.0, -2.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    forward = target - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    aspect = width / height
    half_h = np.tan(np.radians(fov_deg) / 2)
    half_w = half_h * aspect

    ys, xs = np.meshgrid(np.linspace(1, -1, height), np.linspace(-1, 1, width), indexing='ij')
    dirs = (forward[None, None, :]
            + xs[..., None] * half_w * right[None, None, :]
            + ys[..., None] * half_h * true_up[None, None, :])
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    center = np.array([0.0, 0.0, 0.0])
    radius = 0.5
    oc = origin - center
    b = 2 * np.einsum('ijk,k->ij', dirs, oc)
    c = np.dot(oc, oc) - radius ** 2
    disc = b ** 2 - 4 * c
    hit = disc >= 0
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[hit] = np.sqrt(disc[hit])
    t0 = (-b - sqrt_disc) / 2
    t_valid = hit & (t0 > 0)

    idx = np.where(t_valid)
    t_hit = t0[idx]
    hit_pts = origin[None, :] + dirs[idx] * t_hit[:, None]
    normals = (hit_pts - center) / radius
    view_dirs = -dirs[idx]
    n_hits = normals.shape[0]

    helper = np.tile(np.array([0.0, 1.0, 0.0]), (n_hits, 1))
    parallel_mask = np.abs(np.sum(normals * helper, axis=1)) > 0.999
    helper[parallel_mask] = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(helper, normals)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(normals, tangent)

    wo_local = to_local(view_dirs, tangent, bitangent, normals)

    # 簡易2灯照明 (key / fill)。各: (方向ベクトル, 強度RGB)
    lights = [
        (np.array([0.5, 0.6, -0.9]), np.array([1.0, 0.9, 0.75])),
        (np.array([-0.6, 0.2, -0.5]), np.array([0.25, 0.28, 0.35])),
    ]

    color_accum = np.zeros((n_hits, 3))
    for light_dir, intensity in lights:
        ld = light_dir / np.linalg.norm(light_dir)
        ld_tiled = np.tile(ld, (n_hits, 1))
        wi_local = to_local(ld_tiled, tangent, bitangent, normals)

        half, diff = coords.io_to_hd(wi_local, wo_local)
        # 学習データ(rangles_to_rvectors)はハーフベクトルの方位角を常に0に正準化
        # (等方性BRDFの標準的な表現)している。io_to_hdが返す生のhalfは方位角が
        # 0とは限らないため、ここで正準化しないと学習時に未見の入力になってしまう。
        _, theta_h, _ = coords.xyz2sph(*half)
        half_canonical = np.array([np.sin(theta_h), np.zeros_like(theta_h), np.cos(theta_h)])
        rvectors = np.concatenate([half_canonical, diff], axis=0)  # (6,N)

        brdf_val = nbrdf_forward(rvectors, weights)
        radiance = brdf_to_radiance_factor(rvectors, brdf_val)
        # 光源がローカル地平線より下(自己遮蔽)の場合は寄与なし
        below_horizon = wi_local[2] <= 0
        radiance[below_horizon] = 0.0
        color_accum += radiance * intensity[None, :]

    img = np.zeros((height, width, 3))
    img[idx] = color_accum

    print('radiance stats: min=%.4g median=%.4g p90=%.4g p99=%.4g max=%.4g' % (
        img[idx].min(), np.median(img[idx]), np.percentile(img[idx], 90),
        np.percentile(img[idx], 99), img[idx].max()))

    # background: 少しグラデーションのついたニュートラルグレー
    bg = np.array([0.05, 0.05, 0.06])
    background_mask = ~t_valid
    img[background_mask] = bg

    # tonemap (exposure + Reinhard) + gamma
    exposed = img * exposure
    tonemapped = exposed / (1.0 + exposed)
    out = np.clip(tonemapped, 0, 1) ** (1 / 2.2)
    return out


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, '_')
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, 'preview_render.png')

    weights = load_weights(prefix)
    exposure = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    img = render(weights, exposure=exposure)
    plt.imsave(out_path, img)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
