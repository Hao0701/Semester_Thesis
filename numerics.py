import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocess as mp
import numpy as np
from scipy.integrate import dblquad, nquad
from scipy.optimize import fsolve

mu = 1
t = 1
D = 0
g = 2

gap = None


def s_wave_gap(gap):
    options = {"limit": 800, "epsabs": 1e-13}
    return (
        g
        / 2
        * nquad(
            lambda kx, ky: 1 / np.sqrt(xi(kx, ky) ** 2 + gap**2),
            [[-np.pi, np.pi], [-np.pi, np.pi]],
            opts=[options, options],
        )[0]
        / (2 * np.pi) ** 2
        - 1
    )


def xi(kx, ky):
    return -2 * t * (np.cos(kx) + np.cos(ky)) - mu


def delta(kx, ky):
    global gap
    if gap is None:
        gap = fsolve(s_wave_gap, 5.8e-5)[0]
        print("Gap = ", gap)
    return gap


def E0(kx, ky):
    return np.sqrt(delta(kx, ky) ** 2 + xi(kx, ky) ** 2)


def u(kx, ky):
    return np.sqrt(0.5 * (1 + xi(kx, ky) / E0(kx, ky)))


def v(kx, ky):
    return np.sqrt(0.5 * (1 - xi(kx, ky) / E0(kx, ky)))


def Dx(k1x, k1y, k2x, k2y):
    return -D * np.sin(k2y - k1y)


def Dy(k1x, k1y, k2x, k2y):
    return +D * np.sin(k2x - k1x)


def complex_dblquad(func, a, b, gfun, hfun):
    def real_func(kx, ky):
        return np.real(func(kx, ky))

    def imag_func(kx, ky):
        return np.imag(func(kx, ky))

    options = {"limit": 1000, "epsabs": 1e-20}
    real_integral = nquad(real_func, [[a, b], [gfun, hfun]], opts=options)
    imag_integral = nquad(imag_func, [[a, b], [gfun, hfun]], opts=options)
    return real_integral[0] + 1j * imag_integral[0]


C1 = None
C2 = None


def computeAB(kx, ky):
    def util1(k2x, k2y):
        return D * np.cos(k2y) * u(k2x, k2y) * v(k2x, k2y)

    def util2(k2x, k2y):
        return D * np.cos(k2x) * u(k2x, k2y) * v(k2x, k2y)

    global C1, C2
    if C1 is None:
        C1 = complex_dblquad(util1, -np.pi, np.pi, -np.pi, np.pi) / (2 * np.pi) ** 2
        C2 = complex_dblquad(util2, -np.pi, np.pi, -np.pi, np.pi) / (2 * np.pi) ** 2
    A = -u(kx, ky) * v(kx, ky) * (C1 * np.sin(ky) + 1j * C2 * np.sin(kx))
    B = v(kx, ky) ** 2 * (C1 * np.sin(ky) + 1j * C2 * np.sin(kx))
    # def util1(k2x, k2y):
    #     return (
    #         -(Dx(kx, ky, k2x, k2y) - 1j * Dy(kx, ky, k2x, k2y))
    #         * u(kx, ky)
    #         * u(k2x, k2y)
    #         * v(kx, ky)
    #         * v(k2x, k2y)
    #     )

    # def util2(k2x, k2y):
    #     return (
    #         (Dx(kx, ky, k2x, k2y) - 1j * Dy(kx, ky, k2x, k2y))
    #         * u(k2x, k2y)
    #         * v(kx, ky) ** 2
    #         * v(k2x, k2y)
    #     )

    # A = complex_dblquad(util1, -np.pi, np.pi, -np.pi, np.pi) / (2 * np.pi) ** 2
    # B = complex_dblquad(util2, -np.pi, np.pi, -np.pi, np.pi) / (2 * np.pi) ** 2
    return A, B


def H_BdG(kx, ky, A, B):
    H = np.array(
        [
            [E0(kx, ky) / 2, A / 2, B / 2, 0],
            [np.conjugate(A / 2), E0(kx, ky) / 2, 0, -np.conjugate(B / 2)],
            [np.conjugate(B / 2), 0, -E0(kx, ky) / 2, -np.conjugate(A / 2)],
            [0, -B / 2, -A / 2, -E0(kx, ky) / 2],
        ]
    )
    evals, evecs = np.linalg.eigh(H)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def c_dd(kx, ky, A, B):
    _, S = H_BdG(kx, ky, A, B)

    return (
        -u(kx, ky) ** 2
        * (S[1, 0] * np.conjugate(S[3, 0]) + S[1, 1] * np.conjugate(S[3, 1]))
        + u(kx, ky)
        * v(kx, ky)
        * (S[2, 0] * np.conjugate(S[3, 0]) + S[2, 1] * np.conjugate(S[3, 1]))
        - u(kx, ky)
        * v(kx, ky)
        * (np.conjugate(S[0, 2]) * S[1, 2] + np.conjugate(S[0, 3]) * S[1, 3])
        + v(kx, ky) ** 2
        * (np.conjugate(S[0, 2]) * S[2, 2] + np.conjugate(S[0, 3]) * S[2, 3])
    )


def c_uu(kx, ky, A, B):
    _, S = H_BdG(kx, ky, A, B)
    return (
        -u(kx, ky) ** 2
        * (S[0, 0] * np.conjugate(S[2, 0]) + S[0, 1] * np.conjugate(S[2, 1]))
        - u(kx, ky)
        * v(kx, ky)
        * (S[3, 0] * np.conjugate(S[2, 0]) + S[3, 1] * np.conjugate(S[2, 1]))
        + u(kx, ky)
        * v(kx, ky)
        * (np.conjugate(S[1, 2]) * S[0, 2] + np.conjugate(S[1, 3]) * S[0, 3])
        + v(kx, ky) ** 2
        * (np.conjugate(S[1, 2]) * S[3, 2] + np.conjugate(S[1, 3]) * S[3, 3])
    )


def c_du(kx, ky, A, B):
    _, S = H_BdG(kx, ky, A, B)
    return (
        -u(kx, ky) ** 2
        * (S[0, 0] * np.conjugate(S[3, 0]) + S[0, 1] * np.conjugate(S[3, 1]))
        + u(kx, ky)
        * v(kx, ky)
        * (1 - S[3, 0] * np.conjugate(S[3, 0]) - S[3, 1] * np.conjugate(S[3, 1]))
        - u(kx, ky)
        * v(kx, ky)
        * (np.conjugate(S[0, 2]) * S[0, 2] + np.conjugate(S[0, 3]) * S[0, 3])
        - v(kx, ky) ** 2
        * (np.conjugate(S[3, 2]) * S[0, 2] + np.conjugate(S[3, 3]) * S[0, 3])
    )


def c_du_remove(kx, ky, A, B):
    _, S = H_BdG(kx, ky, A, B)
    return (
        -u(kx, ky) ** 2
        * (S[0, 0] * np.conjugate(S[3, 0]) + S[0, 1] * np.conjugate(S[3, 1]))
        + u(kx, ky)
        * v(kx, ky)
        * (0 - S[3, 0] * np.conjugate(S[3, 0]) - S[3, 1] * np.conjugate(S[3, 1]))
        - u(kx, ky)
        * v(kx, ky)
        * (np.conjugate(S[0, 2]) * S[0, 2] + np.conjugate(S[0, 3]) * S[0, 3])
        - v(kx, ky) ** 2
        * (np.conjugate(S[3, 2]) * S[0, 2] + np.conjugate(S[3, 3]) * S[0, 3])
    )

def c_du_unperturbed(kx, ky, A, B):
    return (
        u(kx, ky)
        * v(kx, ky)
    )

def fmt(x, pos):
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


def plot_order(order_funcs, args, titles, fname):
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.style.use("science")
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(11, 20), constrained_layout=True
    )
    axes[3, 0].set_xlabel(r"$k_x$", size=20)
    axes[3, 1].set_xlabel(r"$k_x$", size=20)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for i in range(len(order_funcs)):
        order_func = order_funcs[i]
        with mp.Pool(processes=num_processes) as pool:
            order = np.array(pool.starmap(order_func, args))
        pcm = axes[i, 0].pcolormesh(
            kxs,
            kys,
            order.reshape(N, N).real,
            vmin=np.min(np.concatenate([order.real, order.imag, [-1e-12]])),
            vmax=np.max(np.concatenate([order.real, order.imag, [1e-12]])),
            cmap="coolwarm",
            shading="gouraud",
        )
        # fig.colorbar(pcm, ax=axes[0])
        # axes[i, 0].set_xlabel(r"$k_x$", size=18)
        axes[i, 0].set_ylabel(r"$k_y$", size=18)
        axes[i, 0].text(
            0.5,
            0.9,
            r"\textbf{"+labels[2*i] + " Re(" + titles[i] + ")" +r"}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i, 0].transAxes,
            size=20,
        )

        pcm = axes[i, 1].pcolormesh(
            kxs,
            kys,
            order.reshape(N, N).imag,
            vmin=np.min(np.concatenate([order.real, order.imag, [-1e-12]])),
            vmax=np.max(np.concatenate([order.real, order.imag, [1e-12]])),
            cmap="coolwarm",
            shading="gouraud",
        )
        # axes[i, 1].set_xlabel(r"$k_x$", size=18)
        # axes[i, 1].set_ylabel(r"$k_y$", size=18)
        axes[i, 1].text(
            0.5,
            0.9,
            r"\textbf{"+labels[2*i+1] + " Im(" + titles[i] + ")" + r"}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i, 1].transAxes,
            size=20,
        )
        cbar = fig.colorbar(pcm, ax=axes[i, 1])
        cbar.formatter.set_powerlimits((0, 0))

    plt.savefig(fname)
    # plt.show()

def plot_order_2_4(order_funcs, args, titles, fname):
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)
    plt.style.use("science")

    gridspec = dict(wspace=0, width_ratios=[5, 5, 0.8, 5, 5])
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=(24, 12),gridspec_kw=gridspec,
        constrained_layout=True
    )
    # axes[3, 0].set_xlabel(r"$k_x$", size=18)
    # axes[3, 1].set_xlabel(r"$k_x$", size=18)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    col = 0
    row = 0
    for i in range(len(order_funcs)):
        order_func = order_funcs[i]
        with mp.Pool(processes=num_processes) as pool:
            order = np.array(pool.starmap(order_func, args))

        if col > 4:
            col = 0
            row += 1
        pcm = axes[row, col].pcolormesh(
            kxs,
            kys,
            order.reshape(N, N).real,
            vmin=np.min(np.concatenate([order.real, order.imag, [-1e-12]])),
            vmax=np.max(np.concatenate([order.real, order.imag, [1e-12]])),
            cmap="coolwarm",
            shading="gouraud",
        )
        if col == 0:
            cbar = fig.colorbar(pcm, ax=axes[row, col], location='left')
            cbar.formatter.set_powerlimits((0, 0))
        axes[row, col].set_xlabel(r"$k_x$", size=30)
        axes[row, 0].set_ylabel(r"$k_y$", size=30)
        axes[row, 3].set_ylabel(r"$k_y$", size=30)
        axes[row, col].text(
            0.5,
            0.9,
            r"\textbf{"+labels[2*i] + " Re(" + titles[i] + ")" +r"}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[row, col].transAxes,
            size=26,
        )
        axes[row, col].set_xticks([-2,0,2])

        pcm = axes[row, col+1].pcolormesh(
            kxs,
            kys,
            order.reshape(N, N).imag,
            vmin=np.min(np.concatenate([order.real, order.imag, [-1e-12]])),
            vmax=np.max(np.concatenate([order.real, order.imag, [1e-12]])),
            cmap="coolwarm",
            shading="gouraud",
        )
        axes[row, col+1].set_xlabel(r"$k_x$", size=30)
        # axes[i, 1].set_ylabel(r"$k_y$", size=18)
        axes[row, col+1].text(
            0.5,
            0.9,
            r"\textbf{"+labels[2*i+1] + " Im(" + titles[i] + ")" + r"}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[row, col+1].transAxes,
            size=26,
        )
        axes[row, col+1].tick_params(labelleft=False)
        axes[row, col+1].set_xticks([-2,0,2])
        if col == 3:
            cbar = fig.colorbar(pcm, ax=axes[row, col+1], location='right')
            cbar.formatter.set_powerlimits((0, 0))

        col += 3
        axes[row, 2].set_visible(False)

    plt.savefig(fname)
    # plt.show()


def read_args():
    global mu, g, D
    if len(sys.argv) == 4:
        mu = float(sys.argv[1])
        g = float(sys.argv[2])
        D = float(sys.argv[3])


if __name__ == "__main__":
    num_processes = 1
    N = 100
    read_args()
    kxs, kys = np.linspace(-np.pi, np.pi, N), np.linspace(-np.pi, np.pi, N)
    kxs, kys = np.meshgrid(kxs, kys)

    ks = np.vstack([kxs.ravel(), kys.ravel()]).T
    if os.path.isfile("data/A_%.1f_%.1f_%.1f.npy" % (mu, g, D)):
        As = np.load("data/A_%.1f_%.1f_%.1f.npy" % (mu, g, D))
        Bs = np.load("data/B_%.1f_%.1f_%.1f.npy" % (mu, g, D))
    else:
        with mp.Pool(processes=num_processes) as pool:
            res_mat = np.array(pool.starmap(computeAB, ks))
        As = res_mat[:, 0]
        Bs = res_mat[:, 1]
        np.save("data/A_%.1f_%.1f_%.1f.npy" % (mu, g, D), As)
        np.save("data/B_%.1f_%.1f_%.1f.npy" % (mu, g, D), Bs)

    args = np.hstack([ks, As[:, np.newaxis], Bs[:, np.newaxis]])
    plot_order_2_4(
        [c_dd, c_uu, c_du, c_du_remove],
        args,
        [
            r"$\Phi_{\downarrow \downarrow}$",
            r"$\Phi_{\uparrow \uparrow}$",
            r"$\Phi_{\downarrow \uparrow}$",
            r"$\Phi^\prime_{\downarrow \uparrow}$",
        ],
        "plot/order_%.1f_%.1f_%.1f.pdf" % (mu, g, D),
    )
    # order_func = c_du_unperturbed
    # with mp.Pool(processes=num_processes) as pool:
    #     order = np.array(pool.starmap(order_func, args))
    # print(np.max(order))
    # plot_order(
    #     c_dd,
    #     args,
    #     r"$\Phi_{\downarrow \downarrow}$",
    #     "plot/cdd_%.1f_%.1f_%.1f.pdf" % (mu, g, D),
    # )
    # plot_order(
    #     c_uu,
    #     args,
    #     r"$\Phi_{\uparrow \uparrow}$",
    #     "plot/cuu_%.1f_%.1f_%.1f.pdf" % (mu, g, D),
    # )
    # plot_order(
    #     c_du,
    #     args,
    #     r"$\Phi_{\downarrow \uparrow}$",
    #     "plot/cdu_%.1f_%.1f_%.1f.pdf" % (mu, g, D),
    # )

    # plot_order(
    #     c_du_remove,
    #     args,
    #     r"$\Phi_{\downarrow \uparrow}$",
    #     "plot/cdu_remove_%.1f_%.1f_%.1f.pdf" % (mu, g, D),
    # )

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # pcm = axes[0, 0].pcolormesh(
    #     kxs,
    #     kys,
    #     As.reshape(N, N).real,
    #     vmin=np.min([As.real, As.imag]),
    #     vmax=np.max([As.real, As.imag]),
    # )
    # axes[0, 0].set_title(r"Re($A$)")
    # fig.colorbar(pcm, ax=axes[0, 0])

    # pcm = axes[0, 1].pcolormesh(
    #     kxs,
    #     kys,
    #     As.reshape(N, N).imag,
    #     vmin=np.min([As.real, As.imag]),
    #     vmax=np.max([As.real, As.imag]),
    # )
    # fig.colorbar(pcm, ax=axes[0, 1])
    # axes[0, 1].set_title(r"Im($A$)")

    # pcm = axes[1, 0].pcolormesh(
    #     kxs,
    #     kys,
    #     Bs.reshape(N, N).real,
    #     vmin=np.min([Bs.real, Bs.imag]),
    #     vmax=np.max([Bs.real, Bs.imag]),
    # )
    # axes[1, 0].set_title(r"Re($B$)")
    # fig.colorbar(pcm, ax=axes[1, 0])
    # pcm = axes[1, 1].pcolormesh(
    #     kxs,
    #     kys,
    #     Bs.reshape(N, N).imag,
    #     vmin=np.min([Bs.real, Bs.imag]),
    #     vmax=np.max([Bs.real, Bs.imag]),
    # )
    # axes[1, 1].set_title(r"Im($B$)")
    # fig.colorbar(pcm, ax=axes[1, 1])
    # plt.savefig("%.1f_%.1f_%.1f.pdf" % (mu, g, D))
