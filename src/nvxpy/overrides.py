from functools import partial

import numpy as _np
import autograd.numpy as np
from autograd.extend import defvjp, primitive


_dot = partial(np.einsum, "...ij,...jk->...ik")


def _diag(a):
    return np.eye(a.shape[-1]) * a


def T(x):
    return np.swapaxes(x, -1, -2)


@primitive
def svd(a, full_matrices=True, compute_uv=True):
    return _np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)


# From: https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
# Manually defined here to make gradient stable for diagonal matrices
def grad_svd(usv_, a, full_matrices=True, compute_uv=True):
    def vjp(g):
        usv = usv_

        if not compute_uv:
            s = usv

            # Need U and V so do the whole svd anyway...
            usv = svd(a, full_matrices=False)
            u = usv[0]
            v = np.conj(T(usv[2]))

            return _dot(np.conj(u) * g[..., np.newaxis, :], T(v))

        elif full_matrices:
            raise NotImplementedError(
                "Gradient of svd not implemented for full_matrices=True"
            )

        else:
            u = usv[0]
            s = usv[1]
            v = np.conj(T(usv[2]))

            m, n = a.shape[-2:]

            k = np.min((m, n))
            # broadcastable identity array with shape (1, 1, ..., 1, k, k)
            i = np.reshape(
                np.eye(k), np.concatenate((np.ones(a.ndim - 2, dtype=int), (k, k)))
            )

            f = 1 / (
                (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i) + 1e-8
            )  # <---- change from original

            gu = g[0]
            gs = g[1]
            gv = np.conj(T(g[2]))

            utgu = _dot(T(u), gu)
            vtgv = _dot(T(v), gv)
            t1 = (f * (utgu - np.conj(T(utgu)))) * s[..., np.newaxis, :]
            t1 = t1 + i * gs[..., :, np.newaxis]
            t1 = t1 + s[..., :, np.newaxis] * (f * (vtgv - np.conj(T(vtgv))))

            if np.iscomplexobj(u):
                t1 = t1 + 1j * np.imag(_diag(utgu)) / s[..., np.newaxis, :]

            t1 = _dot(_dot(np.conj(u), t1), T(v))

            if m < n:
                i_minus_vvt = np.reshape(
                    np.eye(n), np.concatenate((np.ones(a.ndim - 2, dtype=int), (n, n)))
                ) - _dot(v, np.conj(T(v)))
                t1 = t1 + np.conj(
                    _dot(_dot(u / s[..., np.newaxis, :], T(gv)), i_minus_vvt)
                )

                return t1

            elif m == n:
                return t1

            elif m > n:
                i_minus_uut = np.reshape(
                    np.eye(m), np.concatenate((np.ones(a.ndim - 2, dtype=int), (m, m)))
                ) - _dot(u, np.conj(T(u)))
                t1 = t1 + T(_dot(_dot(v / s[..., np.newaxis, :], T(gu)), i_minus_uut))

                return t1

    return vjp


defvjp(svd, grad_svd)
