__author__ = "Xinyue"
# Modified by Takayama in 2023-01 for STLCCP

import numpy as np
import cvxpy as cvx


class _ParameterizedLinearization:
    """A reusable tangent with numerical gradients and offsets as parameters."""

    def __init__(self, expr):
        self.inputs = {}

        def replace_affine(node):
            if node.is_affine():
                if not node.variables():
                    return node
                if node.id not in self.inputs:
                    proxy = cvx.Variable(node.shape, complex=node.is_complex())
                    self.inputs[node.id] = (node, proxy)
                return self.inputs[node.id][1]
            return node.copy(args=[replace_affine(arg) for arg in node.args])

        # Differentiate with respect to local affine inputs, rather than every
        # entry of the full trajectory variable (most of which may be unused).
        self.expr = replace_affine(expr)
        self.offset = cvx.Parameter(expr.shape)
        self.tangent = self.offset
        self.gradients = []
        for arg, proxy in self.inputs.values():
            real_grad = cvx.Parameter((proxy.size, expr.size))
            imag_grad = cvx.Parameter((proxy.size, expr.size)) if proxy.is_complex() else None
            flat_arg = cvx.reshape(arg, (arg.size,), order="F")
            if imag_grad is None:
                term = real_grad.T @ flat_arg
            else:
                term = real_grad.T @ cvx.real(flat_arg) + imag_grad.T @ cvx.imag(flat_arg)
            self.tangent += cvx.reshape(term, expr.shape, order="F")
            self.gradients.append((proxy, real_grad, imag_grad))

    def update(self):
        for arg, proxy in self.inputs.values():
            if arg.value is None:
                raise ValueError(
                    "Cannot linearize non-affine expression with missing variable values."
                )
            proxy.value = arg.value
        grad_map = self.expr.grad
        if any(grad_map[proxy] is None for proxy, _, _ in self.gradients):
            return None
        offset = np.asarray(np.real(self.expr.value), dtype=float).reshape(-1, order="F").copy()
        for proxy, real_grad, imag_grad in self.gradients:
            grad = grad_map[proxy]
            if hasattr(grad, "toarray"):
                grad = grad.toarray()
            grad = np.asarray(grad).reshape(real_grad.shape, order="F")
            value = np.asarray(proxy.value).reshape(-1, order="F")
            real_grad.value = np.real(grad)
            offset -= np.real(grad).T @ np.real(value)
            if imag_grad is not None:
                imag_grad.value = np.imag(grad)
                offset -= np.imag(grad).T @ np.imag(value)
        # Compute the intercept numerically: gradient * expansion_point inside
        # CVXPY would multiply parameters and break DPP caching.
        self.offset.value = offset.reshape(self.offset.shape, order="F")
        return self.tangent


def linearize(expr, cache=None):
    """Returns the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

    Args:
        expr: An expression.
        cache: Optional dict retaining parameterized tangents across iterations.

    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    elif cache is not None:
        if expr.id not in cache:
            cache[expr.id] = _ParameterizedLinearization(expr)
        return cache[expr.id].update()
    else:
        if expr.value is None:
            raise ValueError(
                "Cannot linearize non-affine expression with missing variable values."
            )
        tangent = np.real(expr.value) #+ np.imag(expr.value)
        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            complex_flag = False
            if var.is_complex() or np.any(np.iscomplex(grad_map[var])):
                complex_flag = True
            if var.ndim > 1:
                temp = cvx.reshape(
                    cvx.vec(var - var.value), (var.shape[0] * var.shape[1], 1)
                )
                if complex_flag:
                    flattened = np.transpose(np.real(grad_map[var])) @ cvx.real(temp) + \
                    np.transpose(np.imag(grad_map[var])) @ cvx.imag(temp)
                else:
                    flattened = np.transpose(np.real(grad_map[var])) @ temp
                tangent = tangent + cvx.reshape(flattened, expr.shape)
            elif var.size > 1:
                if complex_flag:
                    tangent = tangent + np.transpose(np.real(grad_map[var])) @ (cvx.real(var) - np.real(var.value)) \
                    + np.transpose(np.imag(grad_map[var])) @ (cvx.imag(var) - np.imag(var.value))
                else:
                    tangent = tangent + np.transpose(np.real(grad_map[var])) @ (var - var.value)
            else:
                if complex_flag:
                    tangent = tangent + np.real(grad_map[var]) * (cvx.real(var) - np.real(var.value)) \
                    + np.imag(grad_map[var]) * (cvx.imag(var) - np.imag(var.value))
                else:
                    tangent = tangent + np.real(grad_map[var]) * (var - var.value)
        return tangent
