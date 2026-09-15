"""Make ``mujoco.Renderer`` teardown safe when several renderers share a process.

``mujoco.rendering.classic.renderer.Renderer.close`` frees its GL context
*first* and its ``MjrContext`` *second*.  The MjrContext free issues
``glDelete*`` calls in whichever GL context is current at that moment -- with
two renderers alive (recorder + VLA policy cameras, or a stale renderer being
garbage-collected mid-run) that is the *other* renderer's context, and its
framebuffer/textures get deleted by id.  Every later frame from that renderer
is black.  Reproduced 2026-09-15: render A, free B, render A -> all zeros.

``install()`` replaces ``close`` with an order-correct version: make the dying
renderer's own context current, free the MjrContext, free the GL context, then
restore whatever was current before.  Idempotent; no-op if the classic
renderer is unavailable.
"""
from __future__ import annotations

_INSTALLED = False


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from mujoco.rendering.classic import renderer as _renderer
        import glfw
    except Exception:  # noqa: BLE001 - other GL backends / no renderer
        return False
    Renderer = _renderer.Renderer
    if getattr(Renderer, "_omniq_safe_close", False):
        _INSTALLED = True
        return True

    def close(self, _glfw=glfw) -> None:   # default arg survives interpreter teardown
        if getattr(_glfw, "get_current_context", None) is None:
            return
        rep = getattr(_glfw, "ERROR_REPORTING", None)
        try:
            _glfw.ERROR_REPORTING = "ignore"   # GLFW complains when this runs after its atexit terminate()
            _close(self)
        finally:
            if rep is not None:
                _glfw.ERROR_REPORTING = rep

    def _close(self) -> None:
        try:
            prev = glfw.get_current_context()
        except Exception:  # noqa: BLE001
            prev = None
        ctx = getattr(self, "_gl_context", None)
        own = getattr(ctx, "_context", None) if ctx is not None else None
        if own:
            try:
                ctx.make_current()
            except Exception:  # noqa: BLE001
                pass
        mjr = getattr(self, "_mjr_context", None)
        if mjr is not None:
            try:
                mjr.free()
            except Exception:  # noqa: BLE001
                pass
        self._mjr_context = None
        if ctx is not None:
            try:
                ctx.free()
            except Exception:  # noqa: BLE001
                pass
        self._gl_context = None
        if prev and prev != own:
            try:
                glfw.make_context_current(prev)
            except Exception:  # noqa: BLE001
                pass

    Renderer.close = close
    Renderer._omniq_safe_close = True
    _INSTALLED = True
    return True
