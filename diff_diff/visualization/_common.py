"""Shared utilities for the visualization subpackage."""


def _require_matplotlib():
    """Lazy import matplotlib with clear error message.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. " "Install it with: pip install matplotlib"
        )


def _require_plotly():
    """Lazy import plotly with clear error message.

    Returns
    -------
    module
        The ``plotly.graph_objects`` module.
    """
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install diff-diff[plotly]"
        )


def _plotly_default_layout(fig, *, title=None, xlabel=None, ylabel=None, show_legend=True):
    """Apply standard plotly layout settings.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to configure.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    show_legend : bool, default=True
        Whether to show the legend.
    """
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=show_legend,
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=60, r=30, t=50, b=50),
    )


_CSS_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "darkgrey": (169, 169, 169),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "coral": (255, 127, 80),
    "salmon": (250, 128, 114),
}


def _color_to_rgba(color, alpha=1.0):
    """Convert any color to an ``rgba(r, g, b, a)`` string for plotly.

    Accepts hex colors (``#rrggbb``, ``#rgb``), CSS named colors, and
    falls back to ``matplotlib.colors`` when available.

    Parameters
    ----------
    color : str
        Color specification.
    alpha : float, default=1.0
        Opacity value between 0 and 1.

    Returns
    -------
    str
        An ``rgba(r, g, b, a)`` string.
    """
    if not isinstance(color, str):
        raise ValueError(f"Expected a color string, got {type(color).__name__}")

    # 1. Hex colors: #rrggbb or #rgb
    stripped = color.lstrip("#")
    if color.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in stripped):
        if len(stripped) == 6:
            r = int(stripped[0:2], 16)
            g = int(stripped[2:4], 16)
            b = int(stripped[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        if len(stripped) == 3:
            r = int(stripped[0] * 2, 16)
            g = int(stripped[1] * 2, 16)
            b = int(stripped[2] * 2, 16)
            return f"rgba({r}, {g}, {b}, {alpha})"

    # 2. Named CSS colors
    if color.lower() in _CSS_COLORS:
        r, g, b = _CSS_COLORS[color.lower()]
        return f"rgba({r}, {g}, {b}, {alpha})"

    # 3. Already an rgba/rgb string — override alpha
    if color.startswith("rgba(") or color.startswith("rgb("):
        return color if alpha == 1.0 else color  # pass through for plotly

    # 4. Fallback: try matplotlib.colors if available
    try:
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(color)
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"
    except (ImportError, ValueError):
        pass

    raise ValueError(
        f"Cannot parse color '{color}'. Use hex (#rrggbb), a CSS color name, "
        "or install matplotlib for full color support."
    )


# Default color constants
DEFAULT_BLUE = "#2563eb"
DEFAULT_RED = "#dc2626"
DEFAULT_GREEN = "#22c55e"
DEFAULT_GRAY = "#6b7280"
DEFAULT_DARK = "#1f2937"
DEFAULT_SHADE = "#f0f0f0"
