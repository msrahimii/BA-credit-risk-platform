"""Shared utilities and theme for the Credit Risk Platform."""

import streamlit as st
import plotly.graph_objects as go

# ── Color Palette — Pro Slate + Neon Accents ──
COLORS = {
    "bg_dark": "#e3e3e9",
    "bg_card": "#1a1a26",
    "bg_card_hover": "#222232",
    "accent": "#7c4dff",
    "accent_light": "#b388ff",
    "accent_soft": "#6c63ff",
    "success": "#00e676",
    "danger": "#ff5252",
    "warning": "#ffab40",
    "text": "#e8e8f0",
    "text_muted": "#7a7a90",
    "border": "#2a2a3a",
    "chart_purple": "#b388ff",
    "chart_blue": "#00e5ff",
    "chart_green": "#00e676",
    "chart_red": "#ff6b6b",
    "chart_amber": "#ffab40",
}


def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_theme():
    """Inject Pro dark-mode CSS — slate grays with neon accents."""
    # Load Inter font via <link> (not @import) to avoid blocking other font-face rules
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <style>

    /* Apply Inter font — but NOT to icon elements */
    .stApp { font-family: 'Inter', sans-serif; }
    [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

    /* ───── Stage background (the gap between panels) ───── */
    .stApp {
        background: #141422;
    }

    /* ───── Sidebar — Neumorphic 3D floating panel ───── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e32 0%, #1a1a2e 50%, #171728 100%) !important;
        border-right: none !important;
        border-radius: 0 24px 24px 0;
        margin: 10px 0;
        box-shadow:
            8px 0 24px rgba(0,0,0,0.7),
            16px 0 48px rgba(0,0,0,0.35),
            inset -1px 0 0 rgba(255,255,255,0.06),
            3px 0 20px rgba(124,77,255,0.1);
        z-index: 999;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    /* Neon edge glow */
    section[data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 24px;
        right: -1px;
        width: 2px;
        height: calc(100% - 48px);
        background: linear-gradient(180deg,
            transparent 0%,
            rgba(124,77,255,0.5) 25%,
            rgba(0,229,255,0.35) 50%,
            rgba(124,77,255,0.5) 75%,
            transparent 100%);
        border-radius: 2px;
        box-shadow: 0 0 10px rgba(124,77,255,0.3),
                    0 0 20px rgba(0,229,255,0.1);
        pointer-events: none;
    }

    /* Sidebar nav links */
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
        border-radius: 12px !important;
        margin: 2px 8px !important;
        padding: 10px 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        color: #8888a0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
        background: rgba(124,77,255,0.08) !important;
        border-color: rgba(124,77,255,0.12) !important;
        color: #b388ff !important;
        box-shadow: 0 0 20px rgba(124,77,255,0.05) !important;
    }
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: rgba(124,77,255,0.12) !important;
        border-color: rgba(124,77,255,0.25) !important;
        border-left: 3px solid #7c4dff !important;
        color: #e8e8f0 !important;
        box-shadow:
            0 0 25px rgba(124,77,255,0.08),
            inset 0 0 12px rgba(124,77,255,0.03) !important;
    }

    /* ───── Main panel — 3D rounded container ───── */
    .main .block-container {
        background: linear-gradient(180deg, #181828 0%, #141422 100%);
        border-radius: 24px;
        margin: 10px 14px 10px 6px;
        padding: 2rem 2.5rem !important;
        box-shadow:
            0 4px 24px rgba(0,0,0,0.5),
            0 12px 48px rgba(0,0,0,0.2),
            inset 0 1px 0 rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ───── Glassmorphism cards (st.columns borders) ───── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg,
            rgba(26,26,38,0.7) 0%,
            rgba(22,22,32,0.5) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(124,77,255,0.08);
        border-radius: 20px;
        padding: 8px;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(124,77,255,0.18);
        box-shadow:
            0 12px 48px rgba(0,0,0,0.4),
            0 0 30px rgba(124,77,255,0.04),
            inset 0 1px 0 rgba(255,255,255,0.05);
        transform: translateY(-2px);
    }

    /* ───── Metric cards ───── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg,
            rgba(26,26,38,0.8) 0%,
            rgba(22,22,32,0.6) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(124,77,255,0.08);
        border-radius: 20px;
        padding: 24px 28px;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(124,77,255,0.2);
        box-shadow:
            0 12px 40px rgba(0,0,0,0.35),
            0 0 25px rgba(124,77,255,0.05);
        transform: translateY(-3px);
    }
    div[data-testid="stMetric"] label {
        color: #7a7a90 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e8e8f0 !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* ───── Headers ───── */
    h1 {
        background: linear-gradient(135deg, #e8e8f0 0%, #b388ff 50%, #7c4dff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.03em;
    }
    h2 {
        color: #b388ff !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #9e8bff !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }

    /* ───── Buttons — neon glow ───── */
    .stButton > button {
        background: linear-gradient(135deg, #7c4dff 0%, #651fff 100%);
        color: white !important;
        border: 1px solid rgba(124,77,255,0.3);
        border-radius: 14px;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(124,77,255,0.3);
    }
    .stButton > button:hover {
        box-shadow:
            0 4px 30px rgba(124,77,255,0.5),
            0 0 60px rgba(124,77,255,0.12);
        transform: translateY(-2px);
        border-color: rgba(124,77,255,0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(124,77,255,0.3);
    }

    /* ───── Inputs ───── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #1a1a26 !important;
        border: 1px solid #2a2a3a !important;
        color: #e8e8f0 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: rgba(124,77,255,0.25) !important;
        box-shadow: 0 0 15px rgba(124,77,255,0.06) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #7c4dff !important;
        box-shadow: 0 0 20px rgba(124,77,255,0.12) !important;
    }

    /* ───── Slider ───── */
    .stSlider > div > div > div[role="slider"] {
        background: #7c4dff !important;
        box-shadow: 0 0 10px rgba(124,77,255,0.4) !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7c4dff, #b388ff) !important;
    }

    /* ───── Tabs — neon active state ───── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(26,26,38,0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid #2a2a3a;
        color: #7a7a90;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(124,77,255,0.08);
        border-color: rgba(124,77,255,0.2);
        color: #b388ff;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,77,255,0.18) !important;
        color: white !important;
        border-color: #7c4dff !important;
        box-shadow: 0 0 20px rgba(124,77,255,0.2) !important;
    }

    /* ───── Dataframes ───── */
    .stDataFrame {
        border-radius: 16px !important;
        overflow: hidden;
    }
    .stDataFrame > div {
        border: 1px solid #2a2a3a !important;
        border-radius: 16px !important;
    }

    /* ───── Divider ───── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(124,77,255,0.2) 50%,
            transparent 100%) !important;
    }

    /* ───── Radio buttons ───── */
    .stRadio > div {
        gap: 0.5rem;
    }
    .stRadio > div > label {
        background: rgba(26,26,38,0.5);
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 6px 16px;
        transition: all 0.3s ease;
    }
    .stRadio > div > label:hover {
        border-color: rgba(124,77,255,0.25);
        background: rgba(124,77,255,0.06);
    }

    /* ───── Scrollbar ───── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0c0c14; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c4dff, #6c63ff);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #b388ff; }

    /* ───── Tooltip / Popover ───── */
    div[data-baseweb="popover"] > div {
        background: rgba(12,12,20,0.95) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(124,77,255,0.15) !important;
        border-radius: 12px !important;
    }

    /* ───── Spinner ───── */
    .stSpinner > div {
        border-top-color: #7c4dff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def plotly_layout(fig, height=400):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor="rgba(42,42,58,0.5)", zerolinecolor="rgba(42,42,58,0.5)"),
        yaxis=dict(gridcolor="rgba(42,42,58,0.5)", zerolinecolor="rgba(42,42,58,0.5)"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_muted"]),
        ),
    )
    return fig


def metric_card_html(label, value, delta=None, delta_color="success", icon=None):
    """Create a premium metric card."""
    delta_html = ""
    if delta is not None:
        color = COLORS[delta_color]
        arrow = "&#9650;" if delta_color == "success" else "&#9660;"
        delta_html = (
            f'<div style="display:inline-flex;align-items:center;gap:4px;'
            f'background:{hex_to_rgba(color, 0.12)};'
            f'padding:3px 10px;border-radius:20px;margin-top:8px;">'
            f'<span style="color:{color};font-size:0.7rem;">{arrow}</span>'
            f'<span style="color:{color};font-size:0.78rem;font-weight:600;">{delta}</span>'
            f'</div>'
        )

    glow_color = COLORS.get(delta_color, COLORS["accent"])
    glow_bg = hex_to_rgba(glow_color, 0.06)

    return (
        '<div style="'
        'background:linear-gradient(135deg,rgba(26,26,38,0.8) 0%,rgba(22,22,32,0.6) 100%);'
        'backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);'
        'border:1px solid rgba(124,77,255,0.08);border-radius:20px;padding:28px;'
        'box-shadow:0 8px 32px rgba(0,0,0,0.25),inset 0 1px 0 rgba(255,255,255,0.03);'
        'transition:all 0.4s cubic-bezier(0.4,0,0.2,1);position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;'
        f'background:radial-gradient(circle,{glow_bg} 0%,transparent 70%);border-radius:50%;"></div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.72rem;text-transform:uppercase;'
        f'letter-spacing:0.1em;font-weight:600;position:relative;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:2.2rem;font-weight:800;margin-top:10px;'
        f'letter-spacing:-0.03em;line-height:1;position:relative;">{value}</div>'
        f'{delta_html}'
        '</div>'
    )


def section_header(title, subtitle=None):
    """Create a styled section header with optional subtitle."""
    sub_html = ""
    if subtitle:
        sub_html = (f'<p style="color:{COLORS["text_muted"]};font-size:0.95rem;'
                    f'margin-top:4px;">{subtitle}</p>')
    return (
        '<div style="margin-bottom:1.8rem;">'
        '<h1 style="background:linear-gradient(135deg,#e8e8f0 0%,#b388ff 40%,#7c4dff 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'
        f'font-weight:800;font-size:2.4rem;letter-spacing:-0.03em;margin-bottom:0;">{title}</h1>'
        f'{sub_html}</div>'
    )


def gauge_chart(value, title="", max_val=100, color_ranges=None):
    """Create a premium semicircle gauge chart."""
    if color_ranges is None:
        color_ranges = [
            (0, 30, COLORS["success"]),
            (30, 60, COLORS["warning"]),
            (60, 100, COLORS["danger"]),
        ]

    steps = [dict(range=[r[0], r[1]], color=hex_to_rgba(r[2], 0.2)) for r in color_ranges]
    bar_color = COLORS["success"]
    for low, high, color in color_ranges:
        if low <= value <= high:
            bar_color = color
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 52, "color": COLORS["text"],
                                         "family": "Inter"}},
        title={"text": title, "font": {"size": 15, "color": COLORS["text_muted"],
                                        "family": "Inter"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": COLORS["text_muted"],
                     "tickfont": {"color": COLORS["text_muted"], "size": 11},
                     "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(26,26,38,0.5)",
            "bordercolor": "rgba(124,77,255,0.1)",
            "borderwidth": 1,
            "steps": steps,
            "shape": "angular",
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        height=320,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def status_badge(text, color_key="accent"):
    """Create a glowing status badge."""
    color = COLORS[color_key]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:{hex_to_rgba(color, 0.15)};border:1px solid {hex_to_rgba(color, 0.3)};'
        f'color:{color};padding:5px 14px;border-radius:20px;font-size:0.8rem;font-weight:700;'
        f'letter-spacing:0.03em;text-transform:uppercase;box-shadow:0 0 15px {hex_to_rgba(color, 0.1)};">'
        f'<span style="width:7px;height:7px;background:{color};border-radius:50%;'
        f'box-shadow:0 0 8px {color};"></span>'
        f'{text}</span>'
    )


def info_card(title, content, accent_color=None):
    """Create an info card with accent border."""
    accent = accent_color or COLORS["accent_light"]
    return (
        f'<div style="background:linear-gradient(135deg,rgba(26,26,38,0.6) 0%,rgba(22,22,32,0.4) 100%);'
        f'backdrop-filter:blur(16px);border-left:3px solid {accent};border-radius:0 16px 16px 0;'
        f'padding:20px 24px;margin-bottom:14px;box-shadow:0 4px 20px rgba(0,0,0,0.15);transition:all 0.3s ease;">'
        f'<div style="color:{accent};font-weight:700;font-size:0.78rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">{title}</div>'
        f'<div style="color:{COLORS["text"]};font-size:0.92rem;line-height:1.5;">{content}</div>'
        f'</div>'
    )
