#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for diff-diff v2.7 release."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270     # mm
HEIGHT = 337.5  # mm

# Dark theme palette
BG = (26, 26, 46)          # #1a1a2e
CYAN = (0, 212, 255)       # #00d4ff
WHITE = (255, 255, 255)    # #ffffff
GRAY = (136, 146, 176)     # #8892b0
DARK_PANEL = (22, 33, 62)  # #16213e
ORANGE = (255, 107, 53)    # #ff6b35
GREEN = (80, 250, 123)     # #50fa7b
GOLD = (241, 250, 140)     # #f1fa8c

# Hex colors for matplotlib
CYAN_HEX = "#00d4ff"
WHITE_HEX = "#ffffff"
GRAY_HEX = "#8892b0"
DARK_PANEL_HEX = "#16213e"
ORANGE_HEX = "#ff6b35"


class CarouselV27PDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format=(WIDTH, HEIGHT))
        self.set_auto_page_break(False)
        self._temp_files = []

    def cleanup(self):
        """Remove temporary image files."""
        for f in self._temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass

    # ── Background & Footer ────────────────────────────────────────

    def _add_dark_bg(self):
        """Fill page with dark background."""
        self.set_fill_color(*BG)
        self.rect(0, 0, WIDTH, HEIGHT, "F")

    def _add_footer(self):
        """Add footer with cyan rule and version text."""
        rule_y = HEIGHT - 28
        self.set_draw_color(*CYAN)
        self.set_line_width(0.5)
        self.line(50, rule_y, WIDTH - 50, rule_y)

        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = "v2.7"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 22)
        self.set_text_color(*GRAY)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*CYAN)
        self.cell(v_w, 10, v_text)

    # ── Text Helpers ───────────────────────────────────────────────

    def _centered_text(self, y, text, size=28, bold=True, color=WHITE):
        """Add centered text."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(*color)
        self.cell(WIDTH, size * 0.5, text, align="C")

    # ── Equation Rendering ─────────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=28, color=CYAN_HEX):
        """Render LaTeX equations to transparent PNG."""
        n = len(latex_lines)
        fig_h = max(0.7, 0.55 * n + 0.15)
        fig = plt.figure(figsize=(10, fig_h))

        for i, line in enumerate(latex_lines):
            y_frac = 1.0 - (2 * i + 1) / (2 * n)
            fig.text(
                0.5, y_frac, line,
                fontsize=fontsize, ha="center", va="center",
                color=color,
            )

        fig.patch.set_alpha(0)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.06,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    def _place_equation_centered(self, path, pw, ph, y, max_w=200):
        """Place equation image centered on page at given y."""
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.75)
        display_h = display_w * aspect
        eq_x = (WIDTH - display_w) / 2
        self.image(path, eq_x, y, display_w)
        return display_h

    # ── Diagram: Single Comparison (Slide 2) ───────────────────────

    def _render_single_comparison(self):
        """ATT(g,t) -> single arrow -> one comparison box."""
        fig, ax = plt.subplots(figsize=(9, 2.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # ATT box (left)
        att = mpatches.FancyBboxPatch(
            (0.3, 0.5), 3.2, 1.5, boxstyle="round,pad=0.2",
            facecolor="none", edgecolor=CYAN_HEX, linewidth=2.5,
        )
        ax.add_patch(att)
        ax.text(1.9, 1.25, r"$ATT(g,\,t)$", color=CYAN_HEX,
                fontsize=24, ha="center", va="center", fontweight="bold")

        # Arrow
        ax.annotate(
            "", xy=(6.8, 1.25), xytext=(3.8, 1.25),
            arrowprops=dict(arrowstyle="->", color=GRAY_HEX, lw=2.5,
                            mutation_scale=20),
        )

        # Comparison box (right)
        comp = mpatches.FancyBboxPatch(
            (6.5, 0.5), 3.2, 1.5, boxstyle="round,pad=0.2",
            facecolor="none", edgecolor=GRAY_HEX, linewidth=2,
        )
        ax.add_patch(comp)
        ax.text(8.1, 1.25, r"$(g',\, t_{pre})$", color=GRAY_HEX,
                fontsize=22, ha="center", va="center")

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Diagram: Fan of Comparisons (Slide 3) ─────────────────────

    def _render_fan_comparison(self):
        """ATT(g,t) -> fan of arrows -> multiple comparison boxes."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.2, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # ATT box (left, vertically centered)
        att = mpatches.FancyBboxPatch(
            (0.2, 1.75), 3.2, 1.5, boxstyle="round,pad=0.2",
            facecolor="none", edgecolor=CYAN_HEX, linewidth=2.5,
        )
        ax.add_patch(att)
        ax.text(1.8, 2.5, r"$ATT(g,\,t)$", color=CYAN_HEX,
                fontsize=24, ha="center", va="center", fontweight="bold")

        # Fan of comparison boxes
        labels = [
            r"$(g_1,\, t_1)$",
            r"$(g_2,\, t_2)$",
            r"$(g_3,\, t_3)$",
            r"$(g_4,\, t_4)$",
            r"$(g_5,\, t_5)$",
        ]
        y_positions = np.linspace(4.5, 0.5, len(labels))

        for label, yp in zip(labels, y_positions):
            ax.annotate(
                "", xy=(6.5, yp), xytext=(3.6, 2.5),
                arrowprops=dict(arrowstyle="->", color=CYAN_HEX, lw=1.8,
                                alpha=0.7, mutation_scale=15),
            )
            box = mpatches.FancyBboxPatch(
                (6.3, yp - 0.4), 3.4, 0.8, boxstyle="round,pad=0.1",
                facecolor="none", edgecolor=GRAY_HEX, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(8.0, yp, label, color=GRAY_HEX,
                    fontsize=15, ha="center", va="center")

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── CI Comparison (Slide 4) ────────────────────────────────────

    def _render_ci_comparison(self):
        """Two horizontal CI bars: CS (wider, orange) vs EDiD (narrower, cyan)."""
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.set_xlim(-4, 5)
        ax.set_ylim(-0.5, 3)
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        point_est = 0.8

        # CS bar (wider, orange)
        cs_lo, cs_hi = -1.5, 3.1
        cs_y = 2.0
        ax.plot([cs_lo, cs_hi], [cs_y, cs_y], color=ORANGE_HEX, linewidth=7,
                solid_capstyle="round")
        ax.plot(point_est, cs_y, "o", color=ORANGE_HEX, markersize=14,
                zorder=5)
        ax.text(-3.8, cs_y, "Callaway-\nSant'Anna", color=ORANGE_HEX,
                fontsize=14, ha="left", va="center", fontweight="bold")

        # EDiD bar (narrower, cyan)
        edid_lo, edid_hi = 0.0, 1.6
        edid_y = 0.7
        ax.plot([edid_lo, edid_hi], [edid_y, edid_y], color=CYAN_HEX,
                linewidth=7, solid_capstyle="round")
        ax.plot(point_est, edid_y, "o", color=CYAN_HEX, markersize=14,
                zorder=5)
        ax.text(-3.8, edid_y, "EfficientDiD", color=CYAN_HEX,
                fontsize=14, ha="left", va="center", fontweight="bold")

        # Reference line at point estimate
        ax.axvline(point_est, color=GRAY_HEX, linewidth=1.2, linestyle="--",
                   alpha=0.6)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Code Block (Slide 5) ───────────────────────────────────────

    def _add_code_block(self, x, y, w, token_lines, font_size=13,
                        line_height=12):
        """Render syntax-highlighted code on a dark panel.

        token_lines: list of lists of (text, color_tuple) pairs per line.
        """
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        # Dark panel
        self.set_fill_color(*DARK_PANEL)
        self.rect(x, y, w, total_h, "F")

        self.set_font("Courier", "", font_size)
        char_w = self.get_string_width("M")

        pad_x = 15
        pad_y = 12

        for i, tokens in enumerate(token_lines):
            cx = x + pad_x
            cy = y + pad_y + i * line_height

            for text, color in tokens:
                if not text:
                    continue
                self.set_xy(cx, cy)
                self.set_text_color(*color)
                self.cell(char_w * len(text), 10, text)
                cx += char_w * len(text)

        return total_h

    # ════════════════════════════════════════════════════════════════
    # SLIDES
    # ════════════════════════════════════════════════════════════════

    def slide_01_hook(self):
        """Slide 1: Hook — First open-source implementation."""
        self.add_page()
        self._add_dark_bg()

        # Hero — method name dominates the page
        self._centered_text(55, "Efficient DiD", size=52, color=CYAN)

        # Positioning statement
        self._centered_text(118, "First Open-Source", size=30, color=WHITE)
        self._centered_text(150, "Implementation", size=30, color=WHITE)

        # Badge
        badge_w = 170
        badge_h = 34
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 195
        self.set_draw_color(*CYAN)
        self.set_line_width(1.5)
        self.rect(badge_x, badge_y, badge_w, badge_h, "D")

        self.set_xy(badge_x, badge_y + 8)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*CYAN)
        self.cell(badge_w, 16, "diff-diff v2.7", align="C")

        # Citation
        self.set_xy(0, 247)
        self.set_font("Helvetica", "I", 17)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10, "Chen, Sant'Anna & Xie (2025)", align="C")

        # Tagline
        self._centered_text(278, "Semiparametric efficiency bound",
                            size=17, bold=False, color=GRAY)
        self._centered_text(295, "for staggered DiD",
                            size=17, bold=False, color=GRAY)

        self._add_footer()

    def slide_02_problem(self):
        """Slide 2: The Problem — single comparison per target."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "The Problem", size=38, color=WHITE)

        # Body text
        self._centered_text(80, "Standard staggered DiD uses one", size=18,
                            bold=False, color=GRAY)
        self._centered_text(100, "comparison per target effect --", size=18,
                            bold=False, color=GRAY)
        self._centered_text(120, "leaving valid information unused.", size=18,
                            bold=False, color=GRAY)

        # Diagram
        diag_path, dpw, dph = self._render_single_comparison()
        diag_w = WIDTH * 0.82
        diag_aspect = dph / dpw
        diag_h = diag_w * diag_aspect
        diag_x = (WIDTH - diag_w) / 2
        diag_y = 155
        self.image(diag_path, diag_x, diag_y, diag_w)

        # Label
        self._centered_text(diag_y + diag_h + 10,
                            "Single 2x2 comparison (CS default)",
                            size=16, bold=False, color=GRAY)

        self._add_footer()

    def slide_03_insight(self):
        """Slide 3: The Insight — multiple valid comparisons, GLS weights."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(25, "The Insight", size=38, color=WHITE)

        # Body text
        self._centered_text(68, "Under PT-All, multiple valid comparisons",
                            size=17, bold=False, color=GRAY)
        self._centered_text(86, "exist. EDiD finds the optimal combination.",
                            size=17, bold=False, color=GRAY)

        # Fan diagram
        fan_path, fpw, fph = self._render_fan_comparison()
        fan_w = WIDTH * 0.78
        fan_aspect = fph / fpw
        fan_h = fan_w * fan_aspect
        fan_x = (WIDTH - fan_w) / 2
        fan_y = 105
        self.image(fan_path, fan_x, fan_y, fan_w)

        # Equation
        eq_path, epw, eph = self._render_equations(
            [r"$w^* = \frac{\mathbf{1}' (\Omega^*)^{-1}"
             r"}{\mathbf{1}' (\Omega^*)^{-1} \mathbf{1}}$"],
            fontsize=34,
        )
        eq_y = fan_y + fan_h + 8
        eq_h = self._place_equation_centered(eq_path, epw, eph, eq_y,
                                             max_w=180)

        # Label
        self._centered_text(eq_y + eq_h + 8, "GLS optimal weighting",
                            size=16, bold=True, color=CYAN)

        self._add_footer()

    def slide_04_assumption(self):
        """Slide 4: The Assumption — what PT-All means and when it holds."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(25, "The Assumption", size=38, color=WHITE)
        self._centered_text(58, "Parallel Trends for All Groups", size=22,
                            bold=True, color=CYAN)

        # Definition (single line)
        self._centered_text(92, "All cohorts share common outcome trends,",
                            size=17, bold=False, color=GRAY)
        self._centered_text(110, "treated or not.", size=17, bold=False,
                            color=GRAY)

        # Two comparison panels
        margin = 28
        gap = 12
        panel_w = (WIDTH - margin * 2 - gap) / 2
        panel_h = 65
        panel_y = 132

        panels = [
            {
                "x": margin,
                "label": "PT-Post",
                "label_color": ORANGE,
                "line1": "Trends hold between treated",
                "line2": "cohort and comparison group",
                "result": "= CS post-treatment ATT",
            },
            {
                "x": margin + panel_w + gap,
                "label": "PT-All",
                "label_color": CYAN,
                "line1": "Trends hold across all",
                "line2": "cohorts, treated or not",
                "result": "= EfficientDiD",
            },
        ]

        for panel in panels:
            px = panel["x"]

            # Panel background
            self.set_fill_color(*DARK_PANEL)
            self.set_draw_color(*panel["label_color"])
            self.set_line_width(1.2)
            self.rect(px, panel_y, panel_w, panel_h, "DF")

            # Label badge centered at top edge
            badge_w = 70
            badge_h = 18
            badge_x = px + (panel_w - badge_w) / 2
            badge_y = panel_y - badge_h / 2
            self.set_fill_color(*panel["label_color"])
            self.rect(badge_x, badge_y, badge_w, badge_h, "F")
            self.set_xy(badge_x, badge_y + 3)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*BG)
            self.cell(badge_w, 12, panel["label"], align="C")

            # Description lines
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*WHITE)
            self.set_xy(px + 10, panel_y + 14)
            self.cell(panel_w - 20, 10, panel["line1"])
            self.set_xy(px + 10, panel_y + 28)
            self.cell(panel_w - 20, 10, panel["line2"])

            # Result
            self.set_xy(px + 10, panel_y + 46)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*panel["label_color"])
            self.cell(panel_w - 20, 10, panel["result"])

        # "When does PT-All hold?"
        section_y = panel_y + panel_h + 16
        self._centered_text(section_y, "When does PT-All hold?", size=20,
                            bold=True, color=WHITE)

        examples = [
            "Staggered policy rollouts across regions",
            "Administrative or geographic phasing",
            "Treatment timing unrelated to anticipated effects",
        ]

        y_cursor = section_y + 26
        for example in examples:
            self.set_xy(50, y_cursor)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*CYAN)
            self.cell(10, 10, "-")
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*GRAY)
            self.cell(WIDTH - 100, 10, example)
            y_cursor += 18

        self._add_footer()

    def slide_05_payoff(self):
        """Slide 5: The Payoff — tightest possible CIs."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "The Payoff", size=38, color=WHITE)

        # Body text
        self._centered_text(80, "Achieves the semiparametric efficiency bound",
                            size=18, bold=False, color=GRAY)
        self._centered_text(98, "-- tightest possible confidence intervals.",
                            size=18, bold=False, color=GRAY)

        # CI comparison
        ci_path, cpw, cph = self._render_ci_comparison()
        ci_w = WIDTH * 0.82
        ci_aspect = cph / cpw
        ci_h = ci_w * ci_aspect
        ci_x = (WIDTH - ci_w) / 2
        ci_y = 145
        self.image(ci_path, ci_x, ci_y, ci_w)

        # Annotation
        ann_y = ci_y + ci_h + 15
        self.set_xy(0, ann_y)
        self.set_font("Helvetica", "I", 16)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10,
                  '"Often exceeding 40% gains in precision"',
                  align="C")

        self.set_xy(0, ann_y + 20)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10,
                  "-- Chen, Sant'Anna & Xie (2025)",
                  align="C")

        self._add_footer()

    def slide_06_code(self):
        """Slide 6: The Code — syntax-highlighted API example."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "The Code", size=38, color=WHITE)

        margin = 28
        code_y = 80

        token_lines = [
            [("from", CYAN), (" diff_diff ", WHITE),
             ("import", CYAN), (" EfficientDiD", GOLD)],
            [],  # blank line
            [("edid", WHITE), (" = ", WHITE), ("EfficientDiD", GOLD),
             ("(", WHITE), ("pt_assumption", WHITE), ("=", WHITE),
             ('"all"', GREEN), (")", WHITE)],
            [("results", WHITE), (" = edid.fit(data, ", WHITE),
             ("outcome", WHITE), ("=", WHITE), ('"y"', GREEN),
             (",", WHITE)],
            [("                   ", WHITE), ("unit", WHITE),
             ("=", WHITE), ('"id"', GREEN), (", ", WHITE),
             ("time", WHITE), ("=", WHITE), ('"t"', GREEN),
             (",", WHITE)],
            [("                   ", WHITE), ("first_treat", WHITE),
             ("=", WHITE), ('"g"', GREEN), (",", WHITE)],
            [("                   ", WHITE), ("aggregate", WHITE),
             ("=", WHITE), ('"all"', GREEN), (")", WHITE)],
            [("results", WHITE), (".print_summary()", WHITE)],
        ]

        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines,
        )

        # Subtitle
        self._centered_text(code_y + code_h + 18,
                            "sklearn-like API", size=18, bold=False,
                            color=GRAY)

        self._add_footer()

    def slide_07_safety_net(self):
        """Slide 7: Safety Net — PT-Post post-treatment ATT matches CS."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "Safety Net", size=38, color=WHITE)

        # Body text — scoped to post-treatment ATT
        self._centered_text(78, "Under PT-Post, post-treatment ATT(g,t)",
                            size=18, bold=False, color=GRAY)
        self._centered_text(100, "matches Callaway-Sant'Anna exactly",
                            size=22, bold=True, color=CYAN)

        # Prominent equivalence badge
        badge_w = 200
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 150
        self.set_draw_color(*CYAN)
        self.set_line_width(1.5)
        self.rect(badge_x, badge_y, badge_w, badge_h, "D")
        self.set_xy(badge_x, badge_y + 8)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*CYAN)
        self.cell(badge_w, 18, "ATT point estimates identical",
                  align="C")

        # Corollary citation
        self.set_xy(0, badge_y + badge_h + 25)
        self.set_font("Helvetica", "I", 14)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10,
                  "Corollary 3.2, Chen, Sant'Anna & Xie (2025)",
                  align="C")

        self._add_footer()

    def slide_08_cta(self):
        """Slide 8: CTA — Get Started."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(55, "Get Started", size=42, color=WHITE)

        # pip install badge
        badge_w = 210
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 115
        self.set_fill_color(*CYAN)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 9)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*BG)
        self.cell(badge_w, 16, "pip install diff-diff", align="C")

        # Links
        self._centered_text(178, "github.com/igerber/diff-diff",
                            size=18, color=CYAN)
        self._centered_text(205, "arXiv:2506.17729",
                            size=18, color=CYAN)

        # Wordmark
        self.set_font("Helvetica", "B", 36)
        dd_text = "diff-diff "
        v_text = "v2.7"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, 255)
        self.set_text_color(*WHITE)
        self.cell(dd_w, 20, dd_text)
        self.set_text_color(*CYAN)
        self.cell(v_w, 20, v_text)

        # Subtitle
        self._centered_text(288, "Difference-in-Differences for Python",
                            size=15, bold=False, color=GRAY)

        self._add_footer()


def main():
    pdf = CarouselV27PDF()

    pdf.slide_01_hook()
    pdf.slide_02_problem()
    pdf.slide_03_insight()
    pdf.slide_04_assumption()
    pdf.slide_05_payoff()
    pdf.slide_06_code()
    pdf.slide_07_safety_net()
    pdf.slide_08_cta()

    output_path = Path(__file__).parent / "diff-diff-v27-carousel.pdf"
    pdf.output(str(output_path))
    print(f"PDF saved to: {output_path}")

    pdf.cleanup()


if __name__ == "__main__":
    main()
