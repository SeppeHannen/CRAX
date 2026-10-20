"""The slides of the 2026-10-21 progress update, built from the kept figures and videos.

    /tmp/deck-venv/bin/python -m scripts.analysis.progress_update_deck [--preview]

Needs ``python-pptx`` and ``pillow`` (the project's environment has only the latter; a scratch
``venv`` with both will do). Writes ``docs/acl/progress_update_2026-10-21.pptx``; with
``--preview``, also one PNG per slide under ``/tmp/deck_preview/`` drawn by PIL with the same
geometry, so the layout can be looked at without PowerPoint. The videos are embedded as media
and play from the slide.

Every number on a slide comes from ``experiments/2026-10-19_goal_point_cost_tail_500M.md`` or
``experiments/2026-10-15_goal_point_learners_500M.md``.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/acl/experiments/figures"
TAIL = FIGURES / "2026-10-19_goal_point_cost_tail_500M"
OUTPUT = ROOT / "docs/acl/progress_update_2026-10-21.pptx"
PREVIEW = pathlib.Path("/tmp/deck_preview")

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
INK, MUTED, ACCENT, RED = RGBColor(0x1a, 0x1a, 0x1a), RGBColor(0x59, 0x59, 0x59), RGBColor(0x1f, 0x5f, 0x9e), RGBColor(0xd6, 0x27, 0x28)
FONT = "Calibri"
FONT_FILE, FONT_BOLD_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclasses.dataclass(frozen=True)
class Text:
    """A text box: ``lines`` as paragraphs; a line starting with ``•`` is a bullet."""

    left: int
    top: int
    width: int
    height: int
    lines: Sequence[str]
    size: int = 18
    bold: bool = False
    colour: RGBColor = INK
    align: PP_ALIGN = PP_ALIGN.LEFT


@dataclasses.dataclass(frozen=True)
class Picture:
    path: pathlib.Path
    left: int
    top: int
    width: Optional[int] = None  # one of width/height; the other follows the aspect ratio
    height: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Table:
    """Rows of cells; the first row is the header. ``widths`` are column widths in EMU."""

    left: int
    top: int
    widths: Sequence[int]
    rows: Sequence[Sequence[str]]
    size: int = 15
    row_height: int = Inches(0.6)


@dataclasses.dataclass(frozen=True)
class Video:
    path: pathlib.Path
    poster: pathlib.Path  # the frame shown before play
    left: int
    top: int
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class Slide:
    title: str
    elements: Sequence[object]
    takeaway: str = ""  # one sentence at the bottom, in the accent colour
    notes: str = ""


def title_box(title: str) -> Text:
    return Text(MARGIN, Inches(0.35), SLIDE_W - 2 * MARGIN, Inches(0.9), [title], size=28, bold=True)


def takeaway_box(text: str) -> Text:
    return Text(MARGIN, SLIDE_H - Inches(1.0), SLIDE_W - 2 * MARGIN, Inches(0.6), [text], size=18, bold=True, colour=ACCENT)


def fit(path: pathlib.Path, box_w: int, box_h: int) -> Tuple[int, int]:
    """The size that fits ``path``'s image inside the box, keeping its aspect ratio."""
    with Image.open(path) as image:
        w, h = image.size
    scale = min(box_w / w, box_h / h)
    return int(w * scale), int(h * scale)


def picture_in(path: pathlib.Path, left: int, top: int, box_w: int, box_h: int) -> Picture:
    w, h = fit(path, box_w, box_h)
    return Picture(path, left + (box_w - w) // 2, top + (box_h - h) // 2, width=w, height=h)


# --------------------------------------------------------------------------- the slides

def slides() -> List[Slide]:
    content_top, content_h = Inches(1.35), Inches(5.0)
    full_w = SLIDE_W - 2 * MARGIN
    col_w = (full_w - Inches(0.4)) // 2

    s: List[Slide] = []

    s.append(Slide(
        "Automated curriculum learning under a safety constraint — progress, 21 October",
        [Text(MARGIN, Inches(2.6), full_w, Inches(2.5), [
            "Giuseppe Hannen",
            "",
            "1  The per-context constraint works",
            "2  The best policy was hiding a problem",
            "3  What was tried",
            "4  The result, side by side",
        ], size=22)],
    ))

    s.append(Slide(
        "1 · One λ follows the curriculum; the constraint never holds where the policy is deployed",
        [Text(MARGIN, content_top, full_w, Inches(1.1), [
            "PPO-Lagrange holds its constraint with one scalar λ, updated from the cost of whatever the curriculum samples. Every arm so far: at the budget on its own distribution, three to four times over it on level 3.",
        ], size=17),
         Table(MARGIN, content_top + Inches(1.3), [Inches(2.4), Inches(4.7), Inches(4.7)], [
             ["", "published PPO-Lagrange", "contextual CMDP (this work)"],
             ["multiplier", "one scalar λ", "a function λ(ω), fitted each round to what the integrator asks for"],
             ["policy", "sees sensors only", "is told the context ω: hazard counts, goal radius"],
             ["critics", "see sensors only", "see sensors and ω"],
             ["forms of λ(ω)", "—", "network: a small MLP · additive: monotone per factor, so an unseen context's λ is bounded by its neighbours'"],
         ], size=15, row_height=Inches(0.62))],
        takeaway="Constraint per context, policy told the context: the curriculum can no longer move the constraint.",
    ))

    s.append(Slide(
        "1 · Every arm under λ(ω) ends at the budget on level 3; additive keeps more return",
        [picture_in(FIGURES / "2026-10-15_goal_point_learners_500M/end_state.png", MARGIN, content_top, full_w, content_h)],
        takeaway="Fifteen arms, 500 M steps. Scalar λ: none on budget. λ(ω): all on budget; additive > network on return everywhere. Additive is the default.",
    ))

    s.append(Slide(
        "2 · Replaying the best policy: the typical episode costs half the budget",
        [picture_in(TAIL / "mass_level3_old.png", MARGIN, content_top, Inches(7.2), content_h),
         Text(MARGIN + Inches(7.5), content_top + Inches(0.2), full_w - Inches(7.5), content_h, [
             "Every episode on a grid: goals across, cost up; colour = share of episodes.",
             "",
             "○ median: cost 13.",
             "× mean: 19.4.",
             "",
             "The top row is every episode costing 50 or more: 1 in 20 of the episodes, 29 % of all the cost.",
             "",
             "Orange line: the budget left for the other 95 % once those few are paid for.",
             "",
             "Those few set the multiplier. Every other episode pays for them with caution.",
         ], size=16)],
        takeaway="A constraint on the mean is met by a policy whose failures are few and large.",
    ))

    s.append(Slide(
        "2 · The failure: a gap too narrow, tried again and again",
        [Video(TAIL / "videos/level3_worst.mp4", TAIL / "videos/level3_worst_poster.png", MARGIN, content_top, Inches(6.3), Inches(5.0)),
         Text(MARGIN + Inches(6.7), content_top + Inches(0.2), full_w - Inches(6.7), content_h, [
             "Left: the published baseline (one λ, blind). Right: the best contextual policy — after the fix of part 3.",
             "",
             "The agent drives into a gap, is pushed back, approaches again, for the rest of the episode. The border flashes red at every contact.",
             "",
             "Why it cannot learn otherwise: the lidar reads the distance to each hazard's centre. A flat disc it may cross and a solid cylinder it cannot look identical, and the width of a gap is not in the observation at all.",
         ], size=16)],
        takeaway="Whether the gap is passable is not a function of what the policy sees.",
    ))

    s.append(Slide(
        "3 · Three remedies, after one measurement",
        [Text(MARGIN, content_top, full_w, Inches(0.9), [
            "Measured first, on 384 replayed episodes: contact is intermittent — 68 touches over 700 steps, never more than 10 in a row, a median of 32 steps between touches.",
        ], size=16, colour=MUTED),
         Table(MARGIN, content_top + Inches(1.0), [Inches(2.2), Inches(5.0), Inches(4.9)], [
             ["", "the idea", "what happened"],
             ["a camera", "shows kind, extent and occlusion, on every suite", "the same failure with the wall in frame; evaluations swing between checkpoints — set aside"],
             ["geometry", "the lidar as two arrays, passable and solid, reading the hazards' surface — the missing fact, in the sensor it already reads", "next slide"],
             ["memory", "a trace of recent sensors, actions and costs (τ = 32 steps, the median interval between touches) — so it can know it has been here", "next slide"],
             ["a cost rising with sustained contact", "charge pressing, not brushing", "not run: there is no sustained contact to charge"],
         ], size=14, row_height=Inches(0.7))],
        takeaway="Five arms at 500 M, judged by the per-episode metrics the evaluator now logs and by 2048 replayed episodes each.",
    ))

    s.append(Slide(
        "3 · Surface lidar: the median meets the mean and the tail is gone. Memory alone: neither",
        [picture_in(TAIL / "level3_2x2.png", MARGIN, content_top, full_w, content_h)],
        takeaway="The failure was perceptual. The surface lidar is the observation from here; the memory stays available and off.",
    ))

    s.append(Slide(
        "4 · Before and after, same learner and curriculum: the cloud moves up to the budget and right",
        [picture_in(TAIL / "mass_level3_before_after.png", MARGIN, content_top, full_w, content_h)],
        takeaway="The top row falls from 29 % of all cost to 8 %; the budget left for the ordinary episodes rises from 19.5 to 23.2 of 25; goals 13.1 → 15.5.",
    ))

    s.append(Slide(
        "4 · The published baseline against the best contextual policy, typical episodes",
        [Video(TAIL / "videos/level1_typical.mp4", TAIL / "videos/level1_typical_poster.png", MARGIN, content_top, col_w, Inches(4.6)),
         Video(TAIL / "videos/level3_typical.mp4", TAIL / "videos/level3_typical_poster.png", MARGIN + col_w + Inches(0.4), content_top, col_w, Inches(4.6)),
         Text(MARGIN, content_top + Inches(4.65), col_w, Inches(0.4), ["Level 1 — the baseline hoards the budget; the contextual policy spends it"], size=14, colour=MUTED, align=PP_ALIGN.CENTER),
         Text(MARGIN + col_w + Inches(0.4), content_top + Inches(4.65), col_w, Inches(0.4), ["Level 3 — the contextual policy under budget, the baseline over"], size=14, colour=MUTED, align=PP_ALIGN.CENTER)],
        takeaway="One constraint, one policy: at the budget on every level, more goals than the baseline on every level, a worst case eight times smaller.",
    ))

    s.append(Slide(
        "Two things to say honestly, and what comes next",
        [Text(MARGIN, content_top, col_w, Inches(0.5), ["Avoidance, not escape"], size=20, bold=True),
         Text(MARGIN, content_top + Inches(0.5), col_w, Inches(1.6), [
             "The agent enters a trap a third as often. Once inside, it gets out no more often than before.",
         ], size=18),
         Text(MARGIN, content_top + Inches(2.3), col_w, Inches(0.5), ["A mean constraint spends the budget"], size=20, bold=True),
         Text(MARGIN, content_top + Inches(2.8), col_w, Inches(2.2), [
             "Met by a policy that uses it, the budget is exceeded in four episodes of ten, by small amounts. That is what the constraint asks for — and the per-episode metrics now say so.",
         ], size=18),
         Text(MARGIN + col_w + Inches(0.4), content_top, col_w, Inches(0.5), ["Next"], size=20, bold=True),
         Text(MARGIN + col_w + Inches(0.4), content_top + Inches(0.5), col_w, content_h, [
             "•  A second seed before any claim about which curriculum is best (PLR led uniform under both lidars, one seed each).",
             "",
             "•  Task 2: the held-out hazard-count band — the first generalisation test, on an observation whose cost is not set by a tail.",
             "",
             "•  Recovery once inside a trap: a recurrent policy, or targeted experience.",
             "",
             "•  For Tristan: the surface lidar and the memory part, both opt-in.",
         ], size=18)],
    ))
    return s


# --------------------------------------------------------------------------- pptx

def build(deck: Sequence[Slide], output: pathlib.Path) -> None:
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = SLIDE_W, SLIDE_H
    blank = presentation.slide_layouts[6]
    for spec in deck:
        slide = presentation.slides.add_slide(blank)
        for element in (title_box(spec.title), *spec.elements, *( [takeaway_box(spec.takeaway)] if spec.takeaway else [] )):
            if isinstance(element, Text):
                box = slide.shapes.add_textbox(element.left, element.top, element.width, element.height)
                frame = box.text_frame
                frame.word_wrap = True
                for index, line in enumerate(element.lines):
                    paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
                    paragraph.text = line
                    paragraph.alignment = element.align
                    for run in paragraph.runs:
                        run.font.name, run.font.size, run.font.bold = FONT, Pt(element.size), element.bold
                        run.font.color.rgb = element.colour
            elif isinstance(element, Table):
                shape = slide.shapes.add_table(len(element.rows), len(element.widths), element.left, element.top,
                                               sum(element.widths), element.row_height * len(element.rows))
                table = shape.table
                for column, width in enumerate(element.widths):
                    table.columns[column].width = width
                for r, row in enumerate(element.rows):
                    table.rows[r].height = element.row_height
                    for c, text in enumerate(row):
                        cell = table.cell(r, c)
                        cell.text = text
                        for paragraph in cell.text_frame.paragraphs:
                            for run in paragraph.runs:
                                run.font.name, run.font.size = FONT, Pt(element.size)
                                run.font.bold = r == 0 or c == 0
                                run.font.color.rgb = INK
            elif isinstance(element, Picture):
                slide.shapes.add_picture(str(element.path), element.left, element.top, width=element.width, height=element.height)
            elif isinstance(element, Video):
                slide.shapes.add_movie(str(element.path), element.left, element.top, element.width, element.height,
                                       poster_frame_image=str(element.poster), mime_type="video/mp4")
        if spec.notes:
            slide.notes_slide.notes_text_frame.text = spec.notes
    presentation.save(str(output))
    print(f"wrote {output} ({len(deck)} slides)")


# --------------------------------------------------------------------------- preview

def preview(deck: Sequence[Slide], directory: pathlib.Path, scale: float = 0.1) -> None:
    """One PNG per slide at ``scale`` px per EMU·1000 (0.1 → 1333×750), same geometry as the pptx."""
    directory.mkdir(parents=True, exist_ok=True)
    px = lambda emu: int(emu / 914400 * 96 * scale * 10)  # inches → 96 dpi → scale
    for index, spec in enumerate(deck, 1):
        canvas = Image.new("RGB", (px(SLIDE_W), px(SLIDE_H)), "white")
        draw = ImageDraw.Draw(canvas)
        for element in (title_box(spec.title), *spec.elements, *( [takeaway_box(spec.takeaway)] if spec.takeaway else [] )):
            if isinstance(element, Text):
                font = ImageFont.truetype(FONT_BOLD_FILE if element.bold else FONT_FILE, int(element.size * 96 / 72 * scale * 10 * 0.92))
                y = px(element.top) + 4
                width = px(element.width)
                for line in element.lines:
                    for wrapped in _wrap(draw, line, font, width - 8):
                        x = px(element.left) + 4
                        if element.align == PP_ALIGN.CENTER:
                            x = px(element.left) + (width - draw.textlength(wrapped, font=font)) / 2
                        draw.text((x, y), wrapped, fill=tuple(element.colour), font=font)
                        y += int(font.size * 1.25)
                    if not line:
                        y += int(font.size * 0.3)
            elif isinstance(element, Table):
                font = ImageFont.truetype(FONT_FILE, int(element.size * 96 / 72 * scale * 10 * 0.92))
                bold_font = ImageFont.truetype(FONT_BOLD_FILE, font.size)
                y = px(element.top)
                for r, row in enumerate(element.rows):
                    x = px(element.left)
                    fill = "#dfe7f0" if r == 0 else ("#f3f6fa" if r % 2 else "white")
                    draw.rectangle([x, y, x + px(sum(element.widths)), y + px(element.row_height)], fill=fill, outline="#c8c8c8")
                    for c, (text, width) in enumerate(zip(row, element.widths)):
                        f = bold_font if r == 0 or c == 0 else font
                        ty = y + 6
                        for wrapped in _wrap(draw, text, f, px(width) - 12):
                            draw.text((x + 6, ty), wrapped, fill="#1a1a1a", font=f)
                            ty += int(f.size * 1.25)
                        x += px(width)
                    y += px(element.row_height)
            elif isinstance(element, Picture):
                with Image.open(element.path) as image:
                    canvas.paste(image.resize((px(element.width), px(element.height))), (px(element.left), px(element.top)))
            elif isinstance(element, Video):
                with Image.open(element.poster) as image:
                    w, h = fit(element.poster, px(element.width), px(element.height))
                    canvas.paste(image.resize((w, h)), (px(element.left) + (px(element.width) - w) // 2, px(element.top) + (px(element.height) - h) // 2))
                draw.rectangle([px(element.left), px(element.top), px(element.left + element.width), px(element.top + element.height)], outline="#888")
                draw.text((px(element.left) + 8, px(element.top) + 8), "▶ video", fill="#fff", font=ImageFont.truetype(FONT_BOLD_FILE, 16))
        canvas.save(directory / f"slide_{index:02d}.png")
    print(f"previews in {directory}")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> List[str]:
    words, lines, current = text.split(" "), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def posters() -> None:
    """A poster frame per video (step 300 of the shorter episode), next to the video."""
    from imageio import v3 as iio

    for video in sorted((TAIL / "videos").glob("*.mp4")):
        poster = video.with_name(video.stem + "_poster.png")
        if not poster.exists():
            Image.fromarray(iio.imread(video, index=150)).save(poster)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true")
    arguments = parser.parse_args()
    posters()
    deck = slides()
    build(deck, OUTPUT)
    if arguments.preview:
        preview(deck, PREVIEW)


if __name__ == "__main__":
    main()
