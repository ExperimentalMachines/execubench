"""
The ExecuBench mark, the Gauge, drawn once and written everywhere it appears:

    docs     docs/brand: marks, lockups, icon, social preview (and their PNGs)

    python3 tools/design/mark.py           # SVG outputs
    python3 tools/design/mark.py --png     # also PNGs, rendered by Chrome

A gauge reading high: the sign everyone reads as performance, measured. Its scale is three
faceted segments parted by the family's open cut, and the needle is the one ember part. It
shares ExecuServe's and ExecuPack's palette, cut, single ember part and wordmark
(execuserve/tools/design/mark.py, execupack/tools/design/mark.py). It is drawn front-on rather
than isometric, and built only from straight lines like they are: the facets are deliberate.

Needs shapely (the cut round the needle), and uharfbuzz and fonttools for the lockup
(pip install shapely uharfbuzz fonttools); none of them is a dependency of execubench itself.
The vendored Red Hat Display is in fonts/.
"""

import math
import pathlib
import shutil
import subprocess
import sys

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = pathlib.Path(__file__).resolve().parents[2]
FONT = ROOT / "tools/design/fonts/RedHatDisplay-Bold.ttf"
INK, PAPER, EMBER = "#262626", "#F3F4F7", "#EE4C2C"

# The gauge in a 100-unit drawing: centre, scale radii, sweep (degrees, counter-clockwise from
# east), segments and facets per segment arc.
CX, CY = 50.0, 62.0
R_OUT, R_IN = 46.0, 31.0
SWEEP = (200.0, -20.0)
SEGMENTS, FACETS = 3, 3
# The needle: length from the centre, angle, width at the hub, a short tail; the hub is an
# octagon.
NEEDLE, ANGLE, BASE, TAIL, HUB = 40.0, 55.0, 8.5, 2.2, 8.5

# Full drawing for 40 px and up. The small one keeps the shape and widens the gap round the
# needle, which is the cut that closes up first.
FULL = dict(cut=2.6, needle_gap=2.6)
SMALL = dict(cut=2.6, needle_gap=3.6)


# ---- geometry ----------------------------------------------------------------


def _point(r, deg):
    a = math.radians(deg)
    return CX + r * math.cos(a), CY - r * math.sin(a)


def scale_segments(cut):
    """The scale: SEGMENTS faceted ring segments, each giving up half the cut to each neighbour."""
    a0, a1 = SWEEP
    step = (a0 - a1) / SEGMENTS
    segs = []
    for k in range(SEGMENTS):
        s0, s1 = a0 - k * step, a0 - (k + 1) * step
        outer = [_point(R_OUT, s0 + (s1 - s0) * t / FACETS) for t in range(FACETS + 1)]
        inner = [_point(R_IN, s1 + (s0 - s1) * t / FACETS) for t in range(FACETS + 1)]
        segs.append(Polygon(outer + inner))
    parted = []
    for i, seg in enumerate(segs):
        for j, other in enumerate(segs):
            if j != i:
                seg = seg.difference(other.buffer(cut / 2, join_style="mitre"))
        parted.append(seg)
    return unary_union(parted)


def pointer():
    """The needle and its hub, one ember part."""
    a = math.radians(ANGLE)
    tip = (CX + NEEDLE * math.cos(a), CY - NEEDLE * math.sin(a))
    nx, ny = -math.sin(a), -math.cos(a)
    needle = Polygon([(CX + nx * BASE / 2, CY + ny * BASE / 2), tip, (CX - nx * BASE / 2, CY - ny * BASE / 2),
                      (CX - TAIL * math.cos(a), CY + TAIL * math.sin(a))])  # fmt: skip
    hub = Polygon([_point(HUB, deg) for deg in range(0, 360, 45)])
    return unary_union([needle, hub])


def gauge(spec):
    ember = pointer()
    ink = scale_segments(spec["cut"]).difference(ember.buffer(spec["needle_gap"], join_style="mitre", mitre_limit=4))
    return ink, ember


def polys(geom):
    out = []
    for g in geom.geoms if isinstance(geom, MultiPolygon) else [geom]:
        out.append(list(g.exterior.coords)[:-1])
        out += [list(hole.coords)[:-1] for hole in g.interiors]
    return out


def bounds(polys_):
    xs = [p[0] for poly in polys_ for p in poly]
    ys = [p[1] for poly in polys_ for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def placed(ink, ember, s, tx, ty):
    move = lambda poly: [(tx + s * x, ty + s * y) for x, y in poly]  # noqa: E731
    return [move(p) for p in polys(ink)], [move(p) for p in polys(ember)]


def fit(ink, ember, size, box):
    """Scale and centre so the bounding box fits `box` inside a `size` square."""
    x0, y0, x1, y1 = bounds(polys(ink) + polys(ember))
    s = box / max(x1 - x0, y1 - y0)
    return placed(ink, ember, s, size / 2 - s * (x0 + x1) / 2, size / 2 - s * (y0 + y1) / 2)


def fit_circle(ink, ember, size, radius):
    """Scale and centre so the farthest point sits on a circle: avatars are cropped by any mask."""
    pts = polys(ink) + polys(ember)
    x0, y0, x1, y1 = bounds(pts)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    far = max(math.hypot(x - cx, y - cy) for poly in pts for x, y in poly)
    s = radius / far
    return placed(ink, ember, s, size / 2 - s * cx, size / 2 - s * cy)


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def d(polys_):
    return "".join("M" + "L".join(f"{num(x)} {num(y)}" for x, y in poly) + "Z" for poly in polys_)


# ---- brand kit -----------------------------------------------------------------


def write(rel, text):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("wrote", rel)


def text_path(text, font_file, size, tracking=0.0):
    """HarfBuzz shaping (so kerning applies) and fontTools outlines; baseline at y = 0."""
    import uharfbuzz as hb
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    face = hb.Face(hb.Blob.from_file_path(str(font_file)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(face), buf, {"kern": True, "liga": True})
    tt = TTFont(str(font_file))
    glyphs, order = tt.getGlyphSet(), tt.getGlyphOrder()
    scale = size / face.upem
    pen = SVGPathPen(glyphs)
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        glyphs[order[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, x + pos.x_offset * scale, 0)))
        x += pos.x_advance * scale + tracking * size
    return pen.getCommands(), x - tracking * size


def paths(ink_polys, ember_polys, ink, ember):
    return (
        f'<path fill="{ink}" fill-rule="evenodd" d="{d(ink_polys)}"/>'
        f'<path fill="{ember}" fill-rule="evenodd" d="{d(ember_polys)}"/>'
    )


def mark_svg(ink, ember=EMBER, spec=FULL):
    body = paths(*fit(*gauge(spec), 100, 96.0), ink, ember)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{body}</svg>\n'


def lockup_svg(ink):
    """Mark and wordmark: Red Hat Display Bold at the family's size and tracking. The sibling
    lockups fit their mark into a square 2.15 cap heights on a side. The gauge is about 1.5 times
    as wide as it is tall, so it is set 1.6 cap heights tall (about 2.4 wide, close to the
    siblings' area) and a tenth of a cap height low, where its weight lines up with the word."""
    size = 64.0
    word, advance = text_path("ExecuBench", FONT, size, -0.012)
    cap = 0.7 * size
    height = 2.15 * cap
    g_ink, g_ember = gauge(FULL)
    x0, y0, x1, y1 = bounds(polys(g_ink) + polys(g_ember))
    h = 1.6 * cap
    s = h / (y1 - y0)
    top = height / 2 - h / 2 + 0.1 * cap
    ink_p, ember_p = placed(g_ink, g_ember, s, -s * x0, top - s * y0)
    tx = (x1 - x0) * s + 0.3 * size
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(tx + advance)} {num(height)}">'
        f"{paths(ink_p, ember_p, ink, EMBER)}"
        f'<path fill="{ink}" transform="translate({num(tx)} {num(height / 2 + cap / 2)})" d="{word}"/></svg>\n'
    )


def icon_svg():
    """Square icon for avatars (GitHub, Hugging Face): opaque ink, paper scale, ember needle."""
    body = paths(*fit_circle(*gauge(FULL), 108, 33.0), PAPER, EMBER)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="18 18 72 72"><rect x="18" y="18" width="72" height="72" '
        f'fill="{INK}"/>{body}</svg>\n'
    )


def social_preview_svg():
    """1280 x 640 GitHub social preview: the lockup in paper on ink, nothing else."""
    lock = lockup_svg(PAPER)
    w, h = (float(v) for v in lock.split('viewBox="0 0 ')[1].split('"')[0].split())
    s = 760 / w
    inner = lock[lock.index(">") + 1 : lock.rindex("</svg>")]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640"><rect width="1280" height="640" fill="{INK}"/>'
        f'<g transform="translate({num(640 - w * s / 2)} {num(320 - h * s / 2)}) scale({num(s)})">{inner}</g></svg>\n'
    )


def png(name, w, h, bg):
    chrome = shutil.which("google-chrome") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    page = ROOT / "build" / f"{name}.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text(
        f"<html><body style='margin:0'><img src='{ROOT}/docs/brand/{name}.svg' "
        f"style='display:block;width:{w}px;height:{h}px'></body></html>"
    )
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
         f"--default-background-color={bg}", f"--window-size={w},{h}",
         f"--screenshot={ROOT}/docs/brand/{name}.png", page.as_uri()],
        capture_output=True, check=True,
    )  # fmt: skip
    page.unlink()
    print("wrote", f"docs/brand/{name}.png")


def brand_kit(with_png):
    write("docs/brand/mark.svg", mark_svg(INK))
    write("docs/brand/mark-dark.svg", mark_svg(PAPER))
    write("docs/brand/mark-small.svg", mark_svg(INK, spec=SMALL))
    write("docs/brand/mark-small-dark.svg", mark_svg(PAPER, spec=SMALL))
    write("docs/brand/mark-mono.svg", mark_svg(INK, INK))
    write("docs/brand/lockup.svg", lockup_svg(INK))
    write("docs/brand/lockup-dark.svg", lockup_svg(PAPER))
    write("docs/brand/icon.svg", icon_svg())
    write("docs/brand/social-preview.svg", social_preview_svg())
    if with_png:
        png("mark", 1024, 1024, "00000000")
        png("icon", 512, 512, "ff262626")
        png("social-preview", 1280, 640, "ff262626")
        (ROOT / "build").rmdir()


if __name__ == "__main__":
    brand_kit("--png" in sys.argv)
