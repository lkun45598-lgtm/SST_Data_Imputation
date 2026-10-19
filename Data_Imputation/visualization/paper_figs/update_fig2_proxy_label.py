"""Replace the ambiguous historical-proxy header in the submitted Fig. 2 PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FIGURE_PATH = (
    ROOT
    / "_overleaf_review"
    / "figure"
    / "result"
    / "fig2_causal_sst_proxy_construction.png"
)
PROXY_HEADER = "Historical proxy fields (t-6 to t-1)"
TITLE_BOX = (3150, 0, 5230, 145)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SIZE = 82


def update_proxy_header(path: Path = FIGURE_PATH) -> None:
    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.rectangle(TITLE_BOX, fill=(255, 255, 255, 255))

    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    bounds = draw.textbbox((0, 0), PROXY_HEADER, font=font)
    text_width = bounds[2] - bounds[0]
    x = TITLE_BOX[0] + (TITLE_BOX[2] - TITLE_BOX[0] - text_width) / 2
    draw.text((x, 13), PROXY_HEADER, font=font, fill=(0, 0, 0, 255))

    dpi = image.info.get("dpi", (600, 600))
    temporary_path = path.with_name(f".{path.stem}.tmp.png")
    image.save(temporary_path, format="PNG", dpi=dpi)
    temporary_path.replace(path)


if __name__ == "__main__":
    update_proxy_header()
    print(f"Updated: {FIGURE_PATH}")

