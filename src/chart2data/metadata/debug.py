"""Visual verification overlays for stored pixel truth."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from chart2data.schema import CanonicalSample


def save_debug_overlay(sample: CanonicalSample, dataset_root: Path, output_path: Path) -> None:
    image = Image.open(dataset_root / sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    box = sample.plot_area
    draw.rectangle(
        [(box.left_px, box.top_px), (box.right_px, box.bottom_px)],
        outline=(255, 0, 255),
        width=2,
    )
    for x_px, y_px in zip(sample.pixel_curve.x_px, sample.pixel_curve.y_px, strict=True):
        radius = 2
        draw.ellipse(
            [(x_px - radius, y_px - radius), (x_px + radius, y_px + radius)],
            fill=(255, 140, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
