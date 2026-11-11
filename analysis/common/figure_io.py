"""Shared figure output helper: manuscript-ready PNG (exact print width, 600 dpi) rendered from a larger canvas.

Word re-samples pictures that are much larger than their print size; writing the PNG at the final size
(183 mm = 7.2 in double-column width, 600 dpi) avoids that, and the LANCZOS down-sampling keeps lines and text sharp.
The vector PDF written next to it is unchanged.
"""
import io

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def save_manuscript_png(fig, path, width_in=7.2, out_dpi=600, render_dpi=900, **savefig_kw):
    kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0.015)
    kw.update(savefig_kw)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=render_dpi, **kw)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    w = int(round(width_in * out_dpi))
    h = int(round(im.height * w / im.width))
    im.resize((w, h), Image.LANCZOS).save(path, dpi=(out_dpi, out_dpi), optimize=True)
    return w, h
