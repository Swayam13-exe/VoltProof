"""Make an uploaded bill image safe to send (SPEC §6.12), in memory only.

- decoded size at most 2 MB; PNG or JPEG only, checked by the magic bytes (not the file name or the
  declared type alone)
- opened and verified with Pillow (a broken or oversized image is refused), then re-encoded from the
  pixels alone: EXIF, GPS and every other metadata block are dropped
- the longest side is scaled down to 1600 px
- at most 25 megapixels: the size is read from the header and checked before any pixel is decoded, so a
  small file that declares a huge image is refused without allocating it; Pillow's own decompression-
  bomb guard is an error here too, and every decode failure becomes a BillImageError (a 422 with a plain
  message), never a 500
The image is never written to disk, logged or hashed here.
"""

from __future__ import annotations

import base64
import binascii
import io
import warnings

from PIL import Image

MAX_BYTES = 2 * 1024 * 1024
MAX_SIDE_PX = 1600
MAX_PIXELS = 25_000_000  # 25 megapixels: checked from the header, before decoding
MAGIC = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}
OUTPUT_MIME = "image/jpeg"


class BillImageError(ValueError):
    """The upload is not an image we accept; the message is shown to the user."""


def decode_upload(image_b64: str) -> bytes:
    if len(image_b64) > (MAX_BYTES * 4) // 3 + 8:
        raise BillImageError("The image is larger than 2 MB. Take a smaller photo or crop it.")
    try:
        data = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise BillImageError("The image could not be decoded.") from None
    if len(data) > MAX_BYTES:
        raise BillImageError("The image is larger than 2 MB. Take a smaller photo or crop it.")
    return data


def sanitize_image(data: bytes, mime: str) -> bytes:
    """A clean JPEG (no metadata, at most 1600 px on its longest side) from a PNG or JPEG upload."""
    if len(data) > MAX_BYTES:
        raise BillImageError("The image is larger than 2 MB. Take a smaller photo or crop it.")
    magic = MAGIC.get(mime)
    if magic is None:
        raise BillImageError("Only PNG and JPEG images are accepted.")
    if not data.startswith(magic):
        raise BillImageError("The file is not a PNG or JPEG image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)  # Pillow's soft limit: refuse too
            with Image.open(io.BytesIO(data)) as probe:  # reads the header only
                if probe.width * probe.height > MAX_PIXELS:
                    raise BillImageError("The image has too many pixels (more than 25 megapixels). Take a smaller photo.")
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                pixels = Image.new("RGB", image.size, (255, 255, 255))  # a fresh image: only the pixels carry over
                pixels.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
    except BillImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise BillImageError("The image has too many pixels (more than 25 megapixels). Take a smaller photo.") from None
    except Exception:  # noqa: BLE001 - any decode failure (truncated, corrupt, unknown format) is the user's file, not a server error
        raise BillImageError("The image could not be opened. Try another photo.") from None
    pixels.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    out = io.BytesIO()
    pixels.save(out, format="JPEG", quality=88)  # no exif= argument: nothing but pixels is written
    return out.getvalue()


def encode_for_display(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
