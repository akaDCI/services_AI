from PIL import Image


def visualize_image_with_mask(
    imgs: list[Image.Image],
    masks: list[Image.Image],
    color: tuple[int, int, int] = (255, 0, 0),
) -> list[Image.Image]:
    _overlays = []
    for img, mask in zip(imgs, masks):
        _red_mask = Image.new("RGB", mask.size, color)
        _red_mask.putalpha(mask)
        _red_mask = _red_mask.convert("RGBA")
        img = img.convert("RGBA")
        _blend = Image.alpha_composite(img, _red_mask)
        _overlays.append(Image.blend(
            img, _blend, alpha=0.5).convert("RGB"))

    return _overlays
