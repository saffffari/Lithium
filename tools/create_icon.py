"""Generate the 3P app icon as .ico and .png files."""

from PIL import Image, ImageDraw, ImageFont
import os


def create_icon():
    sizes = [256, 128, 64, 48, 32, 16]
    icons = []

    for size in sizes:
        img = Image.new('RGBA', (size, size), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        # Use a bold system font
        font_size = int(size * 0.58)
        try:
            font = ImageFont.truetype(
                os.path.expandvars(r"%WINDIR%\Fonts\arialbd.ttf"), font_size
            )
        except OSError:
            try:
                font = ImageFont.truetype(
                    os.path.expandvars(r"%WINDIR%\Fonts\segoeui.ttf"), font_size
                )
            except OSError:
                font = ImageFont.load_default()

        text = "3P"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1]

        draw.text((x, y), text, fill=(235, 235, 230, 255), font=font)
        icons.append(img)

    # Save .ico with all sizes
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    os.makedirs(out_dir, exist_ok=True)

    ico_path = os.path.join(out_dir, '3photon.ico')
    icons[0].save(ico_path, format='ICO', sizes=[(s, s) for s in sizes],
                  append_images=icons[1:])
    print(f"Created: {ico_path}")

    # Also save a 256px PNG
    png_path = os.path.join(out_dir, '3photon.png')
    icons[0].save(png_path)
    print(f"Created: {png_path}")


if __name__ == '__main__':
    create_icon()
