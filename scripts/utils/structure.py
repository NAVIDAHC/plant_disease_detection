import os
from PIL import Image, ImageDraw, ImageFont

def generate_structure_representation(root_dir, output_prefix="structure"):
    """
    Generates a .txt and .png representation of the folder structure and filenames.

    Args:
        root_dir (str): The path to the root directory to analyze.
        output_prefix (str): The prefix for the output filenames (e.g., "structure").
    """

    txt_output_file = f"{output_prefix}.txt"
    png_output_file = f"{output_prefix}.png"

    structure_lines = []

    def traverse_directory(directory, level=0):
        if "/dataset" in directory.replace(os.sep, "/"):
            return

        structure_lines.append("  " * level + os.path.basename(directory) + "/")
        try:
            items = sorted(os.listdir(directory))
            for item in items:
                item_path = os.path.join(directory, item)
                if "/dataset" in item_path.replace(os.sep, "/"):
                    continue
                if os.path.isdir(item_path):
                    traverse_directory(item_path, level + 1)
                else:
                    structure_lines.append("  " * (level + 1) + item)
        except OSError as e:
            structure_lines.append("  " * (level + 1) + f"Error accessing: {os.path.basename(directory)} ({e})")

    traverse_directory(root_dir)

    # Create .txt representation
    with open(txt_output_file, "w") as f:
        for line in structure_lines:
            f.write(line + "\n")
    print(f"Text representation saved to '{txt_output_file}'")

    # --- Create .png representation (requires Pillow) ---
    try:
        # Configuration for the image
        line_height = 20
        padding = 10
        font_size = 14
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # Try Arial font
        except IOError:
            font = ImageFont.load_default()

        max_width = 0
        for line in structure_lines:
            width = font.getlength(line)
            max_width = max(max_width, int(width))

        image_height = len(structure_lines) * line_height + 2 * padding
        image_width = max_width + 2 * padding

        img = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(img)

        y_position = padding
        for line in structure_lines:
            draw.text((padding, y_position), line, fill="black", font=font)
            y_position += line_height

        img.save(png_output_file)
        print(f"Image representation saved to '{png_output_file}'")

    except ImportError:
        print("Pillow library is not installed. Cannot create the .png image.")
        print("Please install it using: pip install Pillow")
    except Exception as e:
        print(f"An error occurred while creating the .png image: {e}")

if __name__ == "__main__":
    root_directory = input("Enter the root directory to analyze: ")
    if os.path.isdir(root_directory):
        generate_structure_representation(root_directory, output_prefix="folder_structure")
    else:
        print(f"Error: '{root_directory}' is not a valid directory.")