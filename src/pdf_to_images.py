import fitz  # PyMuPDF
import os

def convert_pdf_to_images(pdf_path, output_folder, dpi=650):

    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        page = doc[page_number]

        # High resolution rendering
        pix = page.get_pixmap(dpi=dpi)

        image_path = os.path.join(
            output_folder,
            f"page_{page_number + 1}.png"
        )

        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()

    return image_paths