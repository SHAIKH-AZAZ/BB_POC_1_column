# import os
# import json
# import cv2
# from tqdm import tqdm

# from config import INPUT_DIR, OUTPUT_DIR
# from pdf_to_images import convert_pdf_to_images
# from vision_extractor import extract_from_image


# # -------------------------------
# # LOAD PROMPT
# # -------------------------------

# def load_prompt():
#     with open(
#         os.path.join(os.path.dirname(__file__), "prompt_1.txt"),
#         "r",
#         encoding="utf-8"
#     ) as f:
#         return f.read()


# # -------------------------------
# # VERTICAL SPLIT
# # -------------------------------

# def split_vertical(image, output_folder):

#     height, width = image.shape[:2]
#     slice_width = width // 4

#     paths = []

#     for i in range(4):
#         left = i * slice_width
#         right = (i + 1) * slice_width if i < 3 else width

#         crop = image[:, left:right]

#         path = os.path.join(output_folder, f"vertical_{i+1}.png")
#         cv2.imwrite(path, crop)

#         paths.append(path)

#     return paths


# # -------------------------------
# # SIZE PARSER
# # -------------------------------

# def parse_size(size_text):
#     try:
#         parts = size_text.lower().replace(" ", "").split("x")
#         width = int(parts[0])
#         length = int(parts[1])
#         return width, length
#     except:
#         return None, None


# # -------------------------------
# # PROCESS PDF
# # -------------------------------

# def process_pdf(pdf_path):

#     file_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     output_folder = os.path.join(OUTPUT_DIR, file_name)
#     os.makedirs(output_folder, exist_ok=True)

#     image_paths = convert_pdf_to_images(
#         pdf_path,
#         output_folder,
#         dpi=950
#     )

#     prompt = load_prompt()
#     final_columns = []

#     for img_path in image_paths:

#         full_img = cv2.imread(img_path)
#         vertical_slices = split_vertical(full_img, output_folder)

#         for slice_path in tqdm(vertical_slices):

#             result = extract_from_image(slice_path, prompt)

#             try:
#                 parsed = json.loads(result)
#                 column_blocks = parsed.get("columns", [])

#                 for block in column_blocks:

#                     column_no = block.get("column_no")

#                     stirrups = {
#                         "dia": ["T8"],        # if global, can improve later
#                         "spacing": ["200C/C"]
#                     }

#                     for lvl in block.get("levels", []):

#                         width, length = parse_size(lvl.get("size"))

#                         final_columns.append({
#                             "column_no": column_no,
#                             "column_name": lvl.get("column_name"),
#                             "size": {
#                                 "width": width,
#                                 "depth": None,
#                                 "length": length
#                             },
#                             "reinforcement": lvl.get("reinforcement", []),
#                             "stirrups": stirrups,
#                             "mix": None,
#                             "steel_grade": None
#                         })

#             except:
#                 continue

#     final_output = {"columns": final_columns}

#     output_file = os.path.join(output_folder, f"{file_name}.json")

#     with open(output_file, "w") as f:
#         json.dump(final_output, f, indent=2)

#     print(f"✅ Output saved to {output_file}")


# # -------------------------------
# # MAIN
# # -------------------------------

# def main():

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     pdf_files = [
#         f for f in os.listdir(INPUT_DIR)
#         if f.lower().endswith(".pdf")
#     ]

#     for pdf in pdf_files:
#         process_pdf(os.path.join(INPUT_DIR, pdf))


# if __name__ == "__main__":
#     main()


import os
import json
import cv2
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# -------------------------------
# LOAD PROMPT
# -------------------------------

def load_prompt():
    with open(
        os.path.join(os.path.dirname(__file__), "prompt_1.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


# -------------------------------
# VERTICAL SPLIT
# -------------------------------

def split_vertical(image, output_folder):

    height, width = image.shape[:2]
    slice_width = width // 4

    paths = []

    for i in range(4):
        left = i * slice_width
        right = (i + 1) * slice_width if i < 3 else width

        crop = image[:, left:right]

        path = os.path.join(output_folder, f"vertical_{i+1}.png")
        cv2.imwrite(path, crop)

        paths.append(path)

    return paths


# -------------------------------
# SIZE PARSER
# -------------------------------

def parse_size(size_text):
    try:
        parts = size_text.lower().replace(" ", "").split("x")
        width = int(parts[0])
        length = int(parts[1])
        return width, length
    except:
        return None, None


# -------------------------------
# PROCESS PDF
# -------------------------------

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = convert_pdf_to_images(
        pdf_path,
        output_folder,
        dpi=950
    )

    prompt = load_prompt()

    final_columns = []
    master_levels = []
    levels_detected = False

    for img_path in image_paths:

        full_img = cv2.imread(img_path)
        vertical_slices = split_vertical(full_img, output_folder)

        for slice_index, slice_path in enumerate(tqdm(vertical_slices)):

            result = extract_from_image(slice_path, prompt)

            try:
                parsed = json.loads(result)
                column_blocks = parsed.get("columns", [])

                for block in column_blocks:

                    column_no = block.get("column_no")

                    stirrups = {
                        "dia": ["T8"],
                        "spacing": ["200C/C"]
                    }

                    levels = block.get("levels", [])

                    # ---------------------------------------
                    # STEP 1: Extract master levels only once
                    # ---------------------------------------
                    if not levels_detected and slice_index == 0:
                        for lvl in levels:
                            name = lvl.get("column_name")
                            if name and name.strip():
                                master_levels.append(name.strip())

                        levels_detected = True

                        # Save detected levels to file
                        levels_file = os.path.join(output_folder, "detected_levels.json")
                        with open(levels_file, "w") as f:
                            json.dump({"levels": master_levels}, f, indent=2)

                        print("✅ Master Levels Stored")

                    # ---------------------------------------
                    # STEP 2: Attach correct level name
                    # ---------------------------------------
                    for i, lvl in enumerate(levels):

                        width, length = parse_size(lvl.get("size"))

                        # Use master levels if available
                        if master_levels:
                            level_name = master_levels[i % len(master_levels)]
                        else:
                            level_name = lvl.get("column_name")

                        final_columns.append({
                            "column_no": column_no,
                            "column_name": level_name,
                            "size": {
                                "width": width,
                                "depth": None,
                                "length": length
                            },
                            "reinforcement": lvl.get("reinforcement", []),
                            "stirrups": stirrups,
                            "mix": None,
                            "steel_grade": None
                        })

            except Exception as e:
                print("⚠ Error parsing slice:", e)
                continue

    final_output = {"columns": final_columns}

    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✅ Output saved to {output_file}")


# -------------------------------
# MAIN
# -------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
