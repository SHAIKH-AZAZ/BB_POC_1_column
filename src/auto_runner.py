import os
import importlib

from config import INPUT_DIR, OUTPUT_DIR, PATTERN_OVERRIDES
from pattern_detector import detect_pattern


def _override_pattern(pdf_name):
    """Return a pinned pattern number for this PDF name, or None to auto-detect."""
    for name, pattern in PATTERN_OVERRIDES.items():
        if name.lower() == pdf_name.lower():
            return pattern
    return None


def run_pattern(pattern_number, pdf_path):

    try:
        module_name = f"main_{pattern_number}"
        module = importlib.import_module(module_name)

        print(f"🔎 Detected Pattern: {pattern_number}")
        print(f"🚀 Running {module_name}.py")

        module.process_pdf(pdf_path)

    except Exception as e:
        print(f"❌ Failed to run pattern {pattern_number}: {e}")


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:

        pdf_path = os.path.join(INPUT_DIR, pdf)
        pdf_name = os.path.splitext(pdf)[0]
        temp_folder = os.path.join(OUTPUT_DIR, pdf_name)

        os.makedirs(temp_folder, exist_ok=True)

        override = _override_pattern(pdf_name)

        try:
            if override is not None:
                print(f"\n📌 Using PATTERN OVERRIDE for {pdf}: Pattern {override} (skipping auto-detect)")
                pattern_number = override
            else:
                print(f"\n📄 Detecting pattern for {pdf}...")
                pattern_number = detect_pattern(pdf_path, temp_folder)
            run_pattern(pattern_number, pdf_path)
        except Exception as e:
            print(f"❌ Pattern detection failed: {e}")


if __name__ == "__main__":
    main()