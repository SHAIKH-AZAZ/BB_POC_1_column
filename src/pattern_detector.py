import json
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def detect_pattern(pdf_path, temp_folder):

    image_paths = convert_pdf_to_images(pdf_path, temp_folder)

    if not image_paths:
        raise Exception("No image generated for detection.")

    first_image = image_paths[0]

    classification_prompt = """
    You are an expert structural engineer identifying RCC COLUMN SCHEDULE patterns from scanned drawings.

    Analyze the VISUAL STRUCTURE and HEADER of the page carefully.
    Return ONLY a single integer (1-14). No explanation. No extra text.

    =====================================================================
    PATTERN DEFINITIONS — READ ALL BEFORE DECIDING
    =====================================================================

    --- PATTERN 1: TYPE-GROUPED COLUMN SCHEDULE WITH FLOOR-WISE SIZE / REINF ROWS ---

    Visual structure:
    - This is a COLUMN SCHEDULE table, usually very wide and dense.
    - The page is mainly a tabular schedule, NOT a drawing grid.
    - TOP header contains TYPE groups such as:
      • TYPE-1
      • TYPE-2
      • TYPE-3
      • TYPE-4
      etc.
    - Under each TYPE header, one or more COLUMN NUMBERS are listed, such as:
      • C1,C7,C8,C14
      • C2,C6,C79
      • C40
      • C70/C72
    - LEFT-MOST side contains FLOOR LEVEL or FLOOR RANGE labels, such as:
      • ABOVE TERRACE LEVEL
      • TERRACE FLOOR LEVEL TO 4TH FLOOR LEVEL
      • 4TH FLOOR LEVEL TO 3RD FLOOR LEVEL
      • 3RD FLOOR LEVEL TO 2ND FLOOR LEVEL
      • 2ND FLOOR LEVEL TO 1ST FLOOR LEVEL
      • 1ST FLOOR LEVEL TO GROUND FLOOR LEVEL
      • GROUND FLOOR LEVEL TO PLINTH LEVEL
      • PLINTH LEVEL TO FOOTING LEVEL

    Row structure:
    - Each floor level is represented by repeated sub-rows.
    - Common sub-row labels include:
      • SIZE
      • REINF.
      • REINF
      • STEEL
    - SIZE rows contain dimensions such as:
      • 700 x 700
      • 750 x 750
      • 850 x 850
      • 300 x 650
    - REINF rows contain main reinforcement such as:
      • 20-T20
      • 4-T20+16-T16
      • 12-T25
      • 4-T25+8-T20
    - Stirrups/ties may be:
      • in a separate common note
      • at the bottom of the schedule
      • absent from individual cells
      • written as links/rings/ties

    Positive identifiers:
    ✓ "SCHEDULE OF COLUMNS" title
    ✓ TYPE-1 / TYPE-2 / TYPE-3 etc. across the top
    ✓ Column numbers grouped below TYPE headers
    ✓ Floor levels/floor ranges down the left side
    ✓ SIZE and REINF. repeated for each floor band
    ✓ Data cells contain text only, not column cross-section drawings

    Reject Pattern 1 if:
    - Cells contain drawn column cross-sections with rebar dots/circles
    - Page is dominated by elevation drawings
    - Header uses footing/pedestal schedule format
    - Table is only COLUMN MARK | SIZE without reinforcement
    - Table has CROSS SECTION OF COLUMN as a drawing column

    → RETURN 1


    --- PATTERN 2: COLUMN MARKED + 3-FLOOR SPAN HEADER (TEXT TABLE, NO DRAWINGS) ---

    Visual structure:
    - Clean tabular schedule (no column cross-section drawings inside cells)
    - Header row contains ALL of these as column titles:
      • "COLUMN MARKED" (leftmost)
      • "FOOTING TO GROUND FLOOR" (with SIZE / STEEL sub-columns)
      • "GROUND FLOOR TO FIRST FLOOR" (with SIZE / STEEL sub-columns)
      • "FIRST FLOOR TO ROOF LEVEL" (with SIZE / STEEL sub-columns)
    - Rows list column marks (AC11, BC13, etc.) with sizes and steel values
    - No drawings/diagrams inside the table cells

    REJECT if:
    - Drawings/cross-sections appear inside cells → NOT 2
    - "COLUMN FROM..." text appears repeatedly vertically → NOT 2

    → RETURN 2


    --- PATTERN 3: CROSS-SECTION DRAWINGS + PEDESTAL/d1/d2 TABLE ---

    Visual structure:
    - Table at the BOTTOM with rows including: d1, d2, PEDESTAL, R.C.C. SIZE,
      FOUNDATION LEVEL, BASEMENT LEVEL, COLUMN MKD.
    - ABOVE the table: multiple column cross-section DRAWINGS arranged in a grid
      (rectangular boxes with rebar dots/circles inside showing stirrups and bars)
    - Each drawing has: bar count, spacing annotation, concrete mix (M-35 etc.)
    - Two rows of drawings: one for FOUNDATION LVL TO BASEMENT SLAB LVL
      and one for BASEMENT FLOOR LVL TO BASEMENT SLAB LVL

    Key identifiers:
    ✓ d1 and d2 rows in the summary table
    ✓ PEDESTAL row in the summary table
    ✓ Cross-section drawings of columns above the table

    → RETURN 3


    --- PATTERN 4: COLUMN ID + GRID REFERENCE + TOP OF COLUMN TABLE ---

    Visual structure:
    - Simple clean table (usually few rows)
    - Header contains ALL of:
      • COLUMN ID (leftmost)
      • SIZE OF COLUMN with sub-headers "a" and "b"
      • NOS. (number of columns)
      • GRID REFERENCE
      • TOP OF COLUMN (with EL(+) values)
    - No drawings; purely numeric/text data

    → RETURN 4


    --- PATTERN 5: COLUMN ID + CROSS SECTION DRAWING TABLE ---

    Visual structure:
    - Table header contains:
      • S.NO. or COLUMN ID
      • COLUMN SIZE with LENGTH (a) and BREADTH (b)
      • CROSS SECTION OF COLUMN (SECTION 1-1) — actual diagram in this column
      • LATERAL TIES
    - Each row has a drawn cross-section diagram showing bar arrangement
    - Lateral tie details shown as separate smaller rectangular boxes

    Key identifiers:
    ✓ "CROSS SECTION OF COLUMN" as a column header
    ✓ Drawn cross-section box inside each data row cell

    → RETURN 5


    --- PATTERN 6: DRAWING SHEET — COLUMN ELEVATIONS WITH REPEATED SECTIONS ---

    Visual structure:
    - Page is dominated by column ELEVATION DRAWINGS (tall vertical rectangles)
    - Drawings are arranged in a GRID by column ID (columns) and floor range (rows)
    - Left column labels show floor ranges (e.g. "COLUMN FROM GROUND FLOOR LVL TO FIRST FLOOR LVL")
    - Bottom row shows column names (TA-C1, TA-C2, etc.)
    - Each cell contains a detailed elevation drawing showing stirrup zones
    - COL. RING/LINK annotations below each drawing
    - Reinforcement bar counts shown at top of each drawing

    Key identifiers:
    ✓ "COLUMN FROM ... TO ..." text on the LEFT SIDE as row labels
    ✓ Detailed elevation drawings (not cross-sections) repeated in a grid
    ✓ Column names (TA-C1, TA-C2 etc.) at the BOTTOM

    → RETURN 6


    --- PATTERN 7: COLUMN SIZE SUMMARY TABLE (SIZE ONLY, NO REINFORCEMENT) ---

    Visual structure:
    - Simple two-column table(s): COLUMN MARK | SIZES
    - Lists column marks with their sizes (e.g. 300 X 1200)
    - NO reinforcement details, NO floor-by-floor data
    - May have multiple tables on the page (e.g. "COLUMN SIZES (BLOCK-C)", "COLUMN SIZES (NON-TOWER)")
    - Table title may reference block names

    Key identifiers:
    ✓ Header: "COLUMN MARK" and "SIZES" — nothing else
    ✓ No steel/reinforcement values anywhere

    → RETURN 7


    --- PATTERN 8: FOOTING + PEDESTAL + BASEMENT COLUMN SCHEDULE ---

    Visual structure:
    - Detailed schedule with MULTIPLE grouped header sections:
      • FOOTING section: EXCAVATION, R.C.C.FOOTING SIZE (BXL), DEPTH (d, D), FOOTING STEEL
      • PEDESTAL SIZE (UP TO TIE)
      • BASEMENT FLOOR COLUMN section: COL. SIZE, VER. STEEL, COL. RING, MIX
    - Rows identified by FOOTING MARK (CP1, CP2, etc.)
    - Each row spans full width with data in all sections

    Key identifiers:
    ✓ "FOOTING MARK" in leftmost header
    ✓ "R.C.C.FOOTING SIZE" header
    ✓ "PEDESTAL SIZE" header
    ✓ "BASEMENT FLOOR COLUMN" header spanning multiple sub-columns

    → RETURN 8


    --- PATTERN 9: LAP-BASED SCHEDULE WITH CROSS-SECTION DRAWINGS ---

    Visual structure:
    - Page contains a LARGE GRID of column cross-section drawings
    - Row labels on the LEFT identify floor levels using LAP terminology:
      • "FOUNDATION LVL." (bottom row)
      • "1st LAP TO FOUNDATION (1ST BASE. ROOF SLAB LVL.)"
      • "2nd LAP TO 1st LAP (PARKING ROOF SLAB)"
      • "3rd LAP TO 2nd LAP (1st FLOOR ROOF SLAB)"
      • "4th LAP TO 3rd LAP (2nd FLOOR ROOF SLAB)"
      • etc.
    - Column headers at top: GROUP 1, GROUP 2, GROUP 3... with column IDs (AC13, AC14 etc.)
    - Each cell contains a cross-section drawing with SIZE, RING, MIX data

    Key identifiers:
    ✓ "LAP TO" appears in LEFT-SIDE row labels
    ✓ "GROUP 1", "GROUP 2"... column headers
    ✓ Cross-section drawings inside cells

    → RETURN 9


    --- PATTERN 10: TYPE-CODED INLINE SCHEDULE TABLE ---

    Visual structure:
    - Table with header: COLUMN NUMBERS at the bottom, and floor-span labels on left
      (e.g. "P05 TO P06", "P06 TO ECO-DECK", "BASE TO LG", etc.)
    - Each cell contains: SIZE with TYPE code (e.g. "400 X 1500 (TYPE-10)")
      followed by steel details and link details
    - TYPE codes embedded inline within the size specification
    - The same TYPE label (like TYPE-10, TYPE-15, TYPE-30) appears in many cells
    - Dense table with many columns (PC157, PC154, GC31 etc. as column numbers)

    Key identifiers:
    ✓ "(TYPE-XX)" appears inside size cells
    ✓ Floor span headers like "BASE TO LG", "LG TO GF", "GF TO P01" etc.
    ✓ Column numbers like PC157, GC31 across the top

    → RETURN 10


    --- PATTERN 11: FULL GRID — CROSS-SECTION DRAWINGS PER COLUMN PER FLOOR ---

    Visual structure:
    - Very large dense grid covering the whole page
    - COLUMN NAMES across the top (C1, C2, C3A, C3, C4, C5... C10, C11)
    - FLOOR LEVELS down the left side (BASEMENT TO GROUND, GROUND TO FIRST, ..., TERRACE TO L.M.R.)
    - Each cell has a cross-section drawing showing bar arrangement
    - Below each drawing: SIZE and REINF. (steel bar count) in small text
    - Bottom row has footing plan references (REFER RAFT FOOTING PLAN...)
    - The grid is extremely dense with dozens of drawings

    Key identifiers:
    ✓ Many column names (C1 through C11 or more) as TOP headers
    ✓ Many floor level names as LEFT-SIDE row labels
    ✓ Every cell contains a cross-section drawing
    ✓ Footing plan references at the very bottom

    → RETURN 11


    --- PATTERN 12: FLOOR-BY-FLOOR VERTICAL SCHEDULE WITH DRAWINGS ---

    Visual structure:
    - Page organized vertically with floors stacked top to bottom
    - LEFT column shows floor labels (e.g. "16 TH FLOOR ROOFTOP (MIX: M250)",
      "11 TH FLOOR ROOFTOP (MIX: M260)", "11TH FLOOR COLUMN",
      "10Th Floor column", "GROUND FLOOR COLUMN", etc.)
    - Each floor row contains multiple column cross-section drawings
    - Each drawing is a small rectangular box showing bar dots
    - At the very bottom: summary row with SIZE and STEEL data
    - Concrete mix and floor number shown on left side of each row
    - Relatively few columns but MANY floors (10-16 floor rows visible)

    Key identifiers:
    ✓ Floor number + "FLOOR ROOFTOP" OR "FLOOR COLUMN" labels on the LEFT,
      often with concrete mix / steel grade
    ✓ Multiple floors stacked vertically (not horizontally)
    ✓ Small cross-section drawings in each cell
    ✓ Summary table at the very bottom with SIZE / STEEL counts

    → RETURN 12


    --- PATTERN 13: LARGE CROSS-SECTION GRID — FLOOR ROWS × COLUMN COLUMNS ---

    Visual structure:
    - Full-page dense grid
    - COLUMN MARKS across top AND bottom (SW1-SW5, SW4, SW44, SW2, SW3, etc.)
    - FLOOR LEVELS down the left side (BASEMENT TO GROUND, GROUND TO FIRST, ..., FOURTEENTH TO TERRACE)
    - Each cell contains a large cross-section drawing showing rebar pattern
    - Under each drawing: SIZE row and STEEL (reinforcement) row with numbers
    - The drawings show complex rebar patterns (many bars in rows/columns)
    - Similar to Pattern 11 but columns are SHEAR WALLS (SW) not regular columns

    Key identifiers:
    ✓ Column names starting with SW (shear wall) or complex patterns
    ✓ Very large cross-section drawings — bigger than Pattern 11 per cell
    ✓ Both SIZE and STEEL rows under each drawing

    → RETURN 13


    --- PATTERN 14: COLUMN MARKED + FLOOR-STACKED DRAWINGS (3 FLOORS VISIBLE) ---

    Visual structure:
    - Page shows 3 floor ROWS stacked: GROUND FLOOR, FIRST FLOOR, ROOF FLOOR
    - Each row contains column cross-section DRAWINGS (rectangular boxes with bar annotations)
    - At the bottom: COLUMN MARKED row listing column groups (AC1,2,48,49 / BC3,5,... etc.)
    - Each drawing shows: bar count (2NOS-T20, 1NOS-T16), stirrup spacing (T8@100C/C, T8@200C/C), SIZE label
    - Concrete mix (M25) shown in a side label
    - The cross-section drawings are LARGER and more detailed than Pattern 11/12/13

    Key identifiers:
    ✓ "COLUMN MARKED" at the BOTTOM as the row label for column IDs
    ✓ GROUND FLOOR / FIRST FLOOR / ROOF FLOOR as LEFT-SIDE row labels
    ✓ Detailed cross-section drawings with bar labels in each cell
    ✓ Only 3 floor rows visible (not many floors like Pattern 12)

    → RETURN 14


    =====================================================================
    DECISION RULES — APPLY IN THIS EXACT ORDER
    =====================================================================

    STEP 1: Does the LEFT-SIDE contain "LAP TO" in floor labels?
    → YES → RETURN 9

    STEP 2: Does the page have a simple 2-column table: COLUMN MARK | SIZES (no reinforcement)?
    → YES → RETURN 7

    STEP 3: Does the header contain FOOTING MARK + R.C.C.FOOTING SIZE + PEDESTAL SIZE + BASEMENT FLOOR COLUMN?
    → YES → RETURN 8

    STEP 4: Does the header contain COLUMN ID + SIZE OF COLUMN (a, b) + NOS. + GRID REFERENCE + TOP OF COLUMN?
    → YES → RETURN 4

    STEP 5: Does the header contain CROSS SECTION OF COLUMN (as a column header with drawn diagrams inside rows)?
    → YES → RETURN 5

    STEP 6: Does the LEFT SIDE contain "COLUMN FROM ... TO ..." labels with elevation drawings (not cross-sections)?
    → YES → RETURN 6

    STEP 7: Are there cross-section drawings AND the summary table at the bottom contains d1, d2, PEDESTAL rows?
    → YES → RETURN 3

    STEP 8: Does the header contain COLUMN MARKED + FOOTING TO GROUND FLOOR + GROUND FLOOR TO FIRST FLOOR + FIRST FLOOR TO ROOF LEVEL (text only, no drawings in cells)?
    → YES → RETURN 2

    STEP 9: Does SIZE data appear with inline TYPE codes like "(TYPE-10)", "(TYPE-15)" etc.?
    → YES → RETURN 10

    STEP 10: Are there 3 floor rows (GROUND/FIRST/ROOF) with detailed cross-section drawings AND "COLUMN MARKED" at the bottom?
    → YES → RETURN 14

    STEP 11: Does the LEFT COLUMN contain floor-level names as ROW LABELS (like "ABOVE TERRACE LEVEL", "4TH FLOOR LEVEL") AND the top has TYPE-grouped or column-grouped headers?
    → YES — but now check the DATA CELLS:
       • If the data cells contain ONLY TEXT/NUMBERS (sizes like "700x700", steel like "20-T25") with NO drawn boxes/diagrams → RETURN 1
       • If the data cells contain cross-section DRAWINGS (rectangular boxes with rebar dots inside) → do NOT return 1, continue to STEP 12

    STEP 12: Does the page show a vertical floor-by-floor stack where the LEFT SIDE has labels like "11TH FLOOR COLUMN", "10Th Floor column", "GROUND FLOOR COLUMN", or labels with "(MIX: M250)/(STEEL:FE500)", and the bottom has a summary SIZE / STEEL table?
    → YES → RETURN 12

    STEP 13: Does the page show a large grid where COLUMN NAMES (C1, C2, C3... or SW1, SW2...) span the TOP and FLOOR LEVELS span the LEFT, with cross-section DRAWINGS inside every cell?
    → YES, column names are regular C1/C2/C3A type → RETURN 11
    → YES, column names are SW (shear walls) or irregular complex shapes → RETURN 13

    STEP 14: Does the LEFT SIDE show floor labels containing "(MIX: M260)" or "(MIX: M250)" or "TH FLOOR ROOFTOP" or "TH FLOOR COLUMN" stacked top-to-bottom, with cross-section drawings in each row?
    → YES → RETURN 12

    → NONE OF THE ABOVE → RETURN 13
"""

    raw = extract_from_image(first_image, classification_prompt)

    try:
        pattern_number = int(raw.strip())
    except:
        raise Exception(f"Invalid pattern response: {raw}")

    return pattern_number
