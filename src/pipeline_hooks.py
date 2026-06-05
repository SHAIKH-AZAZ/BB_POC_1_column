"""Canonical extraction pipeline hook points for quality stages.

This file locks current behavior by documenting where to attach:
1) deterministic validation/rules, and
2) confidence + uncertainty metadata generation.

Canonical backbone modules:
- vision_extractor.py
- pattern_batching.py
- pattern_cleaners.py
- extraction_guard.py
- main_1.py ... main_9.py
"""

PATTERN_HOOK_POINTS = {
    1: "main_1.py: after merged level batch columns are cleaned, before final JSON write",
    2: "main_2.py: after final_columns cleanup/enforcement, before reshape_to_levels",
    3: "main_3.py: after final_columns cleanup loop, before reshape_columns_to_levels",
    4: "main_4.py: after final_columns cleanup loop, before reshape_columns_to_levels",
    5: "main_5.py: after final_columns cleanup loop, before reshape_columns_to_levels",
    6: "main_6.py: after final_columns cleanup loop, before reshape_columns_to_levels",
    7: "main_7.py: after final_columns cleanup loop, before reshape_columns_to_levels",
    8: "main_8.py: after clean_column loop, before reshape_columns_to_levels",
    9: "main_9.py: after standardize_records, before reshape_columns_to_levels",
}
