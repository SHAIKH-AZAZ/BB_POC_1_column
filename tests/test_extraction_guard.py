import os
import sys
import unittest


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from extraction_guard import (  # noqa: E402
    ExtractionState,
    _coerce_stirrups_to_strings,
    build_column_record,
    build_footing_record,
    build_slab_record,
)


def minimal_think(expected_count=1):
    return {
        "image_quality": "clear",
        "table_bounds": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        "expected_count": expected_count,
        "zoom_plan": [],
        "normalization_rules": [],
        "extraction_order": [],
    }


class ExtractionGuardTests(unittest.TestCase):
    def test_reject_add_before_think(self):
        state = ExtractionState("column", "page_1.png", "columns", ["column_no", "column_name"])
        ok, message = state.can_add_record({"source_region_ids": ["h1"]})
        self.assertFalse(ok)
        self.assertIn("think", message)

    def test_reject_unconfirmed_source_region(self):
        state = ExtractionState("column", "page_1.png", "columns", ["column_no", "column_name"])
        state.handle_think(minimal_think())
        ok, message = state.can_add_record({"source_region_ids": ["h1"]})
        self.assertFalse(ok)
        self.assertIn("confirmed", message)

    def test_accept_confirmed_source_region(self):
        state = ExtractionState("column", "page_1.png", "columns", ["column_no", "column_name"])
        state.handle_think(minimal_think())
        state.handle_zoom(
            {
                "region_id": "h1",
                "purpose": "header",
                "x1": 0,
                "y1": 0,
                "x2": 0.1,
                "y2": 0.1,
                "reason": "header",
            }
        )
        ok, _ = state.handle_confirm_read(
            {"region_id": "h1", "text": "C1,C67", "confidence": "high", "applies_to": "header"}
        )
        self.assertTrue(ok)
        ok, _ = state.can_add_record({"source_region_ids": ["h1"]})
        self.assertTrue(ok)

    def test_expected_count_and_duplicate_warnings(self):
        state = ExtractionState("slab", "page_1.png", "slabs", ["slab_id"])
        state.handle_think(minimal_think(expected_count=2))
        records = [{"slab_id": "S1"}, {"slab_id": "S1"}]
        state.validate(records)
        warnings = " ".join(state.warnings)
        self.assertIn("duplicate", warnings)

    def test_column_schema_builder(self):
        record = build_column_record(
            {
                "column_no": "C1",
                "storey_level": "GROUND FLOOR",
                "width": 700,
                "depth": 700,
                "reinforcement": ["20-T20"],
                "ties_dia": "T8",
                "ties_spacing": "150 C/C",
            }
        )
        self.assertEqual(record["column_name"], "GROUND FLOOR")
        self.assertEqual(record["stirrups"]["dia"], ["T8"])

    def test_column_schema_builder_accepts_length(self):
        record = build_column_record(
            {
                "column_no": "C1",
                "storey_level": "GROUND FLOOR",
                "width": 200,
                "depth": None,
                "length": 600,
                "reinforcement": [],
            }
        )
        self.assertEqual(
            record["size"],
            {"width": 200, "depth": None, "length": 600},
        )

    def test_stirrups_coercion_preserves_multiple_diameters_as_lists(self):
        stirrups = _coerce_stirrups_to_strings(
            {"dia": ["T10", "T8"], "spacing": ["100 C/C", "150 C/C"]}
        )
        self.assertEqual(stirrups["dia"], ["T10", "T8"])
        self.assertEqual(stirrups["spacing"], ["100 C/C", "150 C/C"])

    def test_slab_schema_builder(self):
        record = build_slab_record(
            {
                "slab_id": "S1",
                "thickness": 150,
                "steel_along_span": ["T12 @ 100 C/C ALT BENT UP"],
                "steel_across_span": ["T8 @ 150 C/C DIST"],
            }
        )
        self.assertEqual(record["slab_id"], "S1")
        self.assertIn("T12", record["reinforcement"]["dia"])
        self.assertIn("150 C/C", record["reinforcement"]["spacing"])

    def test_footing_schema_builder(self):
        record = build_footing_record(
            {
                "column_id": "GC1",
                "plan_length": 1900,
                "plan_width": 2000,
                "depth_top": 450,
                "depth_bottom": 230,
                "short_span_reinf": "T12 @ 115 C/C",
                "long_span_reinf": "T10 @ 150 C/C",
                "concrete_mix": "M25",
            }
        )
        self.assertEqual(record["size"]["depth"], 230)
        self.assertIn("T12", record["reinforcement"]["dia"])
        self.assertEqual(record["mix"], "M25")


if __name__ == "__main__":
    unittest.main()
