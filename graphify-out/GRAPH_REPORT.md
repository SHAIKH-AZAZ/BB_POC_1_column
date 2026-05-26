# Graph Report - C:\Users\AZAZ\Desktop\POC _ Subh\POC-1\Column\src  (2026-05-26)

## Corpus Check
- Corpus is ~44,845 words - fits in a single context window. You may not need a graph.

## Summary
- 470 nodes · 808 edges · 20 communities (19 shown, 1 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.82)
- Token cost: 90,244 input · 10,027 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Pattern-11 Vision Extractor|Pattern-11 Vision Extractor]]
- [[_COMMUNITY_Pattern 3-7 Level Reshaper|Pattern 3-7 Level Reshaper]]
- [[_COMMUNITY_Pattern-9 Layout Detector|Pattern-9 Layout Detector]]
- [[_COMMUNITY_Pattern-13 Vision Extractor|Pattern-13 Vision Extractor]]
- [[_COMMUNITY_Dispatch & Pattern-14|Dispatch & Pattern-14]]
- [[_COMMUNITY_Pattern-1 Cell Verifier|Pattern-1 Cell Verifier]]
- [[_COMMUNITY_Pattern-10 Layout Extractor|Pattern-10 Layout Extractor]]
- [[_COMMUNITY_Pattern-12 Boundary Detector|Pattern-12 Boundary Detector]]
- [[_COMMUNITY_Extraction Guard  Records|Extraction Guard / Records]]
- [[_COMMUNITY_Conceptual Pipeline Map|Conceptual Pipeline Map]]
- [[_COMMUNITY_Pattern-1 Main|Pattern-1 Main]]
- [[_COMMUNITY_Pattern-8 Cleaners|Pattern-8 Cleaners]]
- [[_COMMUNITY_OpenCV Cell Cropper|OpenCV Cell Cropper]]
- [[_COMMUNITY_Pattern-2 Cleaners|Pattern-2 Cleaners]]
- [[_COMMUNITY_Pattern 9-13 Family|Pattern 9-13 Family]]
- [[_COMMUNITY_Tool-Augmented Vision|Tool-Augmented Vision]]
- [[_COMMUNITY_Grid Detection Concepts|Grid Detection Concepts]]
- [[_COMMUNITY_Rebar Record Builders|Rebar Record Builders]]
- [[_COMMUNITY_JSON Cleaning Util|JSON Cleaning Util]]

## God Nodes (most connected - your core abstractions)
1. `extract_column_schedule()` - 19 edges
2. `extract_column_schedule()` - 18 edges
3. `reshape_columns_to_levels()` - 16 edges
4. `process_page()` - 16 edges
5. `process_pdf()` - 15 edges
6. `extract_with_tools()` - 15 edges
7. `process_pdf()` - 14 edges
8. `process_pdf()` - 14 edges
9. `ExtractionState` - 13 edges
10. `convert_pdf_to_images()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `extract_with_tools()` --calls--> `build_column_record()`  [INFERRED]
  vision_extractor.py → extraction_guard.py
- `process_pdf()` --calls--> `env_enabled()`  [INFERRED]
  main_2.py → pattern_batching.py
- `extract_with_tools()` --calls--> `ExtractionState`  [INFERRED]
  vision_extractor.py → extraction_guard.py
- `_parse_levels_response()` --calls--> `clean_json_string()`  [INFERRED]
  pattern1_cell_verifier.py → extraction_guard.py
- `_parse_verification()` --calls--> `clean_json_string()`  [INFERRED]
  pattern1_cell_verifier.py → extraction_guard.py

## Hyperedges (group relationships)
- **Pattern dispatch pipeline (PDF -> detect -> main_N)** — auto_runner_main, auto_runner_run_pattern, main_1_module, main_2_module, main_3_module [EXTRACTED 1.00]
- **Per-pattern prompt/main module pairing** — main_1_module, prompt_1_doc, main_2_module, prompt_2_doc, main_3_module, prompt_3_doc [EXTRACTED 1.00]
- **Record builder family (column/slab/footing)** — extraction_guard_build_column_record, extraction_guard_build_slab_record, extraction_guard_build_footing_record, extraction_guard_extract_rebar_parts [EXTRACTED 1.00]
- **Pattern-N extractor family** — main_8_module, main_9_module, main_10_module, main_11_module, main_12_module, main_13_module, main_14_module [INFERRED 0.85]
- **Vision extraction pipeline** — pdf_to_images_module, vision_extractor_module, pattern_batching_module [INFERRED 0.85]
- **Prompt-Main per-pattern pairing** — prompt_8_doc, prompt_14_doc, main_8_module, main_14_module [INFERRED 0.75]

## Communities (20 total, 1 thin omitted)

### Community 0 - "Pattern-11 Vision Extractor"
Cohesion: 0.06
Nodes (58): _build_entry(), _build_output_path(), call_vision(), cluster_lines(), _col_sequence_score(), _crop_upscale(), _darkness_peaks(), detect_lines() (+50 more)

### Community 1 - "Pattern 3-7 Level Reshaper"
Cohesion: 0.07
Nodes (40): Convert flat column records into level-centric (Option B) structure.      Input, Convert flat column records into level-centric (Option B) structure.      Input, reshape_columns_to_levels(), clean_column_name(), clean_reinforcement(), clean_size(), clean_stirrups(), load_prompt() (+32 more)

### Community 2 - "Pattern-9 Layout Detector"
Cohesion: 0.07
Nodes (43): _build_vision_prompt(), cluster_1d(), _detect_grid_centres(), _detect_group_labels(), _detect_lap_names(), detect_layout(), _detect_orientation(), extract_from_pdf_page() (+35 more)

### Community 3 - "Pattern-13 Vision Extractor"
Cohesion: 0.10
Nodes (36): _build_entry(), call_vision(), cluster_lines(), _crop_upscale(), _darkness_peaks(), detect_lines(), extend_grid_with_trailing_lines(), extract_col_label_groups() (+28 more)

### Community 4 - "Dispatch & Pattern-14"
Cohesion: 0.10
Nodes (30): main(), run_pattern(), clean_json_string(), clean_mix(), discover_floors(), extract_column_groups(), extract_floor(), fix_alignment() (+22 more)

### Community 5 - "Pattern-1 Cell Verifier"
Cohesion: 0.13
Nodes (30): _cluster(), _coerce_grid_index(), _column_group_key(), _column_ids(), crop_pattern1_cell(), crop_pattern1_level_column(), crop_pattern1_level_row(), detect_levels_from_pattern1_label_crop() (+22 more)

### Community 6 - "Pattern-10 Layout Extractor"
Cohesion: 0.11
Nodes (29): cluster_1d(), _detect_grid_centres_10(), detect_layout_10(), extract_from_pdf_page_10(), _get_all_words(), main(), _median_spacing(), _merge_pages() (+21 more)

### Community 7 - "Pattern-12 Boundary Detector"
Cohesion: 0.11
Nodes (29): call_vision(), cluster_lines(), _crop_upscale(), detect_col_boundaries(), detect_green_row_groups(), detect_row_boundaries(), find_marks_strip_y(), _get_client() (+21 more)

### Community 8 - "Extraction Guard / Records"
Cohesion: 0.14
Nodes (12): _as_int(), _as_list(), build_column_record(), build_footing_record(), build_slab_record(), _dedupe(), extract_rebar_parts(), ExtractionState (+4 more)

### Community 9 - "Conceptual Pipeline Map"
Cohesion: 0.14
Nodes (14): auto_runner.main, run_pattern, Column record schema (column_no/size/reinforcement/stirrups), Pattern detection dispatch, config (paths & API settings), build_column_record, reshape_columns_to_levels, main_1 (Pattern 1 extractor) (+6 more)

### Community 10 - "Pattern-1 Main"
Cohesion: 0.18
Nodes (19): atomic_write_json(), build_level_batch_prompt(), build_level_discovery_context(), build_manifest(), clean_reinforcement(), clean_size(), clean_stirrups(), column_entry_from_record() (+11 more)

### Community 11 - "Pattern-8 Cleaners"
Cohesion: 0.18
Nodes (16): clean_column(), crop_bottom_region(), extract_with_fallback(), has_columns(), load_prompt(), main(), normalize_reinforcement(), normalize_stirrup_dia() (+8 more)

### Community 12 - "OpenCV Cell Cropper"
Cohesion: 0.17
Nodes (15): _cluster(), crop_all_cells(), detect_grid(), _filter_min_spacing(), get_data_col_indices(), get_label_col_index(), cell_cropper.py --------------- Detects the table grid in a column schedule im, Detect the grid and save every cell as a separate PNG.      Returns     ----- (+7 more)

### Community 13 - "Pattern-2 Cleaners"
Cohesion: 0.23
Nodes (13): clean_mix(), clean_reinforcement(), clean_size(), clean_stirrups(), enforce_all_levels(), expand_column_numbers(), format_size(), load_prompt() (+5 more)

### Community 14 - "Pattern 9-13 Family"
Cohesion: 0.18
Nodes (12): main_10 Pattern-10 Extractor, main_11 Pattern-11 Extractor, main_12 Pattern-12 Extractor, main_13 Pattern-13 Extractor, main_9 Generalized Extractor, Prompt 10, Prompt 11, Prompt 12 (+4 more)

### Community 15 - "Tool-Augmented Vision"
Cohesion: 0.25
Nodes (11): main_14 Pattern-14 Extractor, main_8 Pattern-8 Extractor, Pattern1 Cell Verifier, Pattern Batching, Pattern Detector, PDF to Images, Prompt 14, Prompt 8 (+3 more)

### Community 16 - "Grid Detection Concepts"
Cohesion: 0.32
Nodes (8): _cluster, crop_all_cells, detect_grid, _filter_min_spacing, _try_detect_lines, OpenCV table grid detection, ExtractionState, Per-PDF trace JSON file

### Community 17 - "Rebar Record Builders"
Cohesion: 0.67
Nodes (3): build_footing_record, build_slab_record, extract_rebar_parts

## Knowledge Gaps
- **13 isolated node(s):** `build_column_record`, `build_slab_record`, `build_footing_record`, `clean_json_string`, `Visualize Cells` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `reshape_columns_to_levels()` connect `Pattern 3-7 Level Reshaper` to `Pattern-11 Vision Extractor`, `Pattern-9 Layout Detector`, `Pattern-13 Vision Extractor`, `Dispatch & Pattern-14`, `Pattern-10 Layout Extractor`, `Pattern-12 Boundary Detector`, `Extraction Guard / Records`, `Pattern-8 Cleaners`?**
  _High betweenness centrality (0.425) - this node is a cross-community bridge._
- **Why does `extract_column_schedule()` connect `Pattern-11 Vision Extractor` to `Pattern 3-7 Level Reshaper`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `process_pdf()` connect `Pattern-9 Layout Detector` to `Pattern 3-7 Level Reshaper`, `Dispatch & Pattern-14`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `reshape_columns_to_levels()` (e.g. with `process_pdf()` and `extract_column_schedule()`) actually correct?**
  _`reshape_columns_to_levels()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `process_pdf()` (e.g. with `convert_pdf_to_images()` and `extract_with_tools()`) actually correct?**
  _`process_pdf()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cell_cropper.py --------------- Detects the table grid in a column schedule im`, `Merge nearby pixel positions into single median values.`, `Remove lines that are closer than min_spacing to the previous     kept line.  T` to the rest of the system?**
  _116 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Pattern-11 Vision Extractor` be split into smaller, more focused modules?**
  _Cohesion score 0.055523085914669784 - nodes in this community are weakly interconnected._