# Image post-processing: thumbnails, black-and-white, standardized sizes.
# All targets are real files (non-phony) — make only rebuilds what changed.
#
# Usage:
#   make              # build all derivatives
#   make tn           # just thumbnails
#   make bw           # just black-and-white
#   make resized      # just standardized 1200x630 (OpenGraph)

PYTHON := uv run python
CONVERT := magick

# Source images
SRCS := $(wildcard output/*.png)

# Derivatives
TN_DIR   := output/tn
BW_DIR   := output/bw
OG_DIR   := output/resized

TN_IMGS   := $(patsubst output/%.png,$(TN_DIR)/%.png,$(SRCS))
BW_IMGS   := $(patsubst output/%.png,$(BW_DIR)/%.png,$(SRCS))
OG_IMGS   := $(patsubst output/%.png,$(OG_DIR)/%.png,$(SRCS))

# ── All ──────────────────────────────────────────────────────────────────────

all: tn bw resized

tn: $(TN_IMGS)
bw: $(BW_IMGS)
resized: $(OG_IMGS)

# ── Thumbnails (256px wide) ──────────────────────────────────────────────────

$(TN_DIR)/%.png: output/%.png | $(TN_DIR)
	$(CONVERT) $< -resize 256x -quality 85 $@

$(TN_DIR):
	mkdir -p $@

# ── Black and white ─────────────────────────────────────────────────────────

$(BW_DIR)/%.png: output/%.png | $(BW_DIR)
	$(CONVERT) $< -colorspace Gray -quality 90 $@

$(BW_DIR):
	mkdir -p $@

# ── Resized: OpenGraph 1200x630 ─────────────────────────────────────────────
# Standard social/header size. Crops to fill, centers gravity.

$(OG_DIR)/%.png: output/%.png | $(OG_DIR)
	$(CONVERT) $< -resize 1200x630^ -gravity center -extent 1200x630 -quality 90 $@

$(OG_DIR):
	mkdir -p $@

# ── Generate source images ──────────────────────────────────────────────────

generate:
	$(PYTHON) generate.py

generate-force:
	$(PYTHON) generate.py --force

# ── Clean derivatives (not sources) ─────────────────────────────────────────

clean:
	rm -rf $(TN_DIR) $(BW_DIR) $(OG_DIR)

# Only these are phony
.PHONY: all tn bw resized generate generate-force clean
