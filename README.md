# KeyframeSimplifier
Tool for simplification CVAT keyframes 

# Dev installation (MacOS)

```bash
brew install uv
uv venv
uv sync
uv pip install -e ".[dev]"
```

# Run CLI
```bash
uv run cvat-tool --help

# Default: IoU threshold 0.9 only (no field-based filters)
uv run cvat-tool --job-id 123

# With auto-calculated fields (1% of max distances)
uv run cvat-tool --job-id 123 --auto-percent 1.0

# Manual field configuration
uv run cvat-tool --job-id 123 --field position,5.1,l2 --field rotation,0.0,l_inf

# Restore from backup
uv run cvat-tool --job-id 123 --undo
```

# Run tests
```bash
uv run pytest
uv run pytest -v
uv run pytest -v -s
```

# Run visualization example
```bash
uv run python src/cvat_tool/iou/matplot.py
```


# Advanced Usage Examples

# Default behavior: only IoU 0.9
uv run cvat-tool --job-id 3529725

# Custom IoU threshold only
uv run cvat-tool --job-id 3529725 --iou-threshold 0.8

# Undo
uv run cvat-tool --job-id 3529725 --undo 


# Extra (test needed)
# IoU + auto-calculated fields (1%)
uv run cvat-tool --job-id 3529725 --auto-percent 1.0

# Custom IoU + auto-calculated fields
uv run cvat-tool --job-id 3529725 --iou-threshold 0.8 --auto-percent 1.0

# Only IoU, explicitly disable auto-fields (same as default)
uv run cvat-tool --job-id 3529725 --iou-threshold 0.48 --auto-percent 0

# Disable IoU, only use auto-calculated fields
uv run cvat-tool --job-id 3529725 --iou-threshold 2.0 --auto-percent 1.0

# Custom auto-percent (5%)
uv run cvat-tool --job-id 3529725 --auto-percent 5.0

# Combination of custom IoU and auto-percent
uv run cvat-tool --job-id 3529725 --iou-threshold 0.8 --auto-percent 1.0

# Manual field configuration (overrides auto-calculation)
uv run cvat-tool --job-id 3529725 --field position,5.1,l2 --field rotation,0.0,l_inf
