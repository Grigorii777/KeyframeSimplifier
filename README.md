# KeyframeSimplifier
Tool for simplification CVAT keyframes 

# Dev installation (MacOS)

```bash
brew install uv
uv venv
uv sync
uv pip install -e ".[dev]"
```

# Help
```bash
uv run cvat-tool --help
```

# Run tests
```bash
uv run pytest -v -s
```

# Run visualization example
```bash
uv run python src/cvat_tool/iou/matplot.py
```


# Advanced Usage Examples

## IoU-based simplification (Main functionality)

```bash
# Default: IoU threshold 0.9
uv run cvat-tool --job-id 3529725

# Custom IoU threshold
uv run cvat-tool --job-id 3529725 --iou-threshold 0.7

# Restore from backup
uv run cvat-tool --job-id 3529725 --undo
```

## Field-based simplification (Additional, NOT tested with IoU)

```bash
# Auto-calculated fields (1% of max distances)
uv run cvat-tool --job-id 3529725 --iou-threshold 2.0 --auto-percent 1.0

# Custom auto-percent (5%)
uv run cvat-tool --job-id 3529725 --iou-threshold 2.0 --auto-percent 5.0

# Manual field configuration
uv run cvat-tool --job-id 3529725 --iou-threshold 2.0 --field position,5.1,l2 --field rotation,0.0,l_inf
```

**Note:** When using field-based simplification, set `--iou-threshold 2.0` to effectively disable IoU filtering.
