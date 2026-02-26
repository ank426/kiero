import argparse
import shutil
import tempfile
from pathlib import Path

from kiero.batch import detect_batch, inpaint_batch, run_batch
from kiero.utils import (
    extract_cbz,
    is_cbz,
    load_mask,
    make_detector,
    make_inpainter,
    make_pipeline,
    validate_detect,
    validate_inpaint,
    validate_run,
    write_cbz,
)
