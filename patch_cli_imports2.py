import argparse
import shutil
import tempfile
from pathlib import Path

from kiero.batch import detect_batch, inpaint_batch, run_batch
from kiero.utils import (
    extract_cbz,
    write_cbz,
    is_cbz,
    is_image,
    load_mask,
    make_detector,
    make_inpainter,
    make_pipeline,
    require_exists,
    validate_detect,
    validate_inpaint,
    validate_run,
)
