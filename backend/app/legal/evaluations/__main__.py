"""python -m app.legal.evaluations import <reports> — records evaluation runs in the configured Law database."""
import sys

from ...core.db import init_db
from . import main

init_db()
sys.exit(main())
