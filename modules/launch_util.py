import os

import subprocess
from functools import lru_cache


git = os.environ.get('GIT', "git")
fooocus_tag = '1.0.0'


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output([git, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"

