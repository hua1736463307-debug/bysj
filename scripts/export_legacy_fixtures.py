"""Export fixed historical comparison inputs for offline tests, never runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from legacy_schema import BASELINE, REPOSITORY

FILES = ('backend/database.py', 'backend/dataset_versioning.py', 'backend/job_store.py',
         'backend/utils/archive_utils.py', 'backend/annotation_drafts.py', 'backend/model_registry.py', 'backend/yolo_api.py')


def export(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'baseline': BASELINE, 'files': {}}
    for relative in FILES:
        data = subprocess.check_output(['git', '-C', str(REPOSITORY), 'show', f'{BASELINE}:{relative}'])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest['files'][relative] = hashlib.sha256(data).hexdigest()
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(export(parser.parse_args().output), indent=2))
