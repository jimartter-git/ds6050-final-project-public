"""Download Penn Action and extract only labels/*.mat, streaming the archive (no 3.2 GB file)."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import urllib.request

URL = 'https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz'
EXPECTED_CLIPS = 2326  # Official dataset page.


class Progress:
    """File-like wrapper that reports bytes read from the HTTP response."""

    def __init__(self, stream, total, step=100 * 2 ** 20):
        self.stream, self.total, self.step = stream, total, step
        self.read_bytes, self.next_report = 0, step

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.read_bytes += len(chunk)
        if self.read_bytes >= self.next_report:
            total = f' / {self.total / 2 ** 30:.2f} GB' if self.total else ''
            print(f'  {self.read_bytes / 2 ** 30:.2f} GB{total}', file=sys.stderr, flush=True)
            self.next_report += self.step
        return chunk


def extract_labels(stream, labels_dir):
    """Write every <any>/labels/<file>.mat member into labels_dir; return the count."""
    count = 0
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 2 or parts[-2] != 'labels' \
                    or not parts[-1].endswith('.mat'):
                continue
            source = archive.extractfile(member)
            with open(labels_dir / parts[-1], 'wb') as target:
                shutil.copyfileobj(source, target)
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/Penn_Action')
    parser.add_argument('--url', default=URL)
    parser.add_argument('--force', action='store_true', help='replace existing labels')
    args = parser.parse_args(argv)
    labels = Path(args.data_root) / 'labels'
    if labels.exists() and any(labels.iterdir()) and not args.force:
        parser.error(f'{labels} already has files; pass --force to replace them')
    partial = labels.with_name('labels.partial')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    print(f'Streaming {args.url}', file=sys.stderr)
    with urllib.request.urlopen(args.url) as response:
        total = int(response.headers.get('Content-Length') or 0)
        count = extract_labels(Progress(response, total), partial)
    if count == 0:
        shutil.rmtree(partial)
        raise SystemExit('No labels/*.mat found in the archive; inspect its layout manually.')
    shutil.rmtree(labels, ignore_errors=True)
    partial.rename(labels)
    print(f'Extracted {count} label files to {labels}')
    if count != EXPECTED_CLIPS:
        print(f'Warning: expected {EXPECTED_CLIPS} clips; check the download.', file=sys.stderr)


if __name__ == '__main__':
    main()
