import re
import sys
from pathlib import Path

# Jednoliterowe polskie spójniki/przyimki.
PATTERN = re.compile(r'(?<![\\\w])([AaIiOoUuWwZz])\s+(?=[^\s%])', re.UNICODE)

def fix_file(path: Path):
    text = path.read_text(encoding="utf-8")
    fixed = PATTERN.sub(r'\1~', text)

    if fixed != text:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")
        path.write_text(fixed, encoding="utf-8")
        print(f"Poprawiono: {path}  | backup: {backup}")
    else:
        print(f"Bez zmian: {path}")

if len(sys.argv) < 2:
    print("Użycie: python fix-sieroty.py plik.tex [kolejny.tex ...]")
    sys.exit(1)

for filename in sys.argv[1:]:
    fix_file(Path(filename))