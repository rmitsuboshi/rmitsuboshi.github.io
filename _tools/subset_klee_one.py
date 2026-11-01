"""Subset Klee One to the characters used on the Japanese page.

The full fonts are about 9 MB each; the page needs a few hundred glyphs.
Run this again after adding new Japanese characters to index-jp.html,
otherwise those characters fall back to another font.

    python3 -m venv /tmp/fontenv
    /tmp/fontenv/bin/pip install 'fonttools[woff]'
    curl -LO https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-Regular.ttf
    curl -LO https://github.com/google/fonts/raw/main/ofl/kleeone/KleeOne-SemiBold.ttf
    /tmp/fontenv/bin/python _tools/subset_klee_one.py .

Jekyll does not publish directories starting with `_`, so this script is
not served by GitHub Pages.
"""
import pathlib
import sys

from fontTools import subset

root = pathlib.Path(__file__).resolve().parent.parent
source = pathlib.Path(sys.argv[1])

# Latin text is rendered with Play, so keep only the characters Play lacks
# that appear on the page, plus all kana and CJK punctuation so that small
# edits rarely need a rerun.
text = (root / 'index-jp.html').read_text(encoding='utf-8')
chars = {c for c in text if ord(c) > 0x7F}
chars |= {chr(c) for c in range(0x3000, 0x3100)}

css = ["/* Klee One (SIL Open Font License 1.1, see OFL-KleeOne.txt), subset to\n"
       "   the characters of index-jp.html by _tools/subset_klee_one.py. */"]
for weight, name in [(400, 'Regular'), (600, 'SemiBold')]:
    out = root / 'fonts' / f'klee-one-{weight}.woff2'
    options = subset.Options()
    options.flavor = 'woff2'
    font = subset.load_font(source / f'KleeOne-{name}.ttf', options)
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=''.join(sorted(chars)))
    subsetter.subset(font)
    subset.save_font(font, out, options)
    print(f'{out.name}: {out.stat().st_size // 1024} KB')
    css.append(
        "@font-face {\n"
        "  font-family: 'Klee One';\n"
        "  font-style: normal;\n"
        f"  font-weight: {weight};\n"
        "  font-display: swap;\n"
        f"  src: url(klee-one-{weight}.woff2) format('woff2');\n"
        "}")
(root / 'fonts' / 'klee-one.css').write_text('\n'.join(css) + '\n')
print(f'{len(chars)} characters')
