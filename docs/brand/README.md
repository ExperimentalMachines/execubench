# ExecuBench brand kit

The mark is the **Gauge**: a scale of three faceted segments parted by open cuts, and an ember
needle reading high. The artwork files below are written by `tools/design/mark.py`. Change the numbers there and
rerun it (`--png` also renders the PNGs). Do not edit the SVGs by hand.

| File | Use |
|---|---|
| `mark.svg`, `mark-dark.svg` | The mark on light and on dark grounds |
| `mark-small.svg`, `mark-small-dark.svg` | Below 40 px: the same drawing with a wider gap round the needle |
| `mark-mono.svg` | One colour; the cut keeps the needle apart from the scale |
| `lockup.svg`, `lockup-dark.svg` | Mark and wordmark (Red Hat Display Bold, outlined) |
| `icon.svg` / `.png` | 512 px square icon for the GitHub and Hugging Face avatars, opaque ink |
| `social-preview.svg` / `.png` | 1280 x 640 GitHub social preview |
| `mark.png` | 1024 px mark on transparency |

**Family.** ExecuPack makes the model files, ExecuServe runs them on the phone, and ExecuBench
measures them there. The three marks share a palette, the open cut, a single ember part and the
lockup. ExecuServe's Block and ExecuPack's Lidded Box are isometric objects; the Gauge is drawn
front-on, and like them it is built only from straight lines, so its arcs are faceted on purpose.

**Colour.** Ink `#262626` or paper `#F3F4F7` for the scale, ember `#EE4C2C` for the needle and
nowhere else in the mark.

**Lockup.** The sibling lockups fit their mark into a square 2.15 cap heights on a side
(ExecuServe's Block comes out 2.15 wide and 1.94 tall, ExecuPack's box 2.09 wide and 2.15 tall).
The Gauge is about one and a half times as wide as it is tall, so in that square it would
outweigh the word. It is set 1.6 cap heights tall instead (about 2.4 wide, close to the siblings'
area) and a tenth of a cap height low, so its weight lines up with the word.

**Don't** smooth the facets, close the cuts, move the needle off the scale's upper half, add a
second ember part, or put PyTorch's or ExecuTorch's logo inside or beside the mark as one unit.
Credit ExecuTorch in text.

The wordmark's font, Red Hat Display, is vendored in `tools/design/fonts/` under the SIL Open
Font License 1.1 (`LICENSES/OFL-1.1.txt`).
