# Erratum: book names in `reports/e3-manifest.json`

Found 2026-10-01 while building the E3-full corpus. Five `name` labels
do not match the books actually used. Every file matches its
`source_url`, `raw_sha256`, and `sha256`; only the short names are wrong.
No data was corrupted and no score changes.

| `name` in manifest | Gutenberg # | Actual book |
|---|---|---|
| `janeeyre` | 140 | *The Jungle*, Upton Sinclair |
| `twentyk` | 1259 | *Twenty Years After*, Alexandre Dumas |
| `countmonte` | 164 | *Twenty Thousand Leagues under the Sea*, Jules Verne |
| `odyssey` | 8800 | *The Divine Comedy*, Dante (Longfellow translation) |
| `flatland` | 1727 | *The Odyssey*, Homer (Butler translation) |

Notes:
- `wessex` (#3167) is *Wessex Poems and Other Verses* (Hardy), a verse
  collection rather than the prose *Wessex Tales*.
- *Jane Eyre*, *The Count of Monte Cristo*, and *Flatland* were never part
  of any corpus.
- The manifest stays unchanged because it is hash-pinned and checkpoints
  bind to its digest. Treat `source_url` and the hashes as authoritative.
  `scripts/build_e3_full_corpus.py` excludes used books by Gutenberg ID
  and catalog title, so it excludes the books actually used.
