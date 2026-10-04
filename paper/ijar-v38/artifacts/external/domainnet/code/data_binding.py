"""Write DATA_BINDING.json/.md: sha256 + size of the 6 official zips vs HEAD Content-Length in PROTOCOL.json. Not an analysis choice."""
import json, os, sys, time
from common import *
zips, out, proto = sys.argv[1], sys.argv[2], sys.argv[3]
P = json.load(open(proto)); assert sha(proto) == PROTOCOL_SHA256
rows = {}
for d in DOMAINS:
    p = os.path.join(zips, d + '.zip'); t0 = time.time()
    rows[d] = {'url': P['dataset']['zip_urls'][d], 'path': p, 'bytes': os.path.getsize(p), 'expected_bytes_HEAD': P['dataset']['zip_content_length_from_HEAD_at_seal'][d], 'sha256': sha(p)}
    rows[d]['size_match'] = rows[d]['bytes'] == rows[d]['expected_bytes_HEAD']
ok = all(r['size_match'] for r in rows.values())
b = {'addendum_to': PID, 'protocol_sha256': PROTOCOL_SHA256, 'note': 'DATA_BINDING addendum: records archive identity only; not an analysis choice.', 'G1_DOWNLOAD': 'PASS' if ok else 'FAIL', 'zips': rows,
     'label_lists_sha256': P['dataset']['label_list_sha256_cleaned']}
os.makedirs(out, exist_ok=True); dump(os.path.join(out, 'DATA_BINDING.json'), b)
with open(os.path.join(out, 'DATA_BINDING.md'), 'w') as f:
    f.write(f'# DATA_BINDING addendum — {PID}\n\nProtocol sha256 `{PROTOCOL_SHA256}`. Records archive identity only; not an analysis choice.\n\n| domain | URL | bytes | HEAD bytes | match | sha256 |\n|---|---|---:|---:|---|---|\n')
    for d, r in rows.items():
        f.write(f"| {d} | {r['url']} | {r['bytes']} | {r['expected_bytes_HEAD']} | {r['size_match']} | `{r['sha256']}` |\n")
    f.write(f"\nG1_DOWNLOAD: **{b['G1_DOWNLOAD']}**\n")
print(json.dumps({d: (r['sha256'], r['size_match']) for d, r in rows.items()}))
