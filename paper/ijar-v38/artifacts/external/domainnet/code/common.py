"""S1 v2 shared constants. Spec: outputs/fedcore_s1v2_prereg_20260925/PROTOCOL.json (sha256 below)."""
import hashlib, json
PID = 'FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2'
PROTOCOL_SHA256 = '06dd7e1a22f3d504dc76dd115b3f1803754ffe022a9d436e7dcef8a20c7c142b'
V1_PID = 'FEDCORE-V1-PACS-20260916-v1'
V1_PROTOCOL_SHA256 = 'a8c944d57246d2fc46b175982cdada76f1c797da30090a35fbb344bad0cb7c50'
V1_CODE_SHA256 = {
    'v1_pacs_adapter.py': 'ceff2d0e3138cd99f96262e0d7193af5b96895118cc55c820aed6b6d0ba7871e',
    'v1_pacs_analysis.py': 'a8c7d45483c1f52ae11b8d65c95271a205267a153958b171d959b004e1afb7c2',
    'v1_pacs_certifier.py': 'e0e446452c0f9cbd7ceb75221a842083be10f12c33b809e12d302816dc64d954',
    'v1_pacs_gpu_check.py': '7abe4e1793a15d622a55cf3d2d30220f9093ff774cdc1e657077cdffcb24ffbd',
    'v1_pacs_preflight.py': '9313123dd0278d32095e6f91e740a0b0cc3412c61549f5531fe2dcefb456b367',
    'v1_pacs_runner.py': '4442fa54193f563f84b6456b88c3332ab8a7c9b075a1886834fd417e66fba223',
    'v1_pacs_runner_tests.py': '0ed12a957e65acc70e0feaac894306da555bcd98110ce5b22184bfa4f5e95b84',
    'v1_pacs_synthetic.py': 'ac23ebf179629255efbab1e3092421dcb79393e6c15e4ccc16c9810ec4b90095'}
DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
V1_SPLITS = [  # copied verbatim from V1 PROTOCOL
    {'split': 0, 'known': ['dog', 'elephant', 'giraffe', 'guitar'], 'unknown': ['horse', 'house', 'person']},
    {'split': 1, 'known': ['dog', 'elephant', 'house', 'person'], 'unknown': ['giraffe', 'guitar', 'horse']},
    {'split': 2, 'known': ['elephant', 'giraffe', 'horse', 'person'], 'unknown': ['dog', 'guitar', 'house']},
    {'split': 3, 'known': ['dog', 'guitar', 'horse', 'house'], 'unknown': ['elephant', 'giraffe', 'person']},
    {'split': 4, 'known': ['dog', 'elephant', 'guitar', 'house'], 'unknown': ['giraffe', 'horse', 'person']}]
PERSON_UNKNOWN_SPLITS = [0, 3, 4]
PERSON_KNOWN_SPLITS = [1, 2]
MIN_SUPPORT = 10
MISSING_TOL_FRAC = 0.005
BOOT_SEED = 2026092501


def label_map(split):
    s = V1_SPLITS[split]
    return {c: (s['known'].index(c) if c in s['known'] else -1) for c in CLASSES}


def role_of(identity):
    H = int(hashlib.sha256((PID + '|role|' + identity).encode()).hexdigest(), 16)
    return 'proposal' if 2 * H < 2 ** 256 else 'audit'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str); f.write('\n')
