"""
Fetch the third-party code the SOTA adapters import, at pinned versions.

    python -m sota.fetch --root /opt/sota --licensed          # Docker build
    python -m sota.fetch --root "$SOTA_ROOT" cagi aemc_ne      # per job, at runtime
    python -m sota.fetch --root ../third_party --all           # local setup

Nothing here is vendored: this repository and the Docker image are public.
Repos with an open-source license are baked into the image (--licensed).
CAGI (empty LICENSE file) and AEMC-NE (ICLR supplementary zip, no license)
are not redistributed at all — each job downloads them from the official
source, pinned by commit / SHA-256, into its own scratch SOTA_ROOT.
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import urllib.request
import zipfile

# name -> (kind, url, pin, directory under root, license)
SOURCES = {
    "diffputer": ("git", "https://github.com/hengruizhang98/DiffPuter",
                  "2fa55373655b9e910146d94820fc1012da0dfd75", "DiffPuter", "MIT"),
    "cacti":     ("git", "https://github.com/sriramlab/CACTI",
                  "b78bce8505611df982d23f7e4832d0877d09cfe3", "CACTI", "GPL-3.0"),
    "miri":      ("git", "https://github.com/yujhml/MIRI-Imputation",
                  "be49f94c2ebab431a2afe40a0228e01b857fc097", "MIRI-Imputation", "MIT"),
    "newimp":    ("git", "https://github.com/JustusvLiebig/NewImp",
                  "915acfc7d8854ea3a7baf8e4eeb90d85be10f92c", "NewImp", "Apache-2.0"),
    "refidiff":  ("git", "https://github.com/Atik-Ahamed/RefiDiff",
                  "996aab1fc86d9e92f853b816f42123afc9d070fd", "RefiDiff", "Apache-2.0"),
    "cagi":      ("git", "https://github.com/supercocachii/CAGI",
                  "35e543b013f62a979421a2b13e403a8e06da4eac", "CAGI", None),
    "aemc_ne":   ("zip", "https://proceedings.iclr.cc/paper_files/paper/2024/file/"
                         "6cd3ac24cdb789beeaa9f7145670fcae-Supplementary-Conference.zip",
                  "75ab908bca5022fec047cadef9931f24ca31db535dfe891ddd62571d4d817963",
                  "aemc_ne", None),
}
# kfmc, kfsc, altpzf, kcsc fetch nothing: they are self-contained implementations.


def fetch(name, root):
    kind, url, pin, sub, _ = SOURCES[name]
    dest = os.path.join(root, sub)
    if kind == "git":
        if not os.path.isdir(os.path.join(dest, ".git")):
            subprocess.run(["git", "clone", "--quiet", url, dest], check=True)
        subprocess.run(["git", "-C", dest, "checkout", "--quiet", pin], check=True)
        head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        if head != pin:
            raise RuntimeError(f"{name}: HEAD {head} != pinned {pin}")
    else:
        os.makedirs(dest, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
        got = hashlib.sha256(blob).hexdigest()
        if got != pin:
            raise RuntimeError(f"{name}: sha256 {got} != pinned {pin}")
        zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
    print(f"fetched {name} -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", required=True)
    ap.add_argument("--licensed", action="store_true", help="every source with a license")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = list(a.names)
    if a.licensed:
        names += [n for n, s in SOURCES.items() if s[4]]
    if a.all:
        names += list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"nothing to fetch for {unknown} (self-contained or unknown)")
    os.makedirs(a.root, exist_ok=True)
    for n in dict.fromkeys(names):
        fetch(n, a.root)


if __name__ == "__main__":
    main()
