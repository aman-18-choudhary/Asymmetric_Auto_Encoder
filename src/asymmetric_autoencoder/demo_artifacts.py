"""Fetch, verify and place the live-demo artifacts (three checkpoints and two demo assets).

The files are not stored in Git. They are published as assets of a GitHub release and listed in
``demo/artifacts.json`` together with their exact size and SHA-256. Rules:

* Only files named in the manifest are fetched, only over HTTPS (loopback HTTP is allowed for
  tests), and only into a fixed allow-list of destinations (no path traversal).
* A file that already exists is never touched, so local checkpoints are never overwritten.
* Redirects (GitHub serves release assets through one) are followed only over HTTPS and only to
  ``github.com`` or ``release-assets.githubusercontent.com``; anything else is refused.
* Downloads stream into a temporary file and are moved into place only after the size and SHA-256
  match; a mismatch deletes the temporary file and raises ``ArtifactIntegrityError``.
* No credentials are used or required.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20
_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_DEST = re.compile(
    r"^(experiments/results/[a-z0-9_]+_seed0/checkpoints/best\.pt"
    r"|demo/assets/(val_samples\.npz|pca_bases\.pt))$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
USER_AGENT = "asymmetric-autoencoder-demo"
REDIRECT_HOSTS = frozenset({"github.com", "release-assets.githubusercontent.com"})


class ArtifactError(RuntimeError):
    """Base class: the live-demo files could not be provided."""


class ArtifactConfigError(ArtifactError):
    """The manifest or a URL is invalid."""


class ArtifactDownloadError(ArtifactError):
    """A download failed (network, HTTP status, truncated or unwritable)."""


class ArtifactIntegrityError(ArtifactError):
    """A downloaded file did not match its pinned size or SHA-256."""


def check_redirect(url: str) -> None:
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or parts.hostname not in REDIRECT_HOSTS:
        raise ArtifactDownloadError(
            f"refusing redirect to {parts.scheme}://{parts.hostname} "
            f"(allowed: HTTPS to {', '.join(sorted(REDIRECT_HOSTS))})"
        )


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to HTTPS URLs on the allow-listed hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_redirect(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener(*handlers) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(*handlers, SafeRedirectHandler)


def safe_urlopen(request, timeout: float = 60):
    return build_safe_opener().open(request, timeout=timeout)


@dataclass(frozen=True)
class Artifact:
    name: str
    dest: str  # relative to the project / runtime root
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    base_url: str
    artifacts: tuple[Artifact, ...]

    def url(self, artifact: Artifact) -> str:
        return f"{self.base_url.rstrip('/')}/{urllib.parse.quote(artifact.name)}"


def check_url(url: str) -> None:
    parts = urllib.parse.urlparse(url)
    ok = parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK)
    if not ok or not parts.hostname:
        raise ArtifactConfigError(f"refusing non-HTTPS artifact URL: {url!r}")


def load_manifest(path: str | Path) -> Manifest:
    """Read and strictly validate the manifest."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactConfigError(f"cannot read artifact manifest {path}: {exc}") from exc
    base = raw.get("base_url")
    if not isinstance(base, str):
        raise ArtifactConfigError("manifest needs a string 'base_url'")
    check_url(base)
    artifacts, seen = [], set()
    for entry in raw.get("files", []):
        name, dest = entry.get("name"), entry.get("dest")
        size, digest = entry.get("size"), entry.get("sha256")
        if not (isinstance(name, str) and _NAME.match(name)):
            raise ArtifactConfigError(f"invalid artifact name: {name!r}")
        if name in seen:
            raise ArtifactConfigError(f"duplicate artifact name: {name}")
        if not (isinstance(dest, str) and _DEST.match(dest)):
            raise ArtifactConfigError(f"destination not allowed: {dest!r}")
        if not (isinstance(size, int) and not isinstance(size, bool) and size > 0):
            raise ArtifactConfigError(f"invalid size for {name}: {size!r}")
        if not (isinstance(digest, str) and _SHA256.match(digest)):
            raise ArtifactConfigError(f"invalid sha256 for {name}: {digest!r}")
        seen.add(name)
        artifacts.append(Artifact(name, dest, size, digest))
    if not artifacts:
        raise ArtifactConfigError("manifest lists no files")
    return Manifest(base, tuple(artifacts))


def missing(manifest: Manifest, root: Path) -> list[Artifact]:
    return [a for a in manifest.artifacts if not (Path(root) / a.dest).is_file()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(
    manifest: Manifest,
    artifact: Artifact,
    root: Path,
    opener: Callable = safe_urlopen,
    timeout: float = 60,
) -> Path:
    """Download one artifact into ``root`` unless it already exists. Never overwrites."""
    dest = Path(root) / artifact.dest
    if dest.exists():
        return dest
    url = manifest.url(artifact)
    check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ArtifactDownloadError(f"{artifact.name}: server answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise ArtifactDownloadError(f"{artifact.name}: download failed ({exc})") from exc
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=dest.name + ".", suffix=".part", delete=False
        )
    except OSError as exc:
        response.close()
        raise ArtifactDownloadError(f"cannot write {artifact.dest}: {exc}") from exc
    tmp = Path(handle.name)
    try:
        digest, total = hashlib.sha256(), 0
        with handle, contextlib.closing(response):
            while chunk := response.read(CHUNK):
                total += len(chunk)
                if total > artifact.size:
                    raise ArtifactIntegrityError(
                        f"{artifact.name}: larger than the expected {artifact.size} bytes"
                    )
                digest.update(chunk)
                handle.write(chunk)
        if total != artifact.size:
            raise ArtifactIntegrityError(
                f"{artifact.name}: got {total} bytes, expected {artifact.size} (truncated download)"
            )
        if digest.hexdigest() != artifact.sha256:
            raise ArtifactIntegrityError(
                f"{artifact.name}: SHA-256 mismatch (expected {artifact.sha256[:12]}..., "
                f"got {digest.hexdigest()[:12]}...); the file was rejected"
            )
        os.replace(tmp, dest)
    except ArtifactError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise ArtifactDownloadError(f"{artifact.name}: download interrupted ({exc})") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def ensure_artifacts(
    manifest: Manifest,
    root: Path,
    opener: Callable = safe_urlopen,
    timeout: float = 60,
) -> list[Path]:
    """Fetch every missing artifact; return the paths that were downloaded."""
    return [fetch_one(manifest, a, root, opener, timeout) for a in missing(manifest, Path(root))]


def _writable(directory: Path) -> bool:
    try:
        with tempfile.TemporaryFile(dir=directory):
            return True
    except OSError:
        return False


def prepare_runtime_root(project_root: Path, override: str | Path | None = None) -> Path:
    """Directory that holds the live-demo files.

    The project root itself when it is writable (the usual case, including local development).
    Otherwise (or when ``override`` is given) a temporary mirror that contains the tracked
    ``config.yaml`` of every run, so checkpoints can be placed next to their configs.
    """
    project_root = Path(project_root)
    if override is None and _writable(project_root / "demo"):
        return project_root
    mirror = Path(override) if override else Path(tempfile.gettempdir()) / "aae_demo_runtime"
    for config in (project_root / "experiments" / "results").glob("*/config.yaml"):
        target = mirror / "experiments" / "results" / config.parent.name / "config.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(config, target)
    return mirror
