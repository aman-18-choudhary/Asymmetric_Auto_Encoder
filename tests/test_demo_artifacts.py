import hashlib
import io
import json
import threading
import urllib.request
import urllib.response
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from asymmetric_autoencoder import demo_artifacts as da

ROOT = Path(__file__).resolve().parents[1]
DEST = "experiments/results/demo_cifar10_seed0/checkpoints/best.pt"
DEST2 = "demo/assets/val_samples.npz"


class _Server:
    """Local HTTP server for the download tests: serves ``files`` and counts requests."""

    def __init__(self, files: dict[str, bytes], truncate: bool = False, status: int = 200):
        outer = self
        self.hits: list[str] = []
        self.files, self.truncate, self.status = files, truncate, status

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.hits.append(self.path)
                name = self.path.rsplit("/", 1)[-1]
                if outer.status != 200 or name not in outer.files:
                    self.send_response(outer.status if outer.status != 200 else 404)
                    self.end_headers()
                    return
                body = outer.files[name]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[: len(body) // 2] if outer.truncate else body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/rel"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def make_server():
    servers = []

    def make(*args, **kwargs):
        servers.append(_Server(*args, **kwargs))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


def _manifest(url: str, payloads: dict[str, bytes], sha_override: str | None = None):
    arts = tuple(
        da.Artifact(name, dest, len(data), sha_override or hashlib.sha256(data).hexdigest())
        for (name, dest), data in zip(
            (("a.pt", DEST), ("b.npz", DEST2)), payloads.values(), strict=True
        )
    )
    return da.Manifest(url, arts)


PAYLOADS = {"a.pt": b"weights" * 50, "b.npz": b"images" * 70}


def _no_leftovers(root: Path):
    assert not list(root.rglob("*.part"))


# ---------------------------------------------------------------- the repository's manifest
def test_repo_manifest_is_valid_and_points_at_this_repositorys_release():
    m = da.load_manifest(ROOT / "demo" / "artifacts.json")
    assert m.base_url.startswith("https://github.com/aman-18-choudhary/Asymmetric_Auto_Encoder/")
    assert "/releases/download/" in m.base_url
    names = [a.name for a in m.artifacts]
    assert len(names) == 5 == len(set(names)) and len({a.dest for a in m.artifacts}) == 5
    assert sum(a.size for a in m.artifacts) < 30_000_000  # about 24 MB in total


def test_repo_manifest_matches_the_local_artifacts_when_present():
    m = da.load_manifest(ROOT / "demo" / "artifacts.json")
    present = [a for a in m.artifacts if (ROOT / a.dest).is_file()]
    if not present:
        pytest.skip("local artifacts not present")
    for a in present:
        path = ROOT / a.dest
        assert path.stat().st_size == a.size, a.name
        assert da.sha256_file(path) == a.sha256, a.name


def _write(tmp_path, **overrides):
    entry = {"name": "a.pt", "dest": DEST, "size": 10, "sha256": "0" * 64}
    entry.update(overrides)
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"base_url": "https://github.com/x/y/releases/download/t", "files": [entry]})
    )
    return path


@pytest.mark.parametrize(
    "bad",
    [
        {"dest": "../outside.pt"},
        {"dest": "experiments/results/x/../../evil.pt"},
        {"dest": "reports/figures/overwrite.png"},
        {"dest": "experiments/results/demo_seed0/eval_test.json"},
        {"name": "a/b.pt"},
        {"name": ""},
        {"size": 0},
        {"size": -5},
        {"size": "10"},
        {"sha256": "XYZ"},
        {"sha256": "A" * 64},
    ],
)
def test_manifest_rejects_unsafe_or_malformed_entries(tmp_path, bad):
    with pytest.raises(da.ArtifactConfigError):
        da.load_manifest(_write(tmp_path, **bad))


def test_manifest_rejects_plain_http_missing_and_duplicate_entries(tmp_path):
    for base in ("http://example.com/x", "ftp://example.com/x", "file:///etc/passwd"):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"base_url": base, "files": []}))
        with pytest.raises(da.ArtifactConfigError):
            da.load_manifest(path)
    with pytest.raises(da.ArtifactConfigError):
        da.load_manifest(tmp_path / "does_not_exist.json")
    entry = {"name": "a.pt", "dest": DEST, "size": 10, "sha256": "0" * 64}
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"base_url": "https://github.com/x", "files": [entry, entry]}))
    with pytest.raises(da.ArtifactConfigError):
        da.load_manifest(path)


# ---------------------------------------------------------------- downloading
def test_downloads_verify_and_place_files_then_skip_what_exists(tmp_path, make_server):
    server = make_server(PAYLOADS)
    manifest = _manifest(server.url, PAYLOADS)
    got = da.ensure_artifacts(manifest, tmp_path)
    assert sorted(p.name for p in got) == ["best.pt", "val_samples.npz"]
    assert (tmp_path / DEST).read_bytes() == PAYLOADS["a.pt"]
    assert (tmp_path / DEST2).read_bytes() == PAYLOADS["b.npz"]
    assert len(server.hits) == 2
    _no_leftovers(tmp_path)
    assert da.ensure_artifacts(manifest, tmp_path) == []  # nothing missing -> no network
    assert len(server.hits) == 2


def test_existing_files_are_never_overwritten_or_even_requested(tmp_path, make_server):
    server = make_server(PAYLOADS)
    (tmp_path / DEST).parent.mkdir(parents=True)
    (tmp_path / DEST).write_bytes(b"my own retrained checkpoint")
    da.ensure_artifacts(_manifest(server.url, PAYLOADS), tmp_path)
    assert (tmp_path / DEST).read_bytes() == b"my own retrained checkpoint"
    assert [h.rsplit("/", 1)[-1] for h in server.hits] == ["b.npz"]  # only the missing one


def test_hash_mismatch_is_rejected_and_nothing_is_left_behind(tmp_path, make_server):
    server = make_server(PAYLOADS)
    with pytest.raises(da.ArtifactIntegrityError, match="SHA-256 mismatch"):
        da.ensure_artifacts(_manifest(server.url, PAYLOADS, sha_override="1" * 64), tmp_path)
    assert not (tmp_path / DEST).exists()
    _no_leftovers(tmp_path)


def test_wrong_size_downloads_are_rejected(tmp_path, make_server):
    bigger = make_server({"a.pt": PAYLOADS["a.pt"] + b"extra", "b.npz": PAYLOADS["b.npz"]})
    with pytest.raises(da.ArtifactIntegrityError, match="larger than"):
        da.ensure_artifacts(_manifest(bigger.url, PAYLOADS), tmp_path)
    smaller = make_server({"a.pt": PAYLOADS["a.pt"][:-5], "b.npz": PAYLOADS["b.npz"]})
    with pytest.raises(da.ArtifactIntegrityError, match="truncated|expected"):
        da.ensure_artifacts(_manifest(smaller.url, PAYLOADS), tmp_path)
    _no_leftovers(tmp_path)
    assert not (tmp_path / DEST).exists()


def test_interrupted_connection_is_a_clear_error_naming_the_file(tmp_path, make_server):
    server = make_server(PAYLOADS, truncate=True)
    with pytest.raises(da.ArtifactError, match=r"a\.pt.*truncated"):
        da.ensure_artifacts(_manifest(server.url, PAYLOADS), tmp_path)
    _no_leftovers(tmp_path)
    assert not (tmp_path / DEST).exists()


def test_http_errors_and_unreachable_hosts_are_clear_errors(tmp_path, make_server):
    with pytest.raises(da.ArtifactDownloadError, match="HTTP 404"):
        da.ensure_artifacts(_manifest(make_server({}).url, PAYLOADS), tmp_path)
    with pytest.raises(da.ArtifactDownloadError, match="HTTP 500"):
        da.ensure_artifacts(_manifest(make_server(PAYLOADS, status=500).url, PAYLOADS), tmp_path)
    dead = _manifest("http://127.0.0.1:9/none", PAYLOADS)  # nothing listens on port 9
    with pytest.raises(da.ArtifactDownloadError, match="download failed"):
        da.ensure_artifacts(dead, tmp_path, timeout=3)
    _no_leftovers(tmp_path)


def test_non_https_remote_urls_are_refused_before_any_request(tmp_path):
    manifest = _manifest("http://example.com/x", PAYLOADS)
    called = []
    with pytest.raises(da.ArtifactConfigError):
        da.ensure_artifacts(manifest, tmp_path, opener=lambda *a, **k: called.append(a))
    assert not called


class _FakeHTTPS(urllib.request.HTTPSHandler):
    """Answers HTTPS requests from a table of url -> (status, headers, body); no network."""

    def __init__(self, table):
        super().__init__()
        self.table, self.seen = table, []

    def https_open(self, req):
        self.seen.append(req.full_url)
        status, headers, body = self.table[req.full_url]
        message = Message()
        for key, value in headers.items():
            message[key] = value
        response = urllib.response.addinfourl(io.BytesIO(body), message, req.full_url, status)
        response.msg = "OK" if status == 200 else "Found"
        return response


BASE = "https://github.com/o/r/releases/download/v1"
ASSET = "https://release-assets.githubusercontent.com/x/a.pt"


def _redirect_run(tmp_path, location, extra=None):
    table = {f"{BASE}/a.pt": (302, {"Location": location}, b"")}
    table.update(extra or {})
    fake = _FakeHTTPS(table)
    opener = da.build_safe_opener(fake).open
    manifest = da.Manifest(BASE, _manifest(BASE, PAYLOADS).artifacts[:1])
    return fake, lambda: da.ensure_artifacts(manifest, tmp_path, opener=opener)


def test_https_redirect_to_release_assets_host_succeeds_and_is_verified(tmp_path):
    fake, run = _redirect_run(tmp_path, ASSET, {ASSET: (200, {}, PAYLOADS["a.pt"])})
    assert run() == [tmp_path / DEST]
    assert (tmp_path / DEST).read_bytes() == PAYLOADS["a.pt"]
    assert fake.seen == [f"{BASE}/a.pt", ASSET]
    _no_leftovers(tmp_path)


def test_redirected_content_is_still_hash_checked(tmp_path):
    _, run = _redirect_run(tmp_path, ASSET, {ASSET: (200, {}, b"x" * len(PAYLOADS["a.pt"]))})
    with pytest.raises(da.ArtifactIntegrityError, match="SHA-256"):
        run()
    assert not (tmp_path / DEST).exists()
    _no_leftovers(tmp_path)


@pytest.mark.parametrize(
    "location",
    [
        "http://release-assets.githubusercontent.com/x/a.pt",  # plain HTTP
        "https://evil.example.com/a.pt",  # unexpected host
        "https://release-assets.githubusercontent.com.evil.example/a.pt",  # look-alike host
        "http://127.0.0.1:9/a.pt",
    ],
)
def test_http_or_unexpected_host_redirects_are_rejected(tmp_path, location):
    fake, run = _redirect_run(tmp_path, location)
    with pytest.raises(da.ArtifactDownloadError, match="refusing redirect"):
        run()
    assert fake.seen == [f"{BASE}/a.pt"]  # the redirect target was never requested
    assert not (tmp_path / DEST).exists()
    _no_leftovers(tmp_path)


def test_unwritable_destination_is_a_clear_error(tmp_path, make_server):
    blocker = tmp_path / "experiments"
    blocker.write_text("a file where a directory is needed")
    with pytest.raises(da.ArtifactDownloadError, match="cannot write"):
        da.ensure_artifacts(_manifest(make_server(PAYLOADS).url, PAYLOADS), tmp_path)


def test_artifact_code_uses_no_credentials_or_secrets():
    text = (ROOT / "src" / "asymmetric_autoencoder" / "demo_artifacts.py").read_text().lower()
    for word in ("authorization", "bearer", "secrets", "api_key", "password"):
        assert word not in text


# ---------------------------------------------------------------- runtime root
def test_runtime_root_is_the_project_when_writable_else_a_temp_mirror(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "demo").mkdir(parents=True)
    run = project / "experiments" / "results" / "demo_cifar10_seed0"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("seed: 0\n")
    assert da.prepare_runtime_root(project) == project

    mirror = da.prepare_runtime_root(project, override=tmp_path / "mirror")
    assert (
        mirror / "experiments/results/demo_cifar10_seed0/config.yaml"
    ).read_text() == "seed: 0\n"

    monkeypatch.setattr(da, "_writable", lambda _d: False)  # read-only checkout
    monkeypatch.setattr(da.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    ro = da.prepare_runtime_root(project)
    assert ro == tmp_path / "tmp" / "aae_demo_runtime" and (ro / "experiments/results").is_dir()


# ---------------------------------------------------------------- the app, end to end
@pytest.fixture
def app_env(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.delenv("AAE_DISABLE_DOWNLOAD", raising=False)
    monkeypatch.delenv("AAE_RESULTS_DIR", raising=False)
    monkeypatch.delenv("AAE_ASSETS_DIR", raising=False)
    monkeypatch.setenv("AAE_RUNTIME_ROOT", str(tmp_path / "runtime"))
    return tmp_path


def _point_app_at(monkeypatch, tmp_path, manifest: da.Manifest):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "base_url": manifest.base_url,
                "files": [
                    {"name": a.name, "dest": a.dest, "size": a.size, "sha256": a.sha256}
                    for a in manifest.artifacts
                ],
            }
        )
    )
    monkeypatch.setenv("AAE_ARTIFACT_MANIFEST", str(path))


def test_app_reports_download_failures_clearly_and_keeps_results_working(
    app_env, monkeypatch, make_server
):
    from streamlit.testing.v1 import AppTest

    server = make_server(PAYLOADS, status=500)
    _point_app_at(monkeypatch, app_env, _manifest(server.url, PAYLOADS))
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.dataframe) > 0  # Results tab unaffected
    warnings = [w.value for w in at.warning]
    assert len(warnings) == 3 and all("could not be provided" in w for w in warnings)
    assert all("HTTP 500" in w and "Results tab is unaffected" in w for w in warnings)
    hits = len(server.hits)
    at.run()  # within the retry window: no new request on the next interaction
    assert len(server.hits) == hits and not at.exception


def test_app_rejects_a_hash_mismatch_and_loads_nothing(app_env, monkeypatch, make_server):
    from streamlit.testing.v1 import AppTest

    server = make_server(PAYLOADS)
    _point_app_at(monkeypatch, app_env, _manifest(server.url, PAYLOADS, sha_override="2" * 64))
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("SHA-256 mismatch" in w.value for w in at.warning)
    assert not list((app_env / "runtime").rglob("best.pt"))


def test_app_downloads_the_real_artifacts_once_and_all_live_tabs_work(
    app_env, monkeypatch, make_server
):
    from streamlit.testing.v1 import AppTest

    real = da.load_manifest(ROOT / "demo" / "artifacts.json")
    if any(not (ROOT / a.dest).is_file() for a in real.artifacts):
        pytest.skip("local artifacts not present to serve")
    server = make_server({a.name: (ROOT / a.dest).read_bytes() for a in real.artifacts})
    _point_app_at(monkeypatch, app_env, da.Manifest(server.url, real.artifacts))
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=300).run()
    assert not at.exception and not at.warning
    assert len(server.hits) == 5
    for a in real.artifacts:
        assert da.sha256_file(app_env / "runtime" / a.dest) == a.sha256
    assert not any("not available" in i.value for i in at.info)
    at.slider(key="lat-k").set_value(8).run()
    at.radio(key="rob-kind").set_value("salt_pepper").run()
    at.selectbox(key="rec-cls").set_value("ship").run()
    assert not at.exception and len(server.hits) == 5  # no re-download on interaction
    fresh = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=300).run()
    assert not fresh.exception and len(server.hits) == 5  # files already present
