"""下载与解压：来源授权、重定向、大小限制、路径越界和链接拒绝。"""

import io
import json
import stat
import tarfile
import warnings
import zipfile

import pytest
import pytest_asyncio
from aiohttp import web
from test_policy import approve, environment

from lynx_agent.approval.cards import batch_text
from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.tools.builtin.download import DownloadFileTool, validate_url
from lynx_agent.tools.builtin.extract import ExtractArchiveTool
from lynx_agent.tools.errors import ToolExecutionError


async def start_app(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest_asyncio.fixture
async def http_server():
    seen = []

    async def file(request):
        seen.append(request.headers)
        return web.Response(body=b"hello download", content_type="text/plain")

    async def redirect(request):
        raise web.HTTPFound("/file")

    async def big(request):
        return web.Response(body=b"x" * 2000)

    async def archive(request):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("inside.txt", "from zip")
        return web.Response(body=buffer.getvalue(), content_type="application/zip")

    app = web.Application()
    app.router.add_get("/file", file)
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/big", big)
    app.router.add_get("/archive.zip", archive)
    runner, base = await start_app(app)
    yield base, seen
    await runner.cleanup()


@pytest.mark.asyncio
async def test_download_requires_source_authorization_and_does_not_execute(tmp_path, http_server):
    base, seen = http_server
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    tool = DownloadFileTool(env.access, env.manager, allow_private=True)
    cards = []
    approve(env.manager, "batch", cards)
    result = json.loads(await tool.execute(url=f"{base}/file", destination=str(destination),
                                           filename="payload.py", purpose="下载脚本"))
    assert result["executed"] is False and result["bytes"] == len(b"hello download")
    assert (destination / "payload.py").read_bytes() == b"hello download"
    assert result["sha256"] and result["host"] == "127.0.0.1"
    card = cards[0]
    assert card.supports_batch
    text = batch_text(card, 10)
    assert "下载来源" in text and "127.0.0.1" in text and "上传到外部：否" in text and "不自动运行" in text
    assert all("Authorization" not in headers and "Cookie" not in headers for headers in seen)


@pytest.mark.asyncio
async def test_download_redirects_are_revalidated(tmp_path, http_server):
    base, _ = http_server
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    tool = DownloadFileTool(env.access, env.manager, allow_private=True)
    approve(env.manager, "batch")
    await tool.execute(url=f"{base}/redirect", destination=str(destination), filename="same.txt")
    assert (destination / "same.txt").read_bytes() == b"hello download"

    other_app = web.Application()

    async def other_file(request):
        return web.Response(body=b"other")

    other_app.router.add_get("/file", other_file)
    other_runner, other_base = await start_app(other_app)
    cross_app = web.Application()

    async def cross(request):
        raise web.HTTPFound(other_base + "/file")

    cross_app.router.add_get("/cross", cross)
    cross_runner, cross_base = await start_app(cross_app)
    try:
        with pytest.raises(ToolExecutionError, match="未授权来源"):
            await tool.execute(url=f"{cross_base}/cross", destination=str(destination), filename="cross.txt")
        assert not (destination / "cross.txt").exists()
    finally:
        await cross_runner.cleanup()
        await other_runner.cleanup()


@pytest.mark.asyncio
async def test_download_size_limit_and_target_conflict(tmp_path, http_server):
    base, _ = http_server
    env = environment(tmp_path, config=ApprovalConfig(mode="auto", download_max_bytes=1024))
    destination = env.root / "out"
    destination.mkdir()
    tool = DownloadFileTool(env.access, env.manager, allow_private=True)
    approve(env.manager, "batch")
    with pytest.raises(ToolExecutionError, match="大小限制"):
        await tool.execute(url=f"{base}/big", destination=str(destination), filename="big.bin")
    assert not (destination / "big.bin").exists()
    (destination / "exists.bin").write_bytes(b"keep")
    with pytest.raises(ToolExecutionError, match="已存在"):
        await tool.execute(url=f"{base}/file", destination=str(destination), filename="exists.bin")


def test_download_url_validation_rejects_protocol_credentials_and_private_host():
    for url in ("file:///C:/x", "ftp://example.com/x", "http://user:pass@example.com/x",
                "http://127.0.0.1/x", "http://localhost/x", "http://[::1]/x"):
        with pytest.raises(ToolExecutionError):
            validate_url(url)
    assert str(validate_url("https://example.com/file")) == "https://example.com/file"


@pytest.mark.asyncio
async def test_extract_zip_rejects_traversal_links_and_duplicates(tmp_path):
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    tool = ExtractArchiveTool(env.access, env.manager)

    def build(name, entries):
        archive = env.root / name
        with zipfile.ZipFile(archive, "w") as bundle:
            for entry, data in entries:
                bundle.writestr(entry, data)
        return archive

    traversal = build("traversal.zip", [("../evil.txt", "bad")])
    with pytest.raises(ToolExecutionError, match="上级路径|越界"):
        tool.build_plan({"archive": str(traversal), "destination": str(destination), "purpose": ""})
    assert not (env.root / "evil.txt").exists()

    linked = env.root / "link.zip"
    with zipfile.ZipFile(linked, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "target")
    with pytest.raises(ToolExecutionError, match="符号链接"):
        tool.build_plan({"archive": str(linked), "destination": str(destination), "purpose": ""})

    duplicate = env.root / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as bundle:
            bundle.writestr("same.txt", "one")
            bundle.writestr("same.txt", "two")
    with pytest.raises(ToolExecutionError, match="重名"):
        tool.build_plan({"archive": str(duplicate), "destination": str(destination), "purpose": ""})

    tar_link = env.root / "link.tar"
    with tarfile.open(tar_link, "w") as bundle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        bundle.addfile(info)
    with pytest.raises(ToolExecutionError, match="链接"):
        tool.build_plan({"archive": str(tar_link), "destination": str(destination), "purpose": ""})


@pytest.mark.asyncio
async def test_extract_writes_only_inside_destination_and_skips_existing(tmp_path):
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    (destination / "b.txt").write_text("keep", encoding="utf-8")
    archive = env.root / "good.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/a.txt", "hello")
        bundle.writestr("b.txt", "replace")
    approve(env.manager, "batch")
    tool = ExtractArchiveTool(env.access, env.manager)
    result = json.loads(await tool.execute(archive=str(archive), destination=str(destination), purpose="解压"))
    assert result["succeeded"] == 1 and result["skipped"] == 1
    assert (destination / "nested" / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (destination / "b.txt").read_text(encoding="utf-8") == "keep"
    assert not (env.root / "nested").exists()


@pytest.mark.asyncio
async def test_extract_total_size_limit_blocks_archive(tmp_path):
    env = environment(tmp_path, config=ApprovalConfig(mode="auto", extract_max_bytes=1024))
    destination = env.root / "out"
    destination.mkdir()
    archive = env.root / "big.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("big.txt", "x" * 2000)
    approve(env.manager, "batch")
    tool = ExtractArchiveTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="总大小|上限"):
        await tool.execute(archive=str(archive), destination=str(destination))


@pytest.mark.asyncio
async def test_extract_archive_change_after_approval_invalidates(tmp_path, monkeypatch):
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    archive = env.root / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.txt", "one")
    approve(env.manager, "batch")
    tool = ExtractArchiveTool(env.access, env.manager)
    original = tool.authorize

    async def authorize(plan):
        grant = await original(plan)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("injected.txt", "two")
        return grant

    monkeypatch.setattr(tool, "authorize", authorize)
    with pytest.raises(ToolExecutionError, match="已被外部修改"):
        await tool.execute(archive=str(archive), destination=str(destination))
    assert not (destination / "injected.txt").exists()


@pytest.mark.asyncio
async def test_download_then_extract_needs_separate_authorization(tmp_path, http_server):
    base, _ = http_server
    env = environment(tmp_path)
    destination = env.root / "out"
    destination.mkdir()
    approve(env.manager, "batch")
    download = DownloadFileTool(env.access, env.manager, allow_private=True)
    payload = json.loads(await download.execute(url=f"{base}/archive.zip", destination=str(destination),
                                                filename="bundle.zip", purpose="下载归档"))
    assert payload["executed"] is False
    extract = ExtractArchiveTool(env.access, env.manager)
    result = json.loads(await extract.execute(archive=str(destination / "bundle.zip"),
                                              destination=str(destination), purpose="解压归档"))
    assert result["succeeded"] == 1
    assert (destination / "inside.txt").read_text(encoding="utf-8") == "from zip"
