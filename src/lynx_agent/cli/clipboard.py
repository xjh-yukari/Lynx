"""显式用户操作才写剪贴板；不读取剪贴板、不使用 OSC 52。"""

import asyncio
import ctypes
import os
import shutil
import subprocess
import sys


class ClipboardError(Exception):
    pass


def _windows_copy(text):
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user.OpenClipboard.argtypes = [wintypes.HWND]
    user.OpenClipboard.restype = wintypes.BOOL
    user.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
    user.CreateWindowExW.restype = wintypes.HWND
    user.DestroyWindow.argtypes = [wintypes.HWND]
    user.DestroyWindow.restype = wintypes.BOOL
    user.EmptyClipboard.restype = wintypes.BOOL
    user.EmptyClipboard.argtypes = []
    user.CloseClipboard.argtypes = []
    user.CloseClipboard.restype = wintypes.BOOL
    user.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
    user.SetClipboardData.restype = ctypes.c_void_p
    kernel.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel.GlobalAlloc.restype = ctypes.c_void_p
    kernel.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel.GlobalLock.restype = ctypes.c_void_p
    kernel.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel.GlobalUnlock.restype = wintypes.BOOL
    kernel.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel.GlobalFree.restype = ctypes.c_void_p
    # OpenClipboard(NULL) 后 EmptyClipboard 会清空所有者，导致 SetClipboardData 失败。
    # 使用系统预定义类创建自有隐藏窗口，不 ShowWindow，不改变终端焦点。
    owner = user.CreateWindowExW(0, "STATIC", "Lynx clipboard", 0, 0, 0, 0, 0, None, None, None, None)
    if not owner:
        raise ClipboardError("无法建立本地剪贴板所有者")
    opened = False
    handle = None
    try:
        if not user.OpenClipboard(owner):
            raise ClipboardError("剪贴板正被占用，请稍后重试")
        opened = True
        data = text.encode("utf-16-le") + b"\0\0"
        handle = kernel.GlobalAlloc(2, len(data))
        pointer = kernel.GlobalLock(handle) if handle else None
        if not pointer:
            raise ClipboardError("无法准备剪贴板数据")
        ctypes.memmove(pointer, data, len(data))
        kernel.GlobalUnlock(handle)
        if not user.EmptyClipboard() or not user.SetClipboardData(13, handle):
            raise ClipboardError("系统拒绝写入剪贴板")
        handle = None  # 所有权已转交 Windows。
    finally:
        if handle:
            kernel.GlobalFree(handle)
        if opened:
            user.CloseClipboard()
        user.DestroyWindow(owner)


def write_clipboard(text):
    if not isinstance(text, str) or not text or "\0" in text:
        raise ClipboardError("复制内容为空或含不支持的空字符")
    if len(text.encode("utf-8")) > 8 * 1024 * 1024:
        raise ClipboardError("复制内容超过8 MiB，请缩小范围")
    if os.name == "nt":
        _windows_copy(text)
        return
    candidates = [["pbcopy"]] if sys.platform == "darwin" else [
        ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"],
    ]
    command = next((args for args in candidates if shutil.which(args[0])), None)
    if command is None:
        raise ClipboardError("未找到本地剪贴板工具；可使用终端原生选择复制")
    try:
        subprocess.run(command, input=text, encoding="utf-8", capture_output=True, check=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        raise ClipboardError("写入系统剪贴板失败，请检查本地桌面会话") from None


async def copy_text(text):
    await asyncio.to_thread(write_clipboard, text)
