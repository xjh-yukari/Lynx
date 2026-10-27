"""离屏渲染 Lynx TUI 参考图（不依赖终端，纯 Pillow 画格子）。

保留借鉴 dsh-TUI 后采用的配色与信息层级，生成历史设计参考图，
不是当前终端运行截图，也不是实际界面入口。

用法（可选依赖 Pillow，不属于 Agent 运行依赖）：
    python tests/manual/tui_mockups.py
输出：docs/tui-reference/*.png；执行会覆盖同名参考图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[2] / "docs" / "tui-reference"

# ---------------------------------------------------------------- 设计令牌
C = {
    "bg": "#1E2127", "panel": "#262B33", "panel_alt": "#2E343D", "line": "#3A424E",
    "text": "#E6E9EF", "dim": "#98A1AF", "faint": "#6A7280",
    "accent": "#7DA1DE", "accent_dim": "#4A6FA5",
    "exec": "#56B6C2", "mutate": "#E5C07B",
    "success": "#98C379", "error": "#E06C75", "warn": "#E5C07B",
    "diff_add": "#2C4030", "diff_del": "#452C30",
    "diff_add_fg": "#9BD07C", "diff_del_fg": "#E58A90",
    "user": "#D77757", "thumb": "#4A6FA5",
}

PAD = 24
FONT_PX = 27
F_ASCII = r"C:\Windows\Fonts\consola.ttf"
F_ASCII_B = r"C:\Windows\Fonts\consolab.ttf"
F_CJK = r"C:\Windows\Fonts\msyh.ttc"
# 这些字符退回 CJK 字体渲染（覆盖全角标点、制表符、方块、●•）。
SYMBOLS = set("●○•◆›»█▏│─└┌┐┘╭╮╰╯╱╲▸")


@dataclass
class Cell:
    ch: str = " "
    fg: str = C["text"]
    bg: str = C["bg"]
    bold: bool = False
    wide: bool = False


@dataclass
class Canvas:
    cols: int
    rows: int
    cells: list[list[Cell]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cells = [[Cell(bg=C["bg"]) for _ in range(self.cols)] for _ in range(self.rows)]
        self.font = ImageFont.truetype(F_ASCII, FONT_PX)
        self.font_b = ImageFont.truetype(F_ASCII_B, FONT_PX)
        self.font_cjk = ImageFont.truetype(F_CJK, int(FONT_PX * 0.97))
        self.cw = self.font.getlength("M")
        self.ch = round(FONT_PX * 1.34)

    # ---- 绘制原语 ----
    def put(self, x: int, y: int, text: str, fg: str | None = None,
            bg: str | None = None, bold: bool = False) -> int:
        """写入文本，返回结束列号（宽字符占 2 列）。"""
        if not 0 <= y < self.rows:
            return x
        column = x
        for ch in text:
            wide = ord(ch) > 0x2E7F
            if column >= self.cols:
                break
            cell = self.cells[y][column]
            cell.ch, cell.wide = ch, wide
            cell.fg = fg or C["text"]
            cell.bold = bold
            if bg is not None:
                cell.bg = bg
            if wide:
                if column + 1 < self.cols:
                    nxt = self.cells[y][column + 1]
                    nxt.ch, nxt.bg = "", cell.bg
                column += 2
            else:
                column += 1
        return column

    def seq(self, x: int, y: int, parts, bg: str | None = None) -> int:
        """按顺序写多段 (文本, 颜色[, 粗体])，自动处理宽字符推进。"""
        for part in parts:
            text, color = part[0], part[1]
            bold = part[2] if len(part) > 2 else False
            x = self.put(x, y, text, color, bg, bold)
        return x

    def fill(self, x: int, y: int, w: int, h: int, bg: str) -> None:
        for row in range(y, min(y + h, self.rows)):
            for col in range(x, min(x + w, self.cols)):
                self.cells[row][col].bg = bg

    def rect(self, x: int, y: int, w: int, h: int, fg: str, bg: str,
             title: str = "", title_fg: str | None = None) -> None:
        self.fill(x, y, w, h, bg)
        self.put(x, y, "╭" + "─" * (w - 2) + "╮", fg, bg)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, "│", fg, bg)
            self.put(x + w - 1, row, "│", fg, bg)
        self.put(x, y + h - 1, "╰" + "─" * (w - 2) + "╯", fg, bg)
        if title:
            self.put(x + 2, y, f" {title} ", title_fg or fg, bg)

    def hline(self, y: int, fg: str) -> None:
        self.put(0, y, "─" * self.cols, fg)

    # ---- 光栅化 ----
    def render(self) -> Image.Image:
        width = round(self.cols * self.cw) + PAD * 2
        height = self.rows * self.ch + PAD * 2
        image = Image.new("RGB", (width, height), C["bg"])
        draw = ImageDraw.Draw(image)
        # 第一遍画底色：宽字符的第二格会把已画的字形右半截盖掉，必须分两遍。
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.cells[row][col]
                if cell.bg != C["bg"]:
                    x = PAD + col * self.cw
                    y = PAD + row * self.ch
                    draw.rectangle([x, y, x + self.cw, y + self.ch], fill=cell.bg)
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.cells[row][col]
                x = PAD + col * self.cw
                y = PAD + row * self.ch
                if not cell.ch.strip():
                    continue
                if cell.wide or cell.ch in SYMBOLS:
                    draw.text((x + self.cw, y + self.ch / 2), cell.ch,
                              font=self.font_cjk, fill=cell.fg, anchor="mm")
                else:
                    font = self.font_b if cell.bold else self.font
                    draw.text((x, y + self.ch / 2), cell.ch, font=font,
                              fill=cell.fg, anchor="lm")
        return image


# ---------------------------------------------------------------- 图 1：主界面
def mock_chat() -> Canvas:
    c = Canvas(116, 36)
    c.seq(1, 0, [("◆ lynx", C["accent"], True), ("  v0.1.0", C["faint"]),
                 ("   E:\\github\\nanobot", C["dim"]),
                 ("     · 会话 ", C["faint"]), ("2f9c1a7e", C["dim"])])
    c.put(c.cols - 22, 0, "deepseek-flash · max", C["accent_dim"])
    c.hline(1, C["line"])

    x = c.seq(1, 3, [("›  ", C["user"], True)])
    c.put(x, 3, "帮我看看 document_rag 的 RRF 混合检索，有没有问题", C["text"])

    x = c.put(2, 5, "●", C["accent"])
    c.put(x + 1, 5, "看完了。融合本身是对的，但有三处值得改：", C["text"])
    c.seq(4, 6, [("① ", C["faint"]), ("candidates=20 太浅", C["dim"]),
                 ("   ② ", C["faint"]), ("融合后没有相关性门槛", C["dim"])])
    c.seq(4, 7, [("③ ", C["faint"]), ("近重复判据几乎不触发", C["dim"])])

    c.rect(6, 9, 96, 3, C["line"], C["panel"], title="影响排序", title_fg=C["faint"])
    c.put(8, 10, "7 个 chunk 的库上 candidates 已把全库取满，RRF 只剩重排作用", C["dim"], C["panel"])

    x = c.seq(1, 13, [("›  ", C["user"], True)])
    c.put(x, 13, "那就先修第 ②③ 条，别动参数", C["text"])

    x = c.put(2, 15, "○", C["accent"])
    c.seq(x + 1, 15, [("思考", C["dim"]), ("  (ctrl+o 展开)", C["faint"])])
    for index, (name, color, path, extra, seconds) in enumerate((
        ("Read", C["exec"], "src/document_rag/storage.py", "140 lines · 84-125", "0.2s"),
        ("Edit", C["mutate"], "storage.py", "去掉 3-gram 判据，加相对门槛 ratio=0.5", "1.1s"),
        ("Bash", C["exec"], "pytest services/document_rag -q", "4 passed", "3.4s"),
    )):
        y = 16 + index * 2
        c.put(2, y, "•", color, bold=True)
        c.seq(4, y, [(f"{name:<5}", color, True), (path, C["text"])])
        c.put(c.cols - 30, y, seconds, C["faint"])
        c.put(4, y + 1, "└ ", C["faint"])
        c.put(6, y + 1, extra, C["faint"])

    x = c.put(2, 23, "●", C["accent"])
    c.put(x + 1, 23, "改好了：删掉近重复判据、加了相对门槛，测试全绿。", C["text"])
    c.put(4, 24, "需要我把 candidates 也一起调吗？", C["dim"])

    for row in range(3, 26):
        c.put(c.cols - 1, row, "│", C["line"])
    for row in range(9, 19):
        c.put(c.cols - 1, row, "█", C["thumb"])

    c.rect(1, 26, c.cols - 2, 3, C["accent_dim"], C["panel"])
    x = c.seq(3, 27, [("› ", C["user"], True)], bg=C["panel"])
    x = c.put(x, 27, "把 diff 贴给我看", C["text"], C["panel"])
    c.put(x, 27, "▏", C["accent"], C["panel"])
    c.put(c.cols - 20, 27, "shift+tab 切换权限", C["faint"], C["panel"])

    c.seq(2, 30, [("esc 取消", C["faint"]), ("   · ctrl+c 中断", C["faint"]),
                  ("   · /help 查看命令", C["faint"])])

    c.hline(32, C["line"])
    c.seq(1, 33, [("deepseek-flash", C["accent_dim"]), (" · max", C["faint"]),
                  ("  · 缓存 ", C["dim"]), ("96.7%", C["success"]),
                  ("  · ", C["faint"]), ("¥0.06", C["dim"]),
                  ("  · 记忆 3 条", C["dim"])])
    x = c.cols - 36
    for width, color in ((6, C["accent"]), (3, C["exec"]), (2, C["mutate"]), (1, C["warn"])):
        c.put(x, 33, "█" * width, color)
        x += width
    c.seq(x + 1, 33, [("ctx 4.5%", C["dim"]), ("  (45k/1.0M)", C["faint"])])
    return c


# ---------------------------------------------------------------- 图 2：工具与审批
def mock_tool() -> Canvas:
    c = Canvas(100, 25)
    c.seq(1, 0, [("◆ lynx", C["accent"], True), ("   工具卡片 · diff · 审批", C["faint"])])
    c.hline(1, C["line"])

    c.rect(1, 2, 96, 9, C["line"], C["panel"], title="Edit  runner.py", title_fg=C["mutate"])
    c.put(3, 3, "src/lynx_agent/agent/runner.py", C["dim"], C["panel"])
    c.seq(74, 3, [("+2", C["diff_add_fg"]), ("   ", C["dim"]), ("−1", C["diff_del_fg"])],
          bg=C["panel"])
    c.put(84, 3, "0.8s", C["faint"], C["panel"])

    rows = [
        (" 42  ", " def _restored_history(session):", None),
        (" 43  ", "- def _restored_history(session):", "del"),
        (" 43  ", "+ def _restored_history(session, *, strict=False):", "add"),
        (" 44  ", "     return tuple(", None),
        (" 45  ", "-     message.model_copy(update={", "del"),
        (" 45  ", "+     message.model_copy(update={", "add"),
    ]
    for index, (number, line, kind) in enumerate(rows):
        y = 4 + index
        bg = C["diff_add"] if kind == "add" else (C["diff_del"] if kind == "del" else C["panel"])
        fg = (C["diff_add_fg"] if kind == "add" else
              (C["diff_del_fg"] if kind == "del" else C["text"]))
        c.put(2, y, number, C["faint"], bg)
        c.put(8, y, line, fg, bg)
        c.put(8 + len(line), y, " " * 12, fg, bg)

    c.rect(1, 12, 96, 5, C["warn"], C["panel_alt"], title="需要批准", title_fg=C["warn"])
    c.put(3, 13, "修改 1 个文件，执行 0 条命令", C["text"], C["panel_alt"])
    c.put(3, 14, "Enter 允许一次 · a 总是允许 · d 查看完整 diff · n 拒绝 · esc 取消",
          C["dim"], C["panel_alt"])
    c.put(3, 15, "影响：第 3 次修改 runner.py；本次会话累计 +6 −9", C["faint"], C["panel_alt"])

    c.rect(1, 18, 96, 3, C["accent_dim"], C["panel"])
    c.seq(3, 19, [("› ", C["user"], True), ("继续", C["text"])], bg=C["panel"])
    return c


# ---------------------------------------------------------------- 图 3：窄屏
def mock_narrow() -> Canvas:
    c = Canvas(80, 21)
    c.seq(1, 0, [("◆ lynx", C["accent"], True), ("   80 列降级", C["faint"])])
    c.put(c.cols - 16, 0, "deepseek-flash", C["accent_dim"])
    c.hline(1, C["line"])

    x = c.seq(1, 2, [("› ", C["user"], True)])
    c.put(x, 2, "帮我看看这个报错", C["text"])
    x = c.put(2, 4, "●", C["accent"])
    c.put(x + 1, 4, "是系统代理劫持了 127.0.0.1：Clash", C["text"])
    c.put(4, 5, "把本地请求转发出去，回了 502。加", C["text"])
    c.put(4, 6, "NO_PROXY 就好。", C["text"])
    c.put(2, 8, "•", C["exec"], bold=True)
    c.seq(4, 8, [("Edit ", C["exec"], True), ("proxy/server.py", C["text"])])
    c.put(4, 9, "└ +9 −0", C["faint"])

    c.rect(1, 11, 78, 3, C["accent_dim"], C["panel"])
    c.seq(3, 12, [("› ", C["user"], True), ("好", C["text"])], bg=C["panel"])
    c.put(c.cols - 14, 12, "Enter 发送", C["faint"], C["panel"])

    c.hline(14, C["line"])
    c.put(1, 15, "deepseek-flash · max", C["accent_dim"])
    c.put(1, 16, "缓存 96.7% · ¥0.06 · 记忆 3 条", C["dim"])
    c.put(c.cols - 22, 15, "ctx 4.5%", C["dim"])
    c.put(c.cols - 22, 16, "45k / 1.0M", C["faint"])

    c.hline(18, C["line"])
    c.put(1, 19, "窄屏规则：隐藏 banner · 状态栏拆两行 · 工具输出默认折叠 · 审批收成单行", C["faint"])
    return c


# ---------------------------------------------------------------- 图 4：令牌板
def mock_tokens() -> Canvas:
    c = Canvas(104, 27)
    c.put(1, 0, "Lynx TUI 语义令牌（对照 dsh-TUI v0.16.1 dark 调色板）", C["accent"], bold=True)
    c.hline(1, C["line"])
    c.put(2, 2, "组", C["faint"], bold=True)
    c.put(10, 2, "令牌", C["faint"], bold=True)
    c.put(32, 2, "色值", C["faint"], bold=True)
    c.put(46, 2, "用途", C["faint"], bold=True)
    rows = [
        ("品牌", "accent", C["accent"], "焦点、链接、助手标记"),
        ("品牌", "accentDim", C["accent_dim"], "状态栏模型名、输入框描边"),
        ("文本", "text", C["text"], "正文"),
        ("文本", "dim", C["dim"], "次要说明、元信息"),
        ("文本", "faint", C["faint"], "耗时、折叠提示、占位符"),
        ("工具", "exec", C["exec"], "Bash / 检索类工具名"),
        ("工具", "mutate", C["mutate"], "Edit / Write 类工具名"),
        ("状态", "success", C["success"], "通过、缓存命中"),
        ("状态", "error", C["error"], "失败"),
        ("状态", "warn", C["warn"], "审批、权限风险"),
        ("diff", "diffAddBg", C["diff_add"], "新增行底色"),
        ("diff", "diffDelBg", C["diff_del"], "删除行底色"),
        ("diff", "diffAddFg", C["diff_add_fg"], "新增行正文 / 字数"),
        ("diff", "diffDelFg", C["diff_del_fg"], "删除行正文 / 字数"),
        ("用户", "user", C["user"], "用户标记、输入符"),
        ("面板", "panel", C["panel"], "卡片、输入框底色"),
        ("面板", "line", C["line"], "边框、分隔线"),
    ]
    for index, (group, token, color, usage) in enumerate(rows):
        y = 3 + index
        c.put(2, y, group, C["dim"])
        c.put(10, y, token, C["text"], bold=True)
        c.put(32, y, "███", color)
        c.put(36, y, color.upper(), C["dim"])
        c.put(46, y, usage, C["dim"])
    c.hline(21, C["line"])
    c.put(2, 22, "映射到 Lynx 现有 themes.py 的 BASE：", C["accent"])
    c.put(2, 23, "message.user→user · message.marker→accent · message.tool→exec · md.syntax.*→语法组",
          C["dim"])
    c.put(2, 24, "需新增：toolDot{Exec,Read,Write,Web} · diff{Add,Del}{Bg,Fg} · contextBar* · shimmer",
          C["dim"])
    c.put(2, 26, "原则：颜色只表达角色，不表达组件；同一角色在卡片、状态栏、diff 里共用一个令牌。",
          C["faint"])
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, factory in {
        "01-chat-overview.png": mock_chat,
        "02-tool-diff-approval.png": mock_tool,
        "03-narrow-80col.png": mock_narrow,
        "04-theme-tokens.png": mock_tokens,
    }.items():
        image = factory().render()
        image.save(OUT / name)
        print(f"{OUT / name}  {image.width}x{image.height}")


if __name__ == "__main__":
    main()
