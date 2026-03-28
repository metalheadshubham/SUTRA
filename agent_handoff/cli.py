"""
cli.py — Interactive terminal interface for S.U.T.R.A v0.2 (Council Mode).

A Claude Code-style REPL that auto-detects Ollama models and runs the
4-stage council pipeline with live streaming, plugins, and logging.

Run with::

    python -m agent_handoff.cli
    # or
    sutra
"""

import sys
import os
import re
import time
import json
import shutil
import threading
import itertools
import logging
import traceback
import importlib.util
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import ollama

from agent_handoff.cache import DemonstrationCache
from agent_handoff.templates import ANSWER_PROMPT, CRITIQUE_PROMPT, SYNTHESIS_PROMPT
from agent_handoff.utils import hash_query

logging.basicConfig(level=logging.WARNING)

# ── Directories ──────────────────────────────────────────────────────────────

SUTRA_DIR = Path.home() / ".sutra"
PLUGINS_DIR = SUTRA_DIR / "plugins"
LOGS_DIR = SUTRA_DIR / "logs"


def _ensure_dirs():
    """Create ~/.sutra/ directory tree on first run."""
    for d in (SUTRA_DIR, PLUGINS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── ANSI color codes ────────────────────────────────────────────────────────

class C:
    """ANSI escape sequences for terminal styling."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    ITALIC   = "\033[3m"
    UNDER    = "\033[4m"

    # Foreground
    BLACK    = "\033[30m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"

    # Bright foreground
    BBLACK   = "\033[90m"
    BRED     = "\033[91m"
    BGREEN   = "\033[92m"
    BYELLOW  = "\033[93m"
    BBLUE    = "\033[94m"
    BMAGENTA = "\033[95m"
    BCYAN    = "\033[96m"
    BWHITE   = "\033[97m"

    # Background
    BG_BLACK   = "\033[40m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"

    # Cursor control
    CLEAR_LINE = "\033[2K"
    UP         = "\033[1A"
    HIDE_CUR   = "\033[?25l"
    SHOW_CUR   = "\033[?25h"


def _enable_ansi_windows():
    """Enable ANSI escape code processing on Windows."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def _lock_console_size(cols: int = 120, rows: int = 30):
    """Lock the Windows console to a fixed size — disable maximize and resize.

    Only runs on Windows; silently skipped on other platforms.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        import ctypes.wintypes as wt

        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32

        # ── Remove maximize button and thick frame (resize grip) ────────
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            GWL_STYLE = -16
            WS_MAXIMIZEBOX = 0x00010000
            WS_THICKFRAME = 0x00040000
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            style &= ~WS_MAXIMIZEBOX  # disable maximize
            style &= ~WS_THICKFRAME   # disable drag-to-resize
            user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            # Force window to repaint with new style
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_FRAMECHANGED = 0x0020
            user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_FRAMECHANGED,
            )

        # ── Set console buffer and window size ──────────────────────────
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short), ("Top", ctypes.c_short),
                ("Right", ctypes.c_short), ("Bottom", ctypes.c_short),
            ]

        # Shrink window first (can't set buffer smaller than window)
        tiny_rect = SMALL_RECT(0, 0, 1, 1)
        kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(tiny_rect))

        # Set buffer size
        buf_size = COORD(cols, rows)
        kernel32.SetConsoleScreenBufferSize(handle, buf_size)

        # Set window size
        win_rect = SMALL_RECT(0, 0, cols - 1, rows - 1)
        kernel32.SetConsoleWindowInfo(handle, True, ctypes.byref(win_rect))

    except Exception:
        pass


# ── Spinner ──────────────────────────────────────────────────────────────────

class Spinner:
    """Animated terminal spinner with status message."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "", color: str = C.CYAN):
        self.message = message
        self.color = color
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Spinner":
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        frames = itertools.cycle(self.FRAMES)
        sys.stdout.write(C.HIDE_CUR)
        while self._running:
            frame = next(frames)
            sys.stdout.write(f"\r  {self.color}{frame}{C.RESET} {C.DIM}{self.message}{C.RESET}")
            sys.stdout.flush()
            time.sleep(0.08)

    def update(self, message: str):
        self.message = message

    def stop(self, final_message: str = "", symbol: str = "✓", color: str = C.GREEN):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        sys.stdout.write(f"\r{C.CLEAR_LINE}")
        if final_message:
            sys.stdout.write(f"  {color}{symbol}{C.RESET} {final_message}\n")
        sys.stdout.write(C.SHOW_CUR)
        sys.stdout.flush()

    def fail(self, message: str = ""):
        self.stop(final_message=message or self.message, symbol="✗", color=C.RED)


# ── Terminal drawing helpers ─────────────────────────────────────────────────

def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _box(title: str, content: str, color: str = C.CYAN, max_width: int = 0) -> str:
    """Draw a bordered box around content."""
    w = max_width or min(_term_width() - 4, 90)
    inner_w = w - 4

    lines: List[str] = []
    lines.append(f"  {color}╭{'─' * (w - 2)}╮{C.RESET}")
    title_padded = f" {title} "
    pad = inner_w - len(title)
    lines.append(f"  {color}│{C.RESET}{C.BOLD} {title_padded}{' ' * max(0, pad)}{color}│{C.RESET}")
    lines.append(f"  {color}├{'─' * (w - 2)}┤{C.RESET}")
    for line in content.splitlines():
        while len(line) > inner_w:
            lines.append(f"  {color}│{C.RESET} {line[:inner_w]} {color}│{C.RESET}")
            line = line[inner_w:]
        pad_count = inner_w - len(line)
        lines.append(f"  {color}│{C.RESET} {line}{' ' * max(0, pad_count)} {color}│{C.RESET}")
    lines.append(f"  {color}╰{'─' * (w - 2)}╯{C.RESET}")
    return "\n".join(lines)


def _divider(label: str = ""):
    w = min(_term_width() - 4, 90)
    if label:
        pad = w - len(label) - 4
        print(f"\n  {C.BBLACK}──{C.RESET} {C.DIM}{label}{C.RESET} {C.BBLACK}{'─' * max(0, pad)}{C.RESET}")
    else:
        print(f"  {C.BBLACK}{'─' * w}{C.RESET}")


# ── Gradient banner ──────────────────────────────────────────────────────────

def _rgb(r: int, g: int, b: int) -> str:
    """Generate ANSI 24-bit foreground color code."""
    return f"\033[38;2;{r};{g};{b}m"


def _header():
    """Print the SUTRA banner with magenta/cyan foreground colors only."""
    # Hand-coded figlet-style ASCII block letters
    banner_lines = [
        "███████╗ ██╗   ██╗ ████████╗ ██████╗   █████╗ ",
        "██╔════╝ ██║   ██║ ╚══██╔══╝ ██╔══██╗ ██╔══██╗",
        "███████╗ ██║   ██║    ██║    ██████╔╝ ███████║",
        "╚════██║ ██║   ██║    ██║    ██╔══██╗ ██╔══██║",
        "███████║ ╚██████╔╝    ██║    ██║  ██║ ██║  ██║",
        "╚══════╝  ╚═════╝     ╚═╝    ╚═╝  ╚═╝ ╚═╝  ╚═╝",
    ]

    # Foreground-only colors: top half magenta, bottom half cyan
    colors = [C.BMAGENTA] * 3 + [C.BCYAN] * 3

    print()
    for line, color in zip(banner_lines, colors):
        print(f"    {C.BOLD}{color}{line}{C.RESET}")

    print(f"    {C.DIM}Structured Universal Transfer via Retrieval Adaptation{C.RESET}")
    print(f"    {C.DIM}by Shubham Kumar{C.RESET}")
    print(f"    {C.DIM}v0.2.0 · council mode · 8GB-ready{C.RESET}")
    print()


# ── Model detection ──────────────────────────────────────────────────────────

def detect_models(host: Optional[str] = None) -> List[Dict[str, Any]]:
    """Auto-detect all models available in the local Ollama instance."""
    try:
        client = ollama.Client(host=host) if host else ollama.Client()
        response = client.list()
        models = response.get("models", [])
        return models
    except Exception:
        return []


def format_model_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _parse_param_count(name: str) -> Optional[float]:
    """Extract parameter count in billions from model name.

    Examples: 'qwen2.5-coder:3b' → 3.0, 'phi3:mini' → None,
    'qwen2.5:0.5b' → 0.5, 'llama3.2:7b-instruct' → 7.0
    """
    match = re.search(r":(\d+(?:\.\d+)?)b", name.lower())
    if match:
        return float(match.group(1))
    return None


def _classify_role(name: str) -> str:
    """Classify a model as 'small' or 'large' based on parameter count.

    ≤4B = small, ≥6B = large, unknown = 'unknown'.
    """
    params = _parse_param_count(name)
    if params is None:
        return "unknown"
    if params <= 4.0:
        return "small"
    return "large"


def _print_model_recommendations(models: List[Dict[str, Any]]):
    """Print model recommendation block with role detection."""
    print(f"\n  {C.BOLD}Model recommendations:{C.RESET}")
    print(f"  {C.BBLACK}─────────────────────{C.RESET}")

    for m in models:
        name = m.get("name", m.get("model", "unknown"))
        size = m.get("size", 0)
        role = _classify_role(name)
        size_str = format_model_size(size)

        if role == "small":
            role_desc = f"small role  (fast diversity generator, ~{size_str} RAM)"
            star = f" {C.BYELLOW}★{C.RESET}"
        elif role == "large":
            role_desc = f"large role  (strong synthesizer, ~{size_str} RAM)"
            star = f" {C.BYELLOW}★{C.RESET}"
        else:
            role_desc = f"~{size_str} RAM"
            star = ""

        print(f"  {C.BOLD}{name:<22}{C.RESET} → {C.DIM}{role_desc}{C.RESET}{star}")

    print()


def pick_model_council(
    models: List[Dict[str, Any]],
    role: str,
    exclude: Optional[str] = None,
) -> str:
    """Interactive model picker for council roles (small/large)."""
    filtered = models
    if exclude:
        alt = [m for m in models if m.get("name", m.get("model", "")) != exclude]
        if alt:
            filtered = alt

    # Sort: preferred role first
    def sort_key(m):
        name = m.get("name", m.get("model", ""))
        cls = _classify_role(name)
        if cls == role:
            return 0
        elif cls == "unknown":
            return 1
        return 2

    filtered.sort(key=sort_key)
    names = [m.get("name", m.get("model", "unknown")) for m in filtered]

    if not names:
        print(f"  {C.RED}✗{C.RESET} No models found!")
        sys.exit(1)

    if len(names) == 1:
        print(f"  {C.GREEN}✓{C.RESET} Auto-selected {role} model: {C.BOLD}{names[0]}{C.RESET}")
        return names[0]

    label = "small model (stages 1a, 1b, 2)" if role == "small" else "large model (stage 3 — synthesis)"
    print(f"\n  {C.BOLD}{C.CYAN}Select {label}:{C.RESET}\n")

    for i, model in enumerate(filtered):
        name = model.get("name", model.get("model", "unknown"))
        size = format_model_size(model.get("size", 0))
        cls = _classify_role(name)
        star = f" {C.BYELLOW}★{C.RESET}" if cls == role else ""
        marker = f"{C.BBLACK}│{C.RESET}"
        print(f"    {C.CYAN}{i + 1}{C.RESET} {marker} {C.BOLD}{name}{C.RESET}  {C.DIM}({size}){C.RESET}{star}")

    print()
    while True:
        try:
            raw = input(f"  {C.BMAGENTA}▸{C.RESET} Enter number (1-{len(names)}): ").strip()
            if not raw:
                choice = 0
            else:
                choice = int(raw) - 1
            if 0 <= choice < len(names):
                print(f"  {C.GREEN}✓{C.RESET} {role} model: {C.BOLD}{names[choice]}{C.RESET}")
                return names[choice]
        except (ValueError, EOFError):
            pass
        print(f"  {C.RED}  Invalid choice, try again.{C.RESET}")


# ── Plugin system ────────────────────────────────────────────────────────────

PLUGIN_HOOKS = ["pre_query", "post_answer_a", "post_answer_b", "post_critique", "post_synthesis"]


class _FunctionPluginWrapper:
    """Wrap top-level functions as a plugin-like object."""

    def __init__(self, hooks_map: Dict[str, Any]):
        for name, fn in hooks_map.items():
            setattr(self, name, fn)


class PluginManager:
    """Load and manage plugins from ~/.sutra/plugins/."""

    def __init__(self):
        self.plugins: List[Dict[str, Any]] = []  # {name, instance, path, hooks}
        self.failed: List[Dict[str, str]] = []   # {name, path, error}

    def load_all(self):
        """Load all .py files from the plugins directory."""
        self.plugins.clear()
        self.failed.clear()
        if not PLUGINS_DIR.exists():
            return

        for py_file in sorted(PLUGINS_DIR.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Strategy 1: Find plugin class — first class with any hook
                plugin_cls = None
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    if isinstance(obj, type) and any(hasattr(obj, h) for h in PLUGIN_HOOKS):
                        plugin_cls = obj
                        break

                if plugin_cls is not None:
                    instance = plugin_cls()
                    hooks = [h for h in PLUGIN_HOOKS if hasattr(instance, h)]
                    self.plugins.append({
                        "name": py_file.stem,
                        "instance": instance,
                        "path": str(py_file),
                        "hooks": hooks,
                    })
                    continue

                # Strategy 2: Accept top-level functions with hook names
                hooks_map = {}
                for h in PLUGIN_HOOKS:
                    fn = getattr(module, h, None)
                    if callable(fn):
                        hooks_map[h] = fn

                if hooks_map:
                    instance = _FunctionPluginWrapper(hooks_map)
                    self.plugins.append({
                        "name": py_file.stem,
                        "instance": instance,
                        "path": str(py_file),
                        "hooks": list(hooks_map.keys()),
                    })

            except Exception as exc:
                self.failed.append({
                    "name": py_file.stem,
                    "path": str(py_file),
                    "error": f"{type(exc).__name__}: {exc}",
                })

    def call_hook(self, hook_name: str, *args):
        """Call a hook on all plugins that implement it. Fail silently."""
        for plugin in self.plugins:
            if hook_name in plugin["hooks"]:
                try:
                    getattr(plugin["instance"], hook_name)(*args)
                except Exception:
                    pass

    def list_plugins(self):
        """Print loaded plugins, their hooks, and any failures."""
        if not self.plugins and not self.failed:
            print(f"  {C.DIM}No plugins loaded.{C.RESET}")
            print(f"  {C.DIM}Place .py files in {PLUGINS_DIR}{C.RESET}\n")
            return
        if self.plugins:
            print(f"\n  {C.BOLD}Loaded plugins:{C.RESET}\n")
            for p in self.plugins:
                hooks_str = ", ".join(p["hooks"]) if p["hooks"] else "no hooks"
                print(f"    {C.CYAN}{p['name']}{C.RESET}  {C.DIM}({hooks_str}){C.RESET}")
                print(f"      {C.BBLACK}{p['path']}{C.RESET}")
        if self.failed:
            print(f"\n  {C.BOLD}{C.RED}Failed plugins:{C.RESET}\n")
            for f in self.failed:
                print(f"    {C.RED}{f['name']}{C.RESET}  {C.DIM}{f['path']}{C.RESET}")
                print(f"      {C.RED}✗{C.RESET} {f['error']}")
        print()


# ── Built-in logger plugin ───────────────────────────────────────────────────

class _BuiltinLogger:
    """Always-loaded plugin that logs every run to ~/.sutra/logs/YYYY-MM-DD.jsonl."""

    def log_run(
        self,
        query: str,
        model_small: str,
        model_large: str,
        answer_a: str,
        answer_b: str,
        critique: str,
        synthesis: str,
        latencies: Dict[str, float],
    ):
        """Append a JSON line to today's log file."""
        try:
            log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "model_small": model_small,
                "model_large": model_large,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "critique": critique,
                "synthesis": synthesis,
                "latencies": latencies,
            }
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass


_builtin_logger = _BuiltinLogger()


# ── Stage header helper ──────────────────────────────────────────────────────

def _stage_header(stage_label: str, info: str):
    """Print a stage header with right-aligned model/temp info."""
    w = min(_term_width() - 4, 90)
    left = f"── {stage_label} "
    right = f" {info} ──"
    mid_len = w - len(left) - len(right)
    if mid_len < 2:
        mid_len = 2
    print(f"\n  {C.BBLACK}{left}{'─' * mid_len}{right}{C.RESET}")


# ── Council interactive pipeline (display layer) ─────────────────────────────

def run_council_interactive(
    client: ollama.Client,
    model_small: str,
    model_large: str,
    query: str,
    cache: DemonstrationCache,
    plugin_mgr: PluginManager,
    loaded_files: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Run the 4-stage council pipeline with live streaming display.

    Calls client.generate(stream=True) directly for each stage.
    Does NOT use CouncilHandoff.run_detailed() — this IS the display layer.

    Returns the synthesis text, or None on failure.
    """
    # Inject loaded file context
    if loaded_files:
        context_block = "\n\n".join(
            f"# File: {fname}\n```\n{content}\n```"
            for fname, content in loaded_files.items()
        )
        query = f"Context files:\n{context_block}\n\nQuery: {query}"

    # Plugin: pre_query
    plugin_mgr.call_hook("pre_query", query)

    # RAM estimate
    peak_ram_gb = 0.0

    total_t0 = time.perf_counter()
    latencies: Dict[str, float] = {}
    token_counts: Dict[str, int] = {}

    # ── Stage 1a: Answer A ──────────────────────────────────────────────
    _stage_header("Stage 1a", f"Answer A · {model_small} · temp=0.3")
    answer_a, lat, toks = _stream_stage(
        client, model_small, ANSWER_PROMPT.format(query=query),
        temperature=0.3, color=C.CYAN,
    )
    if answer_a is None:
        return None
    latencies["stage_1a"] = lat
    token_counts["stage_1a"] = toks
    print(f"  {C.GREEN}✓{C.RESET} Answer A  {lat:.1f}s · {toks} tokens · {C.DIM}unloaded{C.RESET}")
    plugin_mgr.call_hook("post_answer_a", answer_a)

    # ── Stage 1b: Answer B ──────────────────────────────────────────────
    _stage_header("Stage 1b", f"Answer B · {model_small} · temp=0.8")
    answer_b, lat, toks = _stream_stage(
        client, model_small, ANSWER_PROMPT.format(query=query),
        temperature=0.8, color=C.YELLOW,
    )
    if answer_b is None:
        return None
    latencies["stage_1b"] = lat
    token_counts["stage_1b"] = toks
    print(f"  {C.GREEN}✓{C.RESET} Answer B  {lat:.1f}s · {toks} tokens · {C.DIM}unloaded{C.RESET}")
    plugin_mgr.call_hook("post_answer_b", answer_b)

    # ── Stage 2: Critique ───────────────────────────────────────────────
    _stage_header("Stage 2", f"Critique · {model_small} · temp=0.3")
    critique_prompt = CRITIQUE_PROMPT.format(
        query=query, solution_a=answer_a, solution_b=answer_b,
    )
    critique, lat, toks = _stream_stage(
        client, model_small, critique_prompt,
        temperature=0.3, color=C.MAGENTA,
    )
    if critique is None:
        return None
    latencies["stage_2"] = lat
    token_counts["stage_2"] = toks
    print(f"  {C.GREEN}✓{C.RESET} Critique  {lat:.1f}s · {toks} tokens · {C.DIM}unloaded{C.RESET}")
    plugin_mgr.call_hook("post_critique", critique)

    # ── Stage 3: Synthesis ──────────────────────────────────────────────
    _stage_header("Stage 3", f"Synthesis · {model_large} · temp=0.2")
    synthesis_prompt = SYNTHESIS_PROMPT.format(
        query=query, solution_a=answer_a, solution_b=answer_b,
        critique=critique,
    )
    synthesis, lat, toks = _stream_stage(
        client, model_large, synthesis_prompt,
        temperature=0.2, color=C.BWHITE,
    )
    if synthesis is None:
        return None
    latencies["stage_3"] = lat
    token_counts["stage_3"] = toks
    print(f"  {C.GREEN}✓{C.RESET} Synthesis  {lat:.1f}s · {toks} tokens · {C.DIM}unloaded{C.RESET}")
    plugin_mgr.call_hook("post_synthesis", synthesis)

    # ── Complete ────────────────────────────────────────────────────────
    total_s = time.perf_counter() - total_t0
    total_tokens = sum(token_counts.values())
    latencies["total"] = round(total_s, 1)

    _stage_header("Complete", "")
    print(
        f"  {C.GREEN}✓{C.RESET} Total: {total_s:.1f}s · {total_tokens} tokens "
        f"· {C.DIM}peak RAM ~one model at a time{C.RESET}"
    )

    # Log the run
    _builtin_logger.log_run(
        query=query,
        model_small=model_small,
        model_large=model_large,
        answer_a=answer_a,
        answer_b=answer_b,
        critique=critique,
        synthesis=synthesis,
        latencies=latencies,
    )

    return synthesis


def _stream_stage(
    client: ollama.Client,
    model: str,
    prompt: str,
    temperature: float,
    color: str,
) -> tuple:
    """Stream a single stage, returning (text, latency_seconds, token_count).

    Returns (None, 0, 0) on error.
    """
    spinner = Spinner(f"Waiting for {model}…", color=color).start()
    t0 = time.perf_counter()

    try:
        stream = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature},
            keep_alive=0,
            stream=True,
        )

        chunks: List[str] = []
        first_token = True

        for chunk in stream:
            if first_token:
                spinner.stop()
                first_token = False
            token = chunk.get("response", "")
            chunks.append(token)
            sys.stdout.write(f"{color}{token}{C.RESET}")
            sys.stdout.flush()

        elapsed = time.perf_counter() - t0

        if first_token:
            spinner.fail("No response received")
            return (None, 0, 0)

        text = "".join(chunks)
        # Ensure newline after streamed content
        if text and not text.endswith("\n"):
            print()

        return (text, elapsed, len(chunks))

    except KeyboardInterrupt:
        spinner.stop()
        print(f"\n  {C.YELLOW}↩{C.RESET} Interrupted. Partial output above.")
        return (None, 0, 0)
    except Exception as exc:
        spinner.fail(f"Stage failed: {exc}")
        return (None, 0, 0)


def run_direct_interactive(
    client: ollama.Client,
    model_large: str,
    query: str,
    loaded_files: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Send a query directly to the large model with live streaming.

    Skips the council pipeline entirely. Same visual style as Stage 3
    (bright white, spinner until first token, completion stats).

    Returns the response text, or None on failure.
    """
    # Inject loaded file context
    if loaded_files:
        context_block = "\n\n".join(
            f"# File: {fname}\n```\n{content}\n```"
            for fname, content in loaded_files.items()
        )
        query = f"Context files:\n{context_block}\n\nQuery: {query}"

    _stage_header("Quick", f"Direct · {model_large} · temp=0.2")
    text, lat, toks = _stream_stage(
        client, model_large, query,
        temperature=0.2, color=C.BWHITE,
    )
    if text is None:
        return None
    print(f"  {C.GREEN}✓{C.RESET} Done  {lat:.1f}s · {toks} tokens · {C.DIM}unloaded{C.RESET}")
    return text


# ── REPL commands ────────────────────────────────────────────────────────────

def _cmd_models(model_small: str, model_large: str, models: List[Dict[str, Any]]):
    """Show current model pair + RAM estimates."""
    small_size = "?"
    large_size = "?"
    for m in models:
        name = m.get("name", m.get("model", ""))
        if name == model_small:
            small_size = format_model_size(m.get("size", 0))
        if name == model_large:
            large_size = format_model_size(m.get("size", 0))
    print(f"\n  {C.CYAN}Small{C.RESET} (stages 1a, 1b, 2): {C.BOLD}{model_small}{C.RESET}  {C.DIM}(~{small_size}){C.RESET}")
    print(f"  {C.MAGENTA}Large{C.RESET} (stage 3):          {C.BOLD}{model_large}{C.RESET}  {C.DIM}(~{large_size}){C.RESET}\n")


def _cmd_logs():
    """Show last 5 runs from logs."""
    log_files = sorted(LOGS_DIR.glob("*.jsonl"), reverse=True)
    if not log_files:
        print(f"  {C.DIM}No logs yet.{C.RESET}\n")
        return

    entries = []
    for f in log_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            pass
        if len(entries) >= 5:
            break

    # Sort by timestamp descending and take 5
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    entries = entries[:5]

    if not entries:
        print(f"  {C.DIM}No log entries found.{C.RESET}\n")
        return

    print(f"\n  {C.BOLD}Last {len(entries)} runs:{C.RESET}\n")
    for e in entries:
        ts = e.get("timestamp", "?")[:19]
        q = e.get("query", "?")[:40]
        total = e.get("latencies", {}).get("total", "?")
        print(f"    {C.DIM}{ts}{C.RESET}  {q:<40}  {C.CYAN}{total}s{C.RESET}")
    print()


def _cmd_install(url: str):
    """Download a .py plugin from a URL into ~/.sutra/plugins/."""
    if not url:
        print(f"  {C.RED}✗{C.RESET} Usage: /install <url>\n")
        return

    # Validate URL ends in .py
    clean_url = url.rstrip("/").split("?")[0].split("#")[0]
    if not clean_url.endswith(".py"):
        print(f"  {C.RED}✗{C.RESET} URL must point to a .py file.")
        print(f"  {C.DIM}Example: https://raw.githubusercontent.com/user/repo/main/plugin.py{C.RESET}\n")
        return

    filename = clean_url.split("/")[-1]
    dest = PLUGINS_DIR / filename
    print(f"  {C.DIM}Downloading {url}…{C.RESET}")

    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  {C.GREEN}✓{C.RESET} Installed to {dest}")
        print(f"  {C.DIM}Reload with /plugins to activate.{C.RESET}\n")
    except Exception as exc:
        print(f"  {C.RED}✗{C.RESET} Download failed: {exc}\n")


def _cmd_help():
    """Print all REPL commands."""
    print(f"\n  {C.BOLD}Commands:{C.RESET}")
    cmds = [
        ("/quick <query>",  "skip council, send directly to large model with streaming"),
        ("/council <query>","explicitly run council pipeline on the query"),
        ("/save",           "save last output to a file (prompts for path)"),
        ("/load <file>",    "load a file into context (attached to every query)"),
        ("/unload [file]",  "remove file from context, or all if no arg"),
        ("/project",        "show workspace (saved files) and context (loaded files)"),
        ("/models",         "current model pair + RAM estimates"),
        ("/swap",           "swap small ↔ large roles"),
        ("/cache",          "cache stats"),
        ("/clear",          "clear cache"),
        ("/plugins",        "list loaded plugins, their hooks, file paths"),
        ("/install <url>",  "download .py from URL into ~/.sutra/plugins/, reload"),
        ("/logs",           "last 5 runs: timestamp, query preview, total latency"),
        ("/help",           "show this help"),
        ("/exit",           "quit"),
    ]
    print(f"\n  {C.BOLD}Input:{C.RESET}")
    inputs = [
        ('"""',             "open/close multiline block (paste code, etc.)"),
        ("(any text)",      "loaded files are auto-attached to every query until /unload"),
    ]
    for cmd, desc in cmds:
        print(f"    {C.CYAN}{cmd:<20}{C.RESET} — {C.DIM}{desc}{C.RESET}")
    for cmd, desc in inputs:
        print(f"    {C.CYAN}{cmd:<20}{C.RESET} — {C.DIM}{desc}{C.RESET}")
    print()


# ── REPL ─────────────────────────────────────────────────────────────────────

def _read_multiline() -> Optional[str]:
    """Read a multiline block delimited by triple-quotes.

    Shows ``...`` prompt for each continuation line.
    Returns the joined text, or None if cancelled with Ctrl+C.
    """
    lines: List[str] = []
    try:
        while True:
            continuation = input(f"  {C.DIM}...{C.RESET} ")
            if continuation.strip() == '"""':
                break
            lines.append(continuation)
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {C.YELLOW}↩{C.RESET} Multiline input cancelled.")
        return None
    return "\n".join(lines)


def repl(
    model_small: str,
    model_large: str,
    host: Optional[str] = None,
    models: Optional[List[Dict[str, Any]]] = None,
):
    """Main interactive read-eval-print loop."""
    client = ollama.Client(host=host) if host else ollama.Client()
    cache = DemonstrationCache(default_ttl=3600)
    all_models = models or []
    last_output: Optional[str] = None   # tracks last synthesis / quick output
    workspace: Dict[str, str] = {}      # {filename: full_path} — saved files this session
    loaded_files: Dict[str, str] = {}   # {filename: content} — files attached to context

    # Load plugins
    plugin_mgr = PluginManager()
    plugin_mgr.load_all()
    plugin_count = len(plugin_mgr.plugins)
    if plugin_count:
        print(f"  {C.GREEN}✓{C.RESET} {plugin_count} plugin(s) loaded")

    _cmd_help()

    while True:
        try:
            prompt_str = f"  {C.BMAGENTA}{C.BOLD}❯{C.RESET} "
            query = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {C.DIM}Goodbye!{C.RESET}\n")
            break

        if not query:
            continue

        # ── Multiline input ──────────────────────────────────────────
        if query == '"""':
            ml = _read_multiline()
            if ml is None or not ml.strip():
                continue
            query = ml

        # ── Slash commands ───────────────────────────────────────────
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                print(f"\n  {C.DIM}Goodbye!{C.RESET}\n")
                break

            elif cmd == "/quick":
                q = parts[1].strip() if len(parts) > 1 else ""
                if not q:
                    print(f"  {C.RED}✗{C.RESET} Usage: /quick <query>\n")
                else:
                    try:
                        print()
                        result = run_direct_interactive(
                            client=client,
                            model_large=model_large,
                            query=q,
                            loaded_files=loaded_files,
                        )
                        if result is not None:
                            last_output = result
                        print()
                    except KeyboardInterrupt:
                        print(f"\n  {C.YELLOW}↩{C.RESET} Interrupted. Partial output above.")
                    except Exception:
                        print(f"\n  {C.RED}✗{C.RESET} Unexpected error:\n")
                        traceback.print_exc()
                        print(f"\n  {C.DIM}Returning to prompt.{C.RESET}\n")

            elif cmd == "/council":
                q = parts[1].strip() if len(parts) > 1 else ""
                if not q:
                    print(f"  {C.RED}✗{C.RESET} Usage: /council <query>\n")
                else:
                    try:
                        print()
                        result = run_council_interactive(
                            client=client,
                            model_small=model_small,
                            model_large=model_large,
                            query=q,
                            cache=cache,
                            plugin_mgr=plugin_mgr,
                            loaded_files=loaded_files,
                        )
                        if result is not None:
                            last_output = result
                        print()
                    except KeyboardInterrupt:
                        print(f"\n  {C.YELLOW}↩{C.RESET} Interrupted. Partial output above.")
                    except Exception:
                        print(f"\n  {C.RED}✗{C.RESET} Unexpected error:\n")
                        traceback.print_exc()
                        print(f"\n  {C.DIM}Returning to prompt.{C.RESET}\n")

            elif cmd == "/save":
                if last_output is None:
                    print(f"  {C.RED}✗{C.RESET} Nothing to save yet.\n")
                else:
                    try:
                        raw_path = input(f"  {C.BMAGENTA}▸{C.RESET} Save to (path or filename): ").strip()
                        if not raw_path:
                            print(f"  {C.YELLOW}↩{C.RESET} Cancelled.\n")
                        else:
                            p = Path(raw_path).expanduser()
                            if not p.is_absolute():
                                p = Path(os.getcwd()) / p
                            p = p.resolve()
                            os.makedirs(str(p.parent), exist_ok=True)
                            p.write_text(last_output, encoding="utf-8")
                            fname = p.name
                            workspace[fname] = str(p)
                            print(f"  {C.GREEN}✓{C.RESET} Saved to {C.BOLD}{p}{C.RESET} · registered in workspace\n")
                    except (EOFError, KeyboardInterrupt):
                        print(f"\n  {C.YELLOW}↩{C.RESET} Cancelled.\n")
                    except Exception as exc:
                        print(f"  {C.RED}✗{C.RESET} Save failed: {exc}\n")

            elif cmd == "/models":
                _cmd_models(model_small, model_large, all_models)

            elif cmd == "/swap":
                model_small, model_large = model_large, model_small
                print(f"  {C.GREEN}✓{C.RESET} Swapped! small={C.BOLD}{model_small}{C.RESET}  large={C.BOLD}{model_large}{C.RESET}\n")

            elif cmd == "/cache":
                print(f"  {C.CYAN}Cache entries:{C.RESET} {cache.size}")
                print(f"  {C.CYAN}Default TTL:{C.RESET}   {cache.default_ttl}s\n")

            elif cmd == "/clear":
                cache.clear()
                print(f"  {C.GREEN}✓{C.RESET} Cache cleared.\n")

            elif cmd == "/plugins":
                plugin_mgr.load_all()   # Refresh
                plugin_mgr.list_plugins()

            elif cmd == "/install":
                url = parts[1].strip() if len(parts) > 1 else ""
                _cmd_install(url)
                plugin_mgr.load_all()   # Reload after install

            elif cmd == "/load":
                farg = parts[1].strip() if len(parts) > 1 else ""
                if not farg:
                    print(f"  {C.RED}✗{C.RESET} Usage: /load <filepath or filename>\n")
                else:
                    # Resolve path: check workspace first, then treat as path
                    fpath = None
                    if os.sep not in farg and "/" not in farg and farg in workspace:
                        fpath = Path(workspace[farg])
                    else:
                        fpath = Path(farg).expanduser()
                        if not fpath.is_absolute():
                            fpath = Path(os.getcwd()) / fpath
                        fpath = fpath.resolve()

                    if not fpath.exists():
                        print(f"  {C.RED}✗{C.RESET} File not found: {fpath}\n")
                    else:
                        try:
                            content = fpath.read_text(encoding="utf-8")
                            token_est = int(len(content.split()) * 1.3)
                            if token_est > 800:
                                print(f"  {C.YELLOW}⚠ WARNING: This file is large (~{token_est} tokens).{C.RESET}")
                                print(f"  {C.YELLOW}⚠ The small model (3b) may struggle with this context.{C.RESET}")
                                print(f"  {C.YELLOW}⚠ Consider switching to a larger small model with /swap.{C.RESET}")
                                try:
                                    confirm = input(f"  {C.YELLOW}⚠ Continue anyway? [y/N]: {C.RESET}").strip().lower()
                                except (EOFError, KeyboardInterrupt):
                                    confirm = "n"
                                if confirm != "y":
                                    print(f"  {C.YELLOW}↩{C.RESET} Cancelled.\n")
                                    continue
                            fname = fpath.name
                            loaded_files[fname] = content
                            token_est = int(len(content.split()) * 1.3)
                            print(f"  {C.GREEN}✓{C.RESET} Loaded {C.BOLD}{fname}{C.RESET} (~{token_est} tokens) · attached to context\n")
                        except Exception as exc:
                            print(f"  {C.RED}✗{C.RESET} Failed to read: {exc}\n")

            elif cmd == "/unload":
                farg = parts[1].strip() if len(parts) > 1 else ""
                if not farg:
                    loaded_files.clear()
                    print(f"  {C.GREEN}✓{C.RESET} Context cleared\n")
                elif farg in loaded_files:
                    del loaded_files[farg]
                    print(f"  {C.GREEN}✓{C.RESET} Unloaded {C.BOLD}{farg}{C.RESET}\n")
                else:
                    print(f"  {C.RED}✗{C.RESET} '{farg}' not in context. Use /project to see loaded files.\n")

            elif cmd == "/project":
                if not workspace and not loaded_files:
                    print(f"  {C.DIM}No files in workspace or context yet.{C.RESET}\n")
                else:
                    if workspace:
                        print(f"\n  {C.BOLD}Workspace (saved this session):{C.RESET}\n")
                        for wname, wpath in workspace.items():
                            print(f"    {C.CYAN}{wname:<16}{C.RESET} → {C.DIM}{wpath}{C.RESET}")
                    if loaded_files:
                        print(f"\n  {C.BOLD}Context (attached to council):{C.RESET}\n")
                        for lname, lcontent in loaded_files.items():
                            toks = int(len(lcontent.split()) * 1.3)
                            print(f"    {C.CYAN}{lname:<16}{C.RESET} ~{toks} tokens")
                    print()

            elif cmd == "/logs":
                _cmd_logs()

            elif cmd in ("/help", "/?"):
                _cmd_help()

            else:
                print(f"  {C.YELLOW}?{C.RESET} Unknown command: {cmd}. Type /help\n")

            continue

        # ── Run council pipeline (default) ───────────────────────────
        try:
            print()
            result = run_council_interactive(
                client=client,
                model_small=model_small,
                model_large=model_large,
                query=query,
                cache=cache,
                plugin_mgr=plugin_mgr,
                loaded_files=loaded_files,
            )
            if result is not None:
                last_output = result
            print()
        except KeyboardInterrupt:
            print(f"\n  {C.YELLOW}↩{C.RESET} Interrupted. Partial output above.")
        except Exception:
            print(f"\n  {C.RED}✗{C.RESET} Unexpected error:\n")
            traceback.print_exc()
            print(f"\n  {C.DIM}Returning to prompt.{C.RESET}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="S.U.T.R.A — Council Mode CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python -m agent_handoff.cli
  python -m agent_handoff.cli --model-small qwen2.5-coder:3b --model-large qwen2.5-coder:7b
  python -m agent_handoff.cli --host http://192.168.1.10:11434
""",
    )
    parser.add_argument("--model-small", default=None, help="Small model for stages 1a, 1b, 2")
    parser.add_argument("--model-large", default=None, help="Large model for stage 3 (synthesis)")
    parser.add_argument("--host", default=None, help="Ollama server URL")
    args = parser.parse_args()

    _enable_ansi_windows()
    _ensure_dirs()

    # ── Banner ───────────────────────────────────────────────────────
    _header()

    # ── Detect models ────────────────────────────────────────────────
    spinner = Spinner("Connecting to Ollama…", color=C.CYAN).start()
    models = detect_models(host=args.host)

    if not models:
        spinner.fail("Cannot connect to Ollama.")
        print(f"\n  {C.RED}✗{C.RESET} Cannot connect to Ollama. Fix: run {C.BOLD}ollama serve{C.RESET} in a new terminal")
        print(f"  {C.DIM}Then pull models: {C.RESET}{C.BOLD}ollama pull qwen2.5-coder:3b{C.RESET}\n")
        sys.exit(1)

    model_names = [m.get("name", m.get("model", "?")) for m in models]
    spinner.stop(f"Found {C.BOLD}{len(models)}{C.RESET} model(s): {C.DIM}{', '.join(model_names)}{C.RESET}")

    # ── Model recommendations ────────────────────────────────────────
    _print_model_recommendations(models)

    # ── Select models ────────────────────────────────────────────────
    if args.model_small:
        model_small = args.model_small
        # Validate it exists
        if model_small not in model_names:
            print(f"  {C.RED}✗{C.RESET} Model '{model_small}' not found. Fix: run {C.BOLD}ollama pull {model_small}{C.RESET}")
            sys.exit(1)
        print(f"  {C.GREEN}✓{C.RESET} Small model (flag): {C.BOLD}{model_small}{C.RESET}")
    else:
        model_small = pick_model_council(models, "small")

    if args.model_large:
        model_large = args.model_large
        if model_large not in model_names:
            print(f"  {C.RED}✗{C.RESET} Model '{model_large}' not found. Fix: run {C.BOLD}ollama pull {model_large}{C.RESET}")
            sys.exit(1)
        print(f"  {C.GREEN}✓{C.RESET} Large model (flag): {C.BOLD}{model_large}{C.RESET}")
    else:
        model_large = pick_model_council(models, "large", exclude=model_small)

    # ── RAM warning ──────────────────────────────────────────────────
    small_size = 0
    large_size = 0
    for m in models:
        name = m.get("name", m.get("model", ""))
        if name == model_small:
            small_size = m.get("size", 0)
        if name == model_large:
            large_size = m.get("size", 0)

    peak_gb = max(small_size, large_size) / (1024 ** 3)
    if peak_gb > 6.0:
        print(f"  {C.YELLOW}⚠{C.RESET} Warning: estimated peak RAM {peak_gb:.1f}GB. Close other apps first.")

    # ── Pipeline summary ─────────────────────────────────────────────
    pipeline_desc = (
        f"Query → [{model_small}] temp=0.3 → Answer A\n"
        f"      → [{model_small}] temp=0.8 → Answer B\n"
        f"      → [{model_small}] temp=0.3 → Critique\n"
        f"      → [{model_large}] temp=0.2 → Synthesis"
    )
    print(_box("Council Pipeline", pipeline_desc, color=C.BMAGENTA))

    # ── Start REPL ───────────────────────────────────────────────────
    repl(
        model_small=model_small,
        model_large=model_large,
        host=args.host,
        models=models,
    )


if __name__ == "__main__":
    main()
