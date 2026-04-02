"""
tools.py -- Deterministic tool executor for ARC mode.

Three tool categories:
  ACT    -> write_file, delete_file, replace_in_file
  CHECK  -> read_file, list_dir, file_exists
  VERIFY -> run_command

Every tool returns a ToolResult. No silent retries. No assumptions.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolResult:
    """Structured result from any tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Workspace boundary enforcement
# ---------------------------------------------------------------------------

def _resolve_within_workspace(path: str, cwd: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Resolve `path` into an absolute path under `cwd` only.

    Returns: (resolved_path, error). If error is not None, resolved_path is None.
    """
    if cwd is None:
        return None, "[ERROR] Workspace is not set"

    workspace_root = os.path.abspath(cwd)
    workspace_root_norm = os.path.normcase(workspace_root)

    # Absolute path is allowed only if it stays within workspace.
    if os.path.isabs(path):
        candidate = os.path.abspath(path)
    else:
        candidate = os.path.abspath(os.path.join(workspace_root, path))

    candidate_norm = os.path.normcase(candidate)

    try:
        common = os.path.commonpath([workspace_root_norm, candidate_norm])
    except Exception:
        common = None

    if common != workspace_root_norm:
        return None, "[ERROR] Path outside workspace boundary"

    return candidate, None


# ---------------------------------------------------------------------------
# ACT tools
# ---------------------------------------------------------------------------

def write_file(path: str, content: str, cwd: Optional[str] = None) -> ToolResult:
    """Create or overwrite a file with the given content."""
    try:
        full, err = _resolve_within_workspace(path, cwd)
        if err:
            return ToolResult(success=False, output="", error=err)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        size = os.path.getsize(full)
        return ToolResult(success=True, output=f"File created: {path} ({size} bytes)")
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"write_file failed: {exc}")


def delete_file(path: str, cwd: Optional[str] = None) -> ToolResult:
    """Delete a file."""
    try:
        full, err = _resolve_within_workspace(path, cwd)
        if err:
            return ToolResult(success=False, output="", error=err)
        if not os.path.exists(full):
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        os.remove(full)
        return ToolResult(success=True, output=f"File deleted: {path}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"delete_file failed: {exc}")


def replace_in_file(path: str, old: str, new: str, cwd: Optional[str] = None) -> ToolResult:
    """Replace exact text in a file. Fails if old text not found."""
    try:
        full, err = _resolve_within_workspace(path, cwd)
        if err:
            return ToolResult(success=False, output="", error=err)
        if not os.path.exists(full):
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if old not in content:
            return ToolResult(success=False, output="", error=f"Target text not found in {path}")
        content = content.replace(old, new, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(success=True, output=f"Replaced in {path}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"replace_in_file failed: {exc}")


# ---------------------------------------------------------------------------
# CHECK tools
# ---------------------------------------------------------------------------

def read_file(path: str, cwd: Optional[str] = None) -> ToolResult:
    """Read and return full file contents."""
    try:
        full, err = _resolve_within_workspace(path, cwd)
        if err:
            return ToolResult(success=False, output="", error=err)
        if not os.path.exists(full):
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(success=True, output=content)
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"read_file failed: {exc}")


def list_dir(path: str = ".", cwd: Optional[str] = None) -> ToolResult:
    """List directory contents."""
    try:
        full, err = _resolve_within_workspace(path, cwd)
        if err:
            return ToolResult(success=False, output="", error=err)
        if not os.path.isdir(full):
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")
        entries = sorted(os.listdir(full))
        listing = []
        for e in entries:
            fp = os.path.join(full, e)
            if os.path.isdir(fp):
                listing.append(f"  {e}/")
            else:
                size = os.path.getsize(fp)
                listing.append(f"  {e} ({size} bytes)")
        return ToolResult(success=True, output="\n".join(listing) if listing else "(empty)")
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"list_dir failed: {exc}")


def file_exists(path: str, cwd: Optional[str] = None) -> ToolResult:
    """Check if a file exists."""
    full, err = _resolve_within_workspace(path, cwd)
    if err:
        return ToolResult(success=False, output="", error=err)
    exists = os.path.exists(full)
    return ToolResult(success=True, output=str(exists).lower())


# ---------------------------------------------------------------------------
# VERIFY tools
# ---------------------------------------------------------------------------

# Commands containing these patterns are blocked.
_DANGEROUS = ["rm -rf /", "format c:", "del /s /q c:", "rmdir /s /q c:"]


def run_command(cmd: str, cwd: Optional[str] = None, timeout: int = 30) -> ToolResult:
    """Execute a shell command with timeout and safety checks."""
    if cwd is None:
        return ToolResult(success=False, output="", error="[ERROR] Workspace is not set")
    cmd_lower = cmd.lower().strip()
    for pattern in _DANGEROUS:
        if pattern in cmd_lower:
            return ToolResult(success=False, output="", error=f"Blocked dangerous command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return ToolResult(
                success=False,
                output=output,
                error=f"Exit code {result.returncode}: {err}" if err else f"Exit code {result.returncode}",
            )
        return ToolResult(success=True, output=output)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"Command timed out ({timeout}s): {cmd}")
    except Exception as exc:
        return ToolResult(success=False, output="", error=f"run_command failed: {exc}")


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

# Maps (category, action) -> function
TOOL_REGISTRY = {
    ("ACT", "write_file"): write_file,
    ("ACT", "delete_file"): delete_file,
    ("ACT", "replace_in_file"): replace_in_file,
    ("CHECK", "read_file"): read_file,
    ("CHECK", "list_dir"): list_dir,
    ("CHECK", "file_exists"): file_exists,
    ("VERIFY", "run_command"): run_command,
}


def execute_tool(category: str, action: str, params: dict, cwd: Optional[str] = None) -> ToolResult:
    """Dispatch a parsed tool command to the correct function."""
    if cwd is None:
        return ToolResult(success=False, output="", error="[ERROR] Workspace is not set")
    key = (category.upper(), action)
    func = TOOL_REGISTRY.get(key)
    if func is None:
        return ToolResult(
            success=False, output="",
            error=f"Unknown tool: {category} {action}",
        )
    # Inject cwd into params
    params["cwd"] = cwd
    try:
        return func(**params)
    except TypeError as exc:
        return ToolResult(success=False, output="", error=f"Bad params for {action}: {exc}")
