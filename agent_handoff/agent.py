"""
agent.py -- Strict ARC mode agent loop.

ARC = deterministic tool-only execution environment.
The model is NOT allowed to explain, describe, or plan inside ARC.
It outputs TOOL commands or DONE. Nothing else.

Invalid outputs are rejected and retried (max 2 retries per step).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import ollama

from agent_handoff.tools import ToolResult, execute_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed command types
# ---------------------------------------------------------------------------

@dataclass
class ToolCommand:
    """A parsed TOOL: line from model output."""
    category: str   # ACT, CHECK, VERIFY
    action: str     # write_file, read_file, etc.
    params: dict    # key=value pairs


@dataclass
class DoneSignal:
    """A parsed DONE: line from model output."""
    message: str


@dataclass
class InvalidOutput:
    """Model produced something that is neither TOOL nor DONE."""
    raw: str


# ---------------------------------------------------------------------------
# ARC step record (kept in sliding window)
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One step in the ARC execution history."""
    step: int
    command: str        # raw command string
    result_success: bool
    result_output: str
    result_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Output parser -- strict, no tolerance
# ---------------------------------------------------------------------------

# Pattern: TOOL: CATEGORY action key="value" key="value"
_TOOL_RE = re.compile(
    r'^TOOL:\s+(ACT|CHECK|VERIFY)\s+(\w+)\s*(.*)',
    re.IGNORECASE,
)

# Pattern for key=value or key="value with spaces"
_PARAM_RE = re.compile(
    r'(\w+)=(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|(\S+))',
)

# Pattern: DONE: message
_DONE_RE = re.compile(r'^DONE:\s*(.*)', re.IGNORECASE | re.DOTALL)


def parse_arc_output(raw: str) -> ToolCommand | DoneSignal | InvalidOutput:
    """Parse model output into exactly one of: ToolCommand, DoneSignal, InvalidOutput.

    The model MUST output exactly one line starting with TOOL: or DONE:.
    Anything else is invalid.
    """
    # Strip whitespace, take the first meaningful line
    stripped = raw.strip()
    if not stripped:
        return InvalidOutput(raw="(empty output)")

    # Try multi-line: find the first TOOL: or DONE: line
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check DONE first (simpler)
        done_match = _DONE_RE.match(line)
        if done_match:
            return DoneSignal(message=done_match.group(1).strip())

        # Check TOOL
        tool_match = _TOOL_RE.match(line)
        if tool_match:
            category = tool_match.group(1).upper()
            action = tool_match.group(2)
            param_str = tool_match.group(3)

            # For write_file, content may span multiple lines after the TOOL line
            params = {}
            if action == "write_file" and "content=" in stripped:
                # Extract content specially: everything after content=
                # First get non-content params
                before_content = param_str.split("content=")[0] if "content=" in param_str else param_str
                for m in _PARAM_RE.finditer(before_content):
                    key = m.group(1)
                    val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
                    params[key] = _unescape(val)

                # Extract content: everything between content=" and the last "
                content_match = re.search(r'content="(.*)"', stripped, re.DOTALL)
                if content_match:
                    params["content"] = _unescape(content_match.group(1))
                else:
                    # Try single-quote variant
                    content_match = re.search(r"content='(.*)'", stripped, re.DOTALL)
                    if content_match:
                        params["content"] = _unescape(content_match.group(1))
                    else:
                        # Fallback: grab unquoted content
                        content_match = re.search(r'content=(\S+)', stripped)
                        if content_match:
                            params["content"] = content_match.group(1)
            elif action == "replace_in_file":
                # Similar multi-value extraction for old= and new=
                # Extract path first
                path_match = re.search(r'path="([^"]*)"', param_str)
                if path_match:
                    params["path"] = path_match.group(1)
                else:
                    path_match = re.search(r'path=(\S+)', param_str)
                    if path_match:
                        params["path"] = path_match.group(1)

                # Extract old and new from full stripped text
                old_match = re.search(r'old="((?:[^"\\]|\\.)*)"', stripped)
                new_match = re.search(r'new="((?:[^"\\]|\\.)*)"', stripped)
                if old_match:
                    params["old"] = _unescape(old_match.group(1))
                if new_match:
                    params["new"] = _unescape(new_match.group(1))
            else:
                # Normal param parsing
                for m in _PARAM_RE.finditer(param_str):
                    key = m.group(1)
                    val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
                    params[key] = _unescape(val)

            return ToolCommand(category=category, action=action, params=params)

    # Nothing matched
    return InvalidOutput(raw=stripped[:200])


def _unescape(s: str) -> str:
    """Unescape basic escape sequences from parsed strings."""
    return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# ARC system prompt
# ---------------------------------------------------------------------------

ARC_SYSTEM_PROMPT = """\
You are in ARC mode. You are a deterministic tool executor.

RULES:
- You MUST respond with EXACTLY ONE of the two formats below.
- You must NOT explain, describe, plan, or output natural language.
- You must NOT combine a TOOL command with any other text.
- Respond with ONLY the command. Nothing else.

FORMAT 1 - Execute a tool:
TOOL: <CATEGORY> <action> <key="value" ...>

FORMAT 2 - Signal completion:
DONE: <short status message>

AVAILABLE TOOLS:
  TOOL: ACT write_file path="<path>" content="<content>"
  TOOL: ACT delete_file path="<path>"
  TOOL: ACT replace_in_file path="<path>" old="<text>" new="<text>"
  TOOL: CHECK read_file path="<path>"
  TOOL: CHECK list_dir path="<path>"
  TOOL: CHECK file_exists path="<path>"
  TOOL: VERIFY run_command cmd="<command>"

IMPORTANT:
- After writing a file, use CHECK file_exists to confirm.
- If a previous step failed, try to fix it or DONE with error.
- Never guess filesystem state. Use CHECK first.
"""


# ---------------------------------------------------------------------------
# Plan prompt (runs BEFORE ARC, normal LLM mode)
# ---------------------------------------------------------------------------

PLAN_PROMPT = """\
You are a planning assistant. The user wants you to complete a task using tools.

Available tools:
  ACT write_file, delete_file, replace_in_file
  CHECK read_file, list_dir, file_exists
  VERIFY run_command

Create a SHORT numbered plan (max 6 steps) for this task.
Output ONLY the numbered list. No explanation.

Task: {task}
"""


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Strict ARC mode agent loop.

    1. Plan phase: model generates a short plan (normal LLM mode)
    2. ARC phase: model executes tools deterministically
    3. Exit: model signals DONE or max iterations reached

    Args:
        model: Ollama model tag to use for both planning and ARC.
        client: Ollama client instance.
        max_steps: Hard cap on ARC iterations (default 8).
        cwd: Working directory for tool execution.
        arc_memory_size: Number of recent steps to keep in context (default 3).
    """

    def __init__(
        self,
        model: str,
        client: ollama.Client,
        max_steps: int = 8,
        cwd: Optional[str] = None,
        arc_memory_size: int = 3,
    ):
        if cwd is None:
            raise ValueError("Workspace is not set")
        self.model = model
        self.client = client
        self.max_steps = max_steps
        self.cwd = cwd
        self.arc_memory_size = arc_memory_size

    def plan(self, task: str) -> List[str]:
        """Generate a short execution plan. Returns list of step strings."""
        prompt = PLAN_PROMPT.format(task=task)
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.2},
                keep_alive=0,
            )
            raw = response.get("response", "")
        except Exception as exc:
            logger.error("Plan generation failed: %s", exc)
            return [f"1. Execute task: {task}"]

        # Parse numbered lines
        steps = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Strip leading number/bullet
                cleaned = re.sub(r'^[\d]+[.):\-]\s*', '', line).strip()
                cleaned = re.sub(r'^[-*]\s*', '', cleaned).strip()
                if cleaned:
                    steps.append(cleaned)
        return steps if steps else [f"Execute task: {task}"]

    def run(self, task: str, on_step=None) -> "AgentResult":
        """Execute the full ARC loop.

        Args:
            task: The user's task description.
            on_step: Optional callback(step_num, event_type, data) for UI rendering.
                     event_type: "model_call", "tool_call", "tool_result",
                                 "verify", "invalid", "done", "abort"

        Returns:
            AgentResult with full execution trace.
        """
        # Build initial ARC context
        history: List[StepRecord] = []
        files_created: List[str] = []
        files_modified: List[str] = []
        files_deleted: List[str] = []
        commands_run: List[str] = []

        step = 0
        final_message = ""
        status = "SUCCESS"

        while step < self.max_steps:
            step += 1

            # Notify: model call
            if on_step:
                on_step(step, "model_call", {"model": self.model})

            # Build ARC prompt with sliding window of recent history
            arc_prompt = self._build_arc_prompt(task, history)

            # Call model
            raw_output = self._call_model(arc_prompt)
            if raw_output is None:
                status = "ERROR"
                final_message = "Model call failed"
                if on_step:
                    on_step(step, "abort", {"reason": "Model call failed"})
                break

            # Parse output -- strict enforcement
            parsed = parse_arc_output(raw_output)
            retries = 0
            max_retries = 2

            while isinstance(parsed, InvalidOutput) and retries < max_retries:
                retries += 1
                if on_step:
                    on_step(step, "invalid", {
                        "raw": parsed.raw,
                        "retry": retries,
                        "max_retries": max_retries,
                    })

                # Retry with stricter prompt
                retry_prompt = (
                    f"{arc_prompt}\n\n"
                    f"YOUR LAST OUTPUT WAS INVALID: {parsed.raw[:100]}\n"
                    f"You MUST respond with ONLY: TOOL: ... OR DONE: ...\n"
                    f"Nothing else. Try again."
                )
                raw_output = self._call_model(retry_prompt)
                if raw_output is None:
                    break
                parsed = parse_arc_output(raw_output)

            # Handle parsed result
            if isinstance(parsed, InvalidOutput):
                if on_step:
                    on_step(step, "abort", {"reason": f"Invalid output after {max_retries} retries"})
                status = "ABORT"
                final_message = f"Model produced invalid output after {max_retries} retries"
                break

            if isinstance(parsed, DoneSignal):
                final_message = parsed.message
                if on_step:
                    on_step(step, "done", {"message": parsed.message})
                break

            # It's a ToolCommand -- execute it
            if on_step:
                on_step(step, "tool_call", {
                    "category": parsed.category,
                    "action": parsed.action,
                    "params": parsed.params,
                    "raw": raw_output.strip().splitlines()[0] if raw_output else "",
                })

            result = execute_tool(
                category=parsed.category,
                action=parsed.action,
                params=dict(parsed.params),  # copy to avoid mutation
                cwd=self.cwd,
            )

            if on_step:
                on_step(step, "tool_result", {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                })

            # Track file changes
            if parsed.action == "write_file" and result.success:
                path = parsed.params.get("path", "?")
                if path not in files_created and path not in files_modified:
                    files_created.append(path)
            elif parsed.action == "replace_in_file" and result.success:
                path = parsed.params.get("path", "?")
                if path not in files_modified:
                    files_modified.append(path)
            elif parsed.action == "delete_file" and result.success:
                path = parsed.params.get("path", "?")
                files_deleted.append(path)
            elif parsed.action == "run_command":
                commands_run.append(parsed.params.get("cmd", "?"))

            # Verification for CHECK/VERIFY results
            if parsed.category == "VERIFY" or parsed.category == "CHECK":
                if on_step:
                    on_step(step, "verify", {
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    })

            # Record step in history
            history.append(StepRecord(
                step=step,
                command=f"TOOL: {parsed.category} {parsed.action} {_format_params_short(parsed.params)}",
                result_success=result.success,
                result_output=result.output[:200],  # truncate for memory
                result_error=result.error,
            ))

            # Trim history to sliding window
            if len(history) > self.arc_memory_size:
                history = history[-self.arc_memory_size:]

        else:
            # Max iterations reached
            status = "MAX_STEPS"
            final_message = f"Reached maximum steps ({self.max_steps})"

        return AgentResult(
            task=task,
            status=status,
            message=final_message,
            steps_used=step,
            max_steps=self.max_steps,
            files_created=files_created,
            files_modified=files_modified,
            files_deleted=files_deleted,
            commands_run=commands_run,
        )

    def _build_arc_prompt(self, task: str, history: List[StepRecord]) -> str:
        """Build the ARC prompt with task + recent history only."""
        parts = [ARC_SYSTEM_PROMPT, f"TASK: {task}"]

        if history:
            parts.append("\nRECENT ACTIONS:")
            for rec in history:
                parts.append(f"  [{rec.step}] {rec.command}")
                if rec.result_success:
                    parts.append(f"      -> OK: {rec.result_output[:100]}")
                else:
                    parts.append(f"      -> FAIL: {rec.result_error or 'unknown error'}")

        parts.append("\nNEXT COMMAND:")
        return "\n".join(parts)

    def _call_model(self, prompt: str) -> Optional[str]:
        """Call the model. Returns raw output or None on failure."""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1},  # near-deterministic
                keep_alive=0,
            )
            return response.get("response", "")
        except Exception as exc:
            logger.error("ARC model call failed: %s", exc)
            return None


def _format_params_short(params: dict) -> str:
    """Format params dict compactly, truncating long values."""
    parts = []
    for k, v in params.items():
        if k == "cwd":
            continue
        sv = str(v)
        if len(sv) > 40:
            sv = sv[:37] + "..."
        parts.append(f'{k}="{sv}"')
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Final result of an agent run."""
    task: str
    status: str             # SUCCESS, ERROR, ABORT, MAX_STEPS
    message: str
    steps_used: int
    max_steps: int
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
