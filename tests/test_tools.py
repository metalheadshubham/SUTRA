"""Tests for agent_handoff.tools — deterministic tool executor."""

import os
import tempfile
import pytest

from agent_handoff.tools import (
    ToolResult,
    write_file,
    delete_file,
    replace_in_file,
    read_file,
    list_dir,
    file_exists,
    run_command,
    execute_tool,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory as cwd."""
    return str(tmp_path)


# -- ACT: write_file ---------------------------------------------------------

class TestWriteFile:
    def test_creates_file(self, tmp_dir):
        r = write_file("test.txt", "hello", cwd=tmp_dir)
        assert r.success is True
        assert "test.txt" in r.output
        assert os.path.exists(os.path.join(tmp_dir, "test.txt"))

    def test_creates_nested_dirs(self, tmp_dir):
        r = write_file("a/b/c.txt", "nested", cwd=tmp_dir)
        assert r.success is True
        assert os.path.exists(os.path.join(tmp_dir, "a", "b", "c.txt"))

    def test_overwrites_existing(self, tmp_dir):
        write_file("x.txt", "first", cwd=tmp_dir)
        r = write_file("x.txt", "second", cwd=tmp_dir)
        assert r.success is True
        with open(os.path.join(tmp_dir, "x.txt")) as f:
            assert f.read() == "second"

    def test_blocks_path_escape(self, tmp_dir):
        r = write_file("../escape.txt", "x", cwd=tmp_dir)
        assert r.success is False
        assert r.error == "[ERROR] Path outside workspace boundary"


# -- ACT: delete_file --------------------------------------------------------

class TestDeleteFile:
    def test_deletes_existing(self, tmp_dir):
        write_file("del.txt", "bye", cwd=tmp_dir)
        r = delete_file("del.txt", cwd=tmp_dir)
        assert r.success is True
        assert not os.path.exists(os.path.join(tmp_dir, "del.txt"))

    def test_fails_on_missing(self, tmp_dir):
        r = delete_file("nope.txt", cwd=tmp_dir)
        assert r.success is False
        assert r.error is not None


# -- ACT: replace_in_file ----------------------------------------------------

class TestReplaceInFile:
    def test_replaces_text(self, tmp_dir):
        write_file("r.txt", "hello world", cwd=tmp_dir)
        r = replace_in_file("r.txt", "hello", "goodbye", cwd=tmp_dir)
        assert r.success is True
        with open(os.path.join(tmp_dir, "r.txt")) as f:
            assert f.read() == "goodbye world"

    def test_fails_if_not_found(self, tmp_dir):
        write_file("r2.txt", "abc", cwd=tmp_dir)
        r = replace_in_file("r2.txt", "xyz", "new", cwd=tmp_dir)
        assert r.success is False

    def test_fails_on_missing_file(self, tmp_dir):
        r = replace_in_file("no.txt", "a", "b", cwd=tmp_dir)
        assert r.success is False


# -- CHECK: read_file --------------------------------------------------------

class TestReadFile:
    def test_reads_content(self, tmp_dir):
        write_file("read.txt", "data here", cwd=tmp_dir)
        r = read_file("read.txt", cwd=tmp_dir)
        assert r.success is True
        assert r.output == "data here"

    def test_fails_on_missing(self, tmp_dir):
        r = read_file("missing.txt", cwd=tmp_dir)
        assert r.success is False

    def test_blocks_path_escape(self, tmp_dir):
        # Any attempt to leave the workspace via ../ should be blocked.
        write_file("safe.txt", "ok", cwd=tmp_dir)
        r = read_file("../safe.txt", cwd=tmp_dir)
        assert r.success is False
        assert r.error == "[ERROR] Path outside workspace boundary"


# -- CHECK: list_dir ---------------------------------------------------------

class TestListDir:
    def test_lists_files(self, tmp_dir):
        write_file("a.txt", "a", cwd=tmp_dir)
        write_file("b.txt", "b", cwd=tmp_dir)
        r = list_dir(".", cwd=tmp_dir)
        assert r.success is True
        assert "a.txt" in r.output
        assert "b.txt" in r.output

    def test_fails_on_non_dir(self, tmp_dir):
        write_file("f.txt", "f", cwd=tmp_dir)
        r = list_dir("f.txt", cwd=tmp_dir)
        assert r.success is False


# -- CHECK: file_exists ------------------------------------------------------

class TestFileExists:
    def test_true_when_exists(self, tmp_dir):
        write_file("e.txt", "e", cwd=tmp_dir)
        r = file_exists("e.txt", cwd=tmp_dir)
        assert r.success is True
        assert r.output == "true"

    def test_false_when_missing(self, tmp_dir):
        r = file_exists("nope.txt", cwd=tmp_dir)
        assert r.success is True
        assert r.output == "false"


# -- VERIFY: run_command -----------------------------------------------------

class TestRunCommand:
    def test_echo(self, tmp_dir):
        r = run_command("echo hello", cwd=tmp_dir)
        assert r.success is True
        assert "hello" in r.output

    def test_blocks_dangerous(self, tmp_dir):
        r = run_command("rm -rf /", cwd=tmp_dir)
        assert r.success is False
        assert "Blocked" in r.error


# -- execute_tool dispatch ----------------------------------------------------

class TestExecuteTool:
    def test_dispatch_write(self, tmp_dir):
        r = execute_tool("ACT", "write_file", {"path": "d.txt", "content": "dispatched"}, cwd=tmp_dir)
        assert r.success is True

    def test_dispatch_unknown(self, tmp_dir):
        r = execute_tool("ACT", "teleport", {}, cwd=tmp_dir)
        assert r.success is False
        assert "Unknown tool" in r.error
