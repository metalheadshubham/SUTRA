"""Tests for agent_handoff.agent — ARC output parser and enforcement."""

import pytest

from agent_handoff.agent import (
    parse_arc_output,
    ToolCommand,
    DoneSignal,
    InvalidOutput,
)


# -- Valid TOOL commands ------------------------------------------------------

class TestParseToolCommands:
    def test_basic_check(self):
        result = parse_arc_output('TOOL: CHECK file_exists path="index.html"')
        assert isinstance(result, ToolCommand)
        assert result.category == "CHECK"
        assert result.action == "file_exists"
        assert result.params["path"] == "index.html"

    def test_act_write_file(self):
        result = parse_arc_output('TOOL: ACT write_file path="test.html" content="<html></html>"')
        assert isinstance(result, ToolCommand)
        assert result.category == "ACT"
        assert result.action == "write_file"
        assert result.params["path"] == "test.html"
        assert result.params["content"] == "<html></html>"

    def test_verify_run_command(self):
        result = parse_arc_output('TOOL: VERIFY run_command cmd="python test.py"')
        assert isinstance(result, ToolCommand)
        assert result.category == "VERIFY"
        assert result.action == "run_command"
        assert result.params["cmd"] == "python test.py"

    def test_act_delete_file(self):
        result = parse_arc_output('TOOL: ACT delete_file path="old.txt"')
        assert isinstance(result, ToolCommand)
        assert result.action == "delete_file"
        assert result.params["path"] == "old.txt"

    def test_check_list_dir(self):
        result = parse_arc_output('TOOL: CHECK list_dir path="."')
        assert isinstance(result, ToolCommand)
        assert result.action == "list_dir"

    def test_check_read_file(self):
        result = parse_arc_output('TOOL: CHECK read_file path="main.py"')
        assert isinstance(result, ToolCommand)
        assert result.action == "read_file"

    def test_case_insensitive_category(self):
        result = parse_arc_output('TOOL: act write_file path="a.txt" content="x"')
        assert isinstance(result, ToolCommand)
        assert result.category == "ACT"

    def test_replace_in_file(self):
        result = parse_arc_output('TOOL: ACT replace_in_file path="f.txt" old="hello" new="bye"')
        assert isinstance(result, ToolCommand)
        assert result.action == "replace_in_file"
        assert result.params["old"] == "hello"
        assert result.params["new"] == "bye"


# -- Valid DONE signals -------------------------------------------------------

class TestParseDoneSignal:
    def test_basic_done(self):
        result = parse_arc_output("DONE: task completed successfully")
        assert isinstance(result, DoneSignal)
        assert result.message == "task completed successfully"

    def test_done_with_empty_message(self):
        result = parse_arc_output("DONE:")
        assert isinstance(result, DoneSignal)

    def test_done_case_insensitive(self):
        result = parse_arc_output("done: finished")
        assert isinstance(result, DoneSignal)


# -- Invalid output (MUST be rejected) ----------------------------------------

class TestParseInvalid:
    def test_plain_text(self):
        result = parse_arc_output("I will now create the file for you.")
        assert isinstance(result, InvalidOutput)

    def test_empty_string(self):
        result = parse_arc_output("")
        assert isinstance(result, InvalidOutput)

    def test_whitespace_only(self):
        result = parse_arc_output("   \n  \n  ")
        assert isinstance(result, InvalidOutput)

    def test_json_output(self):
        result = parse_arc_output('{"action": "write_file", "path": "x"}')
        assert isinstance(result, InvalidOutput)

    def test_markdown_output(self):
        result = parse_arc_output("```python\nprint('hello')\n```")
        assert isinstance(result, InvalidOutput)

    def test_partial_tool_prefix(self):
        # "TOOL" without colon is invalid
        result = parse_arc_output("TOOL write_file")
        assert isinstance(result, InvalidOutput)


# -- Edge cases ---------------------------------------------------------------

class TestEdgeCases:
    def test_multiline_with_tool_on_first_line(self):
        raw = 'TOOL: CHECK file_exists path="x.txt"\nSome extra text'
        result = parse_arc_output(raw)
        # Should pick up the TOOL line
        assert isinstance(result, ToolCommand)

    def test_multiline_content_write(self):
        raw = 'TOOL: ACT write_file path="hello.html" content="<html>\n<body>Hello</body>\n</html>"'
        result = parse_arc_output(raw)
        assert isinstance(result, ToolCommand)
        assert result.action == "write_file"
        assert "<html>" in result.params.get("content", "")

    def test_unquoted_path(self):
        result = parse_arc_output('TOOL: CHECK file_exists path=test.txt')
        assert isinstance(result, ToolCommand)
        assert result.params["path"] == "test.txt"
