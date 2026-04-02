"""Run S.U.T.R.A CLI with ``python -m agent_handoff``."""

import sys

def main():
    # If called with "benchmark" subcommand, run benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        sys.argv.pop(1)  # Remove "benchmark" from args
        from agent_handoff.benchmark import main as bench_main
        bench_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "agent":
        # Direct agent mode: sutra agent "create index.html"
        sys.argv.pop(1)
        task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
        if not task:
            print("Usage: sutra agent <task>")
            sys.exit(1)
        from agent_handoff.cli import (
            _enable_ansi_windows, _ensure_dirs, _header,
            detect_models, pick_model_council, run_agent_interactive,
        )
        import ollama as _ollama
        _enable_ansi_windows()
        _ensure_dirs()
        _header()
        models = detect_models()
        if not models:
            print("Cannot connect to Ollama. Run: ollama serve")
            sys.exit(1)
        model = pick_model_council(models, "small")
        client = _ollama.Client()
        import os
        run_agent_interactive(client=client, model=model, task=task, cwd=os.getcwd())
    else:
        from agent_handoff.cli import main as cli_main
        cli_main()

if __name__ == "__main__":
    main()
