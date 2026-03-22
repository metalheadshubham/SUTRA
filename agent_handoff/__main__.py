"""Run S.U.T.R.A CLI with ``python -m agent_handoff``."""

import sys

def main():
    # If called with "benchmark" subcommand, run benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        sys.argv.pop(1)  # Remove "benchmark" from args
        from agent_handoff.benchmark import main as bench_main
        bench_main()
    else:
        from agent_handoff.cli import main as cli_main
        cli_main()

if __name__ == "__main__":
    main()
