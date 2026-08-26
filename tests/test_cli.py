"""Command-line behaviour, including the exit codes the README documents."""

import pytest

import arxiv_lit_reviewer as cli


@pytest.fixture
def parser():
    return cli.build_parser()


class TestArgumentValidation:
    def test_target_papers_cannot_exceed_max_results(self, parser, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        args = parser.parse_args(["run", "--query", "q", "--max-results", "2",
                                  "--target-papers", "5"])
        with pytest.raises(SystemExit):
            cli.command_run(args, parser)

    def test_missing_credentials_is_rejected_before_any_work(self, parser):
        args = parser.parse_args(["run", "--query", "q"])
        with pytest.raises(SystemExit):
            cli.command_run(args, parser)

    def test_defaults_match_what_the_readme_documents(self, parser):
        args = parser.parse_args(["run", "--query", "q"])
        assert args.max_results == 30
        assert args.target_papers == 4
        assert args.retriever == "hybrid-rerank"
        assert args.top_k == 5


class TestExitCodes:
    def test_successful_run_returns_ok(self, parser, monkeypatch, tmp_path):
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setattr(
            "arxiv_reviewer.workflow.run_reviewer", lambda **_kwargs: {"status": "complete"}
        )
        args = parser.parse_args(["run", "--query", "q", "--data-dir", str(tmp_path)])
        assert cli.command_run(args, parser) == cli.EXIT_OK

    def test_crash_returns_failed_rather_than_a_traceback(self, parser, monkeypatch,
                                                          tmp_path, capsys):
        monkeypatch.setenv("GOOGLE_API_KEY", "x")

        def exploding(**_kwargs):
            raise RuntimeError("the vector store is unreachable")

        monkeypatch.setattr("arxiv_reviewer.workflow.run_reviewer", exploding)
        args = parser.parse_args(["run", "--query", "q", "--data-dir", str(tmp_path),
                                  "--thread-id", "crashed"])
        assert cli.command_run(args, parser) == cli.EXIT_FAILED

        output = capsys.readouterr().out
        assert "run failed" in output
        # The thread survives the crash, so the user must be told how to resume it.
        assert "crashed" in output

    def test_interrupt_returns_failed_and_names_the_thread(self, parser, monkeypatch,
                                                           tmp_path, capsys):
        monkeypatch.setenv("GOOGLE_API_KEY", "x")

        def interrupted(**_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("arxiv_reviewer.workflow.run_reviewer", interrupted)
        args = parser.parse_args(["run", "--query", "q", "--data-dir", str(tmp_path),
                                  "--thread-id", "paused"])
        assert cli.command_run(args, parser) == cli.EXIT_FAILED
        assert "paused" in capsys.readouterr().out

    def test_unknown_thread_returns_invalid(self, parser, tmp_path):
        args = parser.parse_args(["status", "--thread-id", "missing",
                                  "--data-dir", str(tmp_path)])
        assert cli.command_status(args, parser) == cli.EXIT_INVALID


class TestStatusWithoutCredentials:
    def test_status_needs_no_api_key(self, parser, tmp_path):
        # The README promises this: reading a thread must never require a key or a
        # network call, and the autouse fixtures here enforce both.
        args = parser.parse_args(["status", "--thread-id", "nope",
                                  "--data-dir", str(tmp_path)])
        assert cli.command_status(args, parser) == cli.EXIT_INVALID
