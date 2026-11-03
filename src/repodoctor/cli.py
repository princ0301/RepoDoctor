import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv

from repodoctor.config import Settings, get_settings
from repodoctor.demo import make_demo
from repodoctor.evals.build import CaseRejected, build_case, feature_signals, pair_problem
from repodoctor.evals.cases import (
    CaseSpec,
    case_id,
    load_cases,
    load_specs,
    save_cases,
    save_specs,
)
from repodoctor.evals.discover import discover
from repodoctor.evals.run import attempt_kind, load_results, run_eval, summarize
from repodoctor.graph import build_hello_graph
from repodoctor.models import ENDPOINTS, ModelConfigError, make_chat_model, probe, role_specs
from repodoctor.runner import (
    FixRequest,
    RunnerError,
    parse_issue_file,
    run_fix,
    run_fix_issue,
)
from repodoctor.tools.github import (
    GitHubError,
    fetch_issue,
    fetch_pr,
    fetch_pr_text,
    github_client,
)
from repodoctor.tools.issue_ref import (
    InvalidIssueUrl,
    IssueRef,
    RepoNotAllowed,
    check_allowed,
    parse_issue_url,
)
from repodoctor.tools.repo import (
    GitError,
    ProjectInfo,
    clone_issue_repo,
    detect_project,
    remove_dir,
)
from repodoctor.tools.sandbox import Sandbox, SandboxError, ensure_image, image_exists

# Loads .env into os.environ so LangSmith picks up LANGSMITH_* variables
load_dotenv()

app = typer.Typer(no_args_is_help=True, help="RepoDoctor: autonomous issue fixer")

RUN_ERRORS = (
    RunnerError, GitHubError, ModelConfigError, SandboxError, GitError, ValueError,
)


def _resolve_issue(url: str) -> IssueRef:
    try:
        ref = parse_issue_url(url)
        check_allowed(ref, get_settings().allowed_repo_list)
    except (InvalidIssueUrl, RepoNotAllowed) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    return ref


def _clone_and_detect(ref: IssueRef) -> tuple[Path, ProjectInfo]:
    dest = get_settings().work_dir.resolve() / f"{ref.owner}__{ref.repo}"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Cloning {ref.full_name} ...")
    try:
        clone_issue_repo(ref, dest)
    except GitError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    return dest, detect_project(dest)


def _print_result(final: dict, settings: Settings) -> None:
    typer.echo(f"Relevant files: {', '.join(final.get('relevant_files', [])) or '-'}")
    typer.echo(f"Hypothesis:     {final.get('hypothesis', '-')}")
    if final.get("repro_file"):
        typer.echo(
            f"Generated test: {final['target_tests'][0]} "
            f"(accepted on attempt {final.get('repro_attempts', 1)})"
        )
    typer.echo(f"Fix attempts:   {final.get('attempts', 0)}/{settings.max_attempts}")
    for h in final.get("history", []):
        first = (h.get("feedback") or "-").splitlines()[0]
        typer.echo(f"  attempt {h['attempt']}: {first}")

    if final.get("error"):
        typer.echo(f"\nERROR   {final['error']}")
        raise typer.Exit(1)
    diff = final.get("full_diff") or final.get("diff")
    if diff:
        typer.echo("\n" + diff)
    if final.get("success"):
        typer.echo("RESULT: SUCCESS (nothing was pushed; the diff above is the change)")
        return
    typer.echo(f"RESULT: FAILED\nLast feedback:\n{final.get('feedback', '')}")
    raise typer.Exit(1)


@app.command()
def hello(name: str = "world") -> None:
    """Run the placeholder graph (check that it shows up in LangSmith)."""
    result = build_hello_graph().invoke({"name": name}, config={"run_name": "hello"})
    typer.echo(result["greeting"])


@app.command()
def doctor() -> None:
    """Check that your environment is ready."""
    for tool in ("uv", "git", "rg", "docker"):
        typer.echo(f"{'OK     ' if shutil.which(tool) else 'MISSING'} {tool}")

    s = get_settings()
    required = {
        "OPENROUTER_API_KEY": s.openrouter_api_key,
        "GITHUB_TOKEN": s.github_token,
        "GITHUB_USERNAME": s.github_username,
        "ALLOWED_REPOS": s.allowed_repos,
    }
    for label, value in required.items():
        typer.echo(f"{'OK     ' if value else 'MISSING'} {label}")

    if s.nvidia_api_key:
        typer.echo("OK      NVIDIA_API_KEY")
    else:
        typer.echo("INFO    NVIDIA_API_KEY not set (NVIDIA models via OpenRouter only)")

    typer.echo(f"INFO    DRY_RUN={s.dry_run}")
    typer.echo(f"INFO    explore={s.llm_explore}  fix={s.llm_fix}")
    typer.echo(f"INFO    fallbacks={s.fallback_specs or '-'}")
    if not s.dry_run:
        typer.echo("WARN    DRY_RUN is off: the agent can push branches and open PRs")


@app.command("nvidia-models")
def nvidia_models(
    contains: str = typer.Option("nemotron", help="Only show model ids containing this text"),
) -> None:
    """List models available on NVIDIA's API with your key (to pick LLM_EXPLORE / LLM_FIX)."""
    key = get_settings().nvidia_api_key
    if not key:
        typer.echo("ERROR   NVIDIA_API_KEY is not set")
        raise typer.Exit(1)
    try:
        r = httpx.get(
            f"{ENDPOINTS['nvidia'][0]}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        typer.echo(f"ERROR   {type(e).__name__}: {e}")
        raise typer.Exit(1) from e
    ids = sorted(
        m["id"] for m in r.json().get("data", []) if contains.lower() in m["id"].lower()
    )
    typer.echo("\n".join(ids) or "(no matching models)")


@app.command("check-llm")
def check_llm() -> None:
    """Test every configured model (credentials, model id, tool calling, structured output)."""
    s = get_settings()
    specs: list[str] = []
    for role in ("explore", "fix"):
        for spec in role_specs(role, s):
            if spec not in specs:
                specs.append(spec)

    all_ok = True
    for spec in specs:
        try:
            model = make_chat_model(spec, s)
        except ModelConfigError as e:
            typer.echo(f"SKIP    {spec}: {e}")
            continue
        for check, problem in probe(model).items():
            if problem is None:
                typer.echo(f"OK      {spec} [{check}]")
            else:
                all_ok = False
                typer.echo(f"FAIL    {spec} [{check}]: {problem}")
    raise typer.Exit(0 if all_ok else 1)


@app.command("check-issue")
def check_issue(url: str) -> None:
    """Validate an issue URL against the allowlist."""
    ref = _resolve_issue(url)
    typer.echo(f"OK      {ref.full_name} #{ref.number}")


@app.command("inspect")
def inspect_repo(url: str) -> None:
    """Clone the repo behind an issue and detect how to test it."""
    ref = _resolve_issue(url)
    dest, info = _clone_and_detect(ref)
    typer.echo(f"Cloned to      {dest}")
    if not info.supported:
        typer.echo(f"UNSUPPORTED    {info.reason}")
        raise typer.Exit(2)
    typer.echo(f"Test command   {' '.join(info.test_command)}")
    typer.echo(f"Extras         {info.extras or '-'}")
    typer.echo(f"Dep groups     {info.dependency_groups or '-'}")
    typer.echo(f"Req files      {info.requirement_files or '-'}")
    typer.echo(f"Python         {info.requires_python or 'not specified'}")


@app.command()
def smoke(url: str) -> None:
    """Run the repo's own test suite in the sandbox and record the baseline."""
    ref = _resolve_issue(url)
    dest, info = _clone_and_detect(ref)
    if not info.supported:
        typer.echo(f"UNSUPPORTED    {info.reason}")
        raise typer.Exit(2)

    s = get_settings()
    try:
        if not image_exists():
            typer.echo("Building sandbox image (first run only, takes a few minutes) ...")
            ensure_image()
        with Sandbox(dest, memory=s.sandbox_memory, cpus=s.sandbox_cpus) as sb:
            typer.echo("Installing dependencies (network on) ...")
            r = sb.install(info)
            if not r.ok:
                typer.echo(r.output)
                typer.echo(
                    f"\nERROR   install failed (exit {r.exit_code}, "
                    f"{'timeout' if r.timed_out else 'error'})"
                )
                raise typer.Exit(3)
            sb.lock_network()
            typer.echo("Running tests (network off) ...")
            r, report = sb.run_suite(info, timeout=s.sandbox_timeout_s)
    except SandboxError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e

    if report is None:
        typer.echo(r.output)
        why = "timeout" if r.timed_out else f"exit {r.exit_code}"
        typer.echo(
            f"\nERROR   no test report produced ({why}); "
            "pytest may have crashed during collection"
        )
        raise typer.Exit(1)

    typer.echo(
        f"Baseline: {report.total} tests | {report.passed} passed, {report.failed} failed, "
        f"{report.errors} errors, {report.skipped} skipped | {r.duration_s:.0f}s"
    )
    if report.clean:
        typer.echo("Result: CLEAN baseline")
        return
    typer.echo(
        f"Result: {len(report.failing)} pre-existing failures (the agent will ignore these)"
    )
    for test_id, msg in list(report.failing.items())[:10]:
        typer.echo(f"  - {test_id}: {msg}")
    if len(report.failing) > 10:
        typer.echo(f"  ... and {len(report.failing) - 10} more")


@app.command("make-demo")
def make_demo_cmd() -> None:
    """Create tiny repos with one seeded bug for end-to-end runs."""
    root = get_settings().work_dir.resolve().parent / "demo"
    demo = make_demo(root)
    typer.echo(f"Created {demo.repo_dir}")
    typer.echo(f"Created {demo.blind_repo_dir}\n")
    typer.echo("1) You supply the failing test:")
    typer.echo(
        f'  uv run repodoctor fix --repo-dir "{demo.repo_dir}" '
        f'--issue-file "{demo.issue_file}" --test {demo.test_id}'
    )
    typer.echo("\n2) The agent writes the failing test itself:")
    typer.echo(
        f'  uv run repodoctor fix --repo-dir "{demo.blind_repo_dir}" '
        f'--issue-file "{demo.issue_file}"'
    )


@app.command()
def fix(
    repo_dir: Path = typer.Option(
        ...,
        "--repo-dir",
        exists=True,
        file_okay=False,
        help="Local git repo containing the bug (changes must be committed)",
    ),
    issue_file: Path = typer.Option(
        ...,
        "--issue-file",
        exists=True,
        dir_okay=False,
        help="Markdown file: first line is the title, the rest is the body",
    ),
    test: list[str] | None = typer.Option(
        None,
        "--test",
        help="pytest node id that must pass after the fix (repeatable). "
        "Omit it and the agent writes the failing test itself.",
    ),
) -> None:
    """Run the fix loop on a local repo."""
    settings = get_settings()
    try:
        title, body = parse_issue_file(issue_file.read_text(encoding="utf-8"))
        final = run_fix(FixRequest(repo_dir, title, body, list(test or [])), settings)
    except RUN_ERRORS as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    _print_result(final, settings)


@app.command("fix-issue")
def fix_issue(
    url: str,
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Closed issue: run on the code as it was before the merged fix",
    ),
    test: list[str] | None = typer.Option(
        None, "--test", help="Your own failing test id (default: the agent writes one)"
    ),
    fix_pr: int | None = typer.Option(
        None,
        "--fix-pr",
        help="With --replay: number of the merged PR that fixed the issue "
        "(use it if it is not detected automatically)",
    ),
) -> None:
    """Run the fix loop on a real GitHub issue (repo must be in ALLOWED_REPOS)."""
    ref = _resolve_issue(url)
    settings = get_settings()
    if not settings.github_token:
        typer.echo("WARN    GITHUB_TOKEN not set: GitHub allows only 60 API requests/hour")
    try:
        final, intake = run_fix_issue(
            ref, settings, replay=replay, targets=list(test or []), fix_pr=fix_pr,
            log=typer.echo,
        )
    except RUN_ERRORS as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    typer.echo("")
    try:
        _print_result(final, settings)
    finally:
        if intake.fixing_pr_url:
            typer.echo(f"\nCompare with the real fix: {intake.fixing_pr_url}")

@app.command("eval-discover")
def eval_discover(
    repo: str,
    limit: int = typer.Option(8, help="Stop after this many candidates"),
    max_prs: int = typer.Option(200, help="How many recent merged PRs to scan"),
    out: Path = typer.Option(Path("evals/specs.json"), "--out"),
) -> None:
    """Find (issue, fixing PR) candidates in a repo and add them to the specs file."""
    settings = get_settings()
    owner, _, name = repo.partition("/")
    try:
        check_allowed(IssueRef(owner, name, 0), settings.allowed_repo_list)
        with github_client(settings.github_token) as client:
            found = discover(client, repo, limit=limit, max_prs=max_prs)
    except (RepoNotAllowed, GitHubError) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    specs = load_specs(out) if out.exists() else []
    known = {(s.repo, s.issue) for s in specs}
    added = [s for s in found if (s.repo, s.issue) not in known]
    save_specs(out, specs + added)
    for s in added:
        typer.echo(f"  + {s.repo}#{s.issue}  (fixed by PR #{s.fix_pr})")
    typer.echo(f"{len(added)} new candidate(s); {len(specs) + len(added)} in {out}")


@app.command("eval-build")
def eval_build(
    specs_file: Path = typer.Option(Path("evals/specs.json"), "--specs"),
    out: Path = typer.Option(Path("evals/cases.json"), "--out"),
    compare: Path = typer.Option(
        Path("evals/cases.json"), "--compare",
        help="Earlier cases file: changed ground truth is reported against it",
    ),
    retry_rejected: bool = typer.Option(
        False, "--retry-rejected", help="Try again the specs rejected in earlier builds"
    ),
    dev: bool = typer.Option(
        False, "--dev", help="Add cases that are new (not in --compare) to evals/dev_ids.txt"
    ),
) -> None:
    """Validate specs and compute ground truth (which tests the fix must turn green)."""
    if not specs_file.exists():
        typer.echo(f"ERROR   {specs_file} does not exist")
        raise typer.Exit(1)
    settings = get_settings()
    known = {c.id: c for c in load_cases(compare)} if compare.exists() else {}
    cases = load_cases(out) if out.exists() else []
    done = {c.id for c in cases}
    rejected_file = out.with_suffix(".rejected.json")
    rejected: dict[str, str] = (
        json.loads(rejected_file.read_text(encoding="utf-8")) if rejected_file.exists() else {}
    )
    with github_client(settings.github_token) as client:
        for spec in load_specs(specs_file):
            cid = case_id(spec.repo, spec.issue)
            if cid in done:
                typer.echo(f"Skipping {cid} (already built)")
                continue
            if cid in rejected and not retry_rejected:
                typer.echo(f"Skipping {cid} (rejected before: {rejected[cid][:80]})")
                continue
            typer.echo(f"Building {spec.repo}#{spec.issue} (PR #{spec.fix_pr}) ...")
            try:
                case = build_case(spec, settings, client, typer.echo)
            except (CaseRejected, RepoNotAllowed) as e:
                typer.echo(f"  REJECTED  {e}")
                rejected[cid] = str(e)[:200]
                rejected_file.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
                continue
            except (GitHubError, GitError, SandboxError) as e:
                typer.echo(f"  ERROR     {e} (not recorded as rejected; rerun to retry)")
                continue
            cases.append(case)
            save_cases(out, cases)
            typer.echo(
                f"  OK        {len(case.fail_to_pass)} fail-to-pass, "
                f"{len(case.pass_to_pass)} pass-to-pass tests"
            )
            old = known.get(case.id)
            if old is None and dev:
                text = DEV_FILE.read_text(encoding="utf-8") if DEV_FILE.exists() else ""
                sep = "" if (not text or text.endswith("\n")) else "\n"
                DEV_FILE.write_text(text + sep + case.id + "\n", encoding="utf-8")
                typer.echo("            added to the dev set")
            elif old is not None and old.fail_to_pass != case.fail_to_pass:
                lost = sorted(set(old.fail_to_pass) - set(case.fail_to_pass))
                typer.echo(
                    f"  CHANGED   fail-to-pass {len(old.fail_to_pass)} -> "
                    f"{len(case.fail_to_pass)}; no longer required, e.g. {lost[:2]}"
                )
    typer.echo(f"{len(cases)} case(s) in {out}")


EXCLUDED_FILE = Path("evals/excluded.txt")
DEV_FILE = Path("evals/dev_ids.txt")


def _excluded_ids(path: Path = EXCLUDED_FILE) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@app.command("eval-inspect")
def eval_inspect(
    case_ids: list[str],
    cases_file: Path = typer.Option(Path("evals/cases.json"), "--cases"),
) -> None:
    """Show what a case asks for (issue, PR text, tests) to judge whether it is fair."""
    cases = {c.id: c for c in load_cases(cases_file)}
    with github_client(get_settings().github_token) as client:
        for cid in case_ids:
            case = cases.get(cid)
            if case is None:
                typer.echo(f"ERROR   unknown case {cid}")
                continue
            try:
                issue = fetch_issue(client, case.ref)
                pr_title, pr_body = fetch_pr_text(client, case.ref, case.fix_pr)
            except GitHubError as e:
                typer.echo(f"ERROR   {cid}: {e}")
                continue
            typer.echo("=" * 78)
            typer.echo(f"{cid}   issue #{case.issue}   fixed by PR #{case.fix_pr}")
            typer.echo(f"\nISSUE: {issue.title}\n{issue.body.strip()[:1500]}")
            typer.echo(f"Labels: {', '.join(issue.labels) or '-'}")
            typer.echo(f"\nPR: {pr_title}\n{pr_body.strip()[:1000]}")
            typer.echo(f"\nMaintainers changed: {', '.join(case.src_files)}")
            typer.echo(f"Fail-to-pass tests ({len(case.fail_to_pass)}):")
            for t in case.fail_to_pass[:12]:
                typer.echo(f"  {t}")
            if len(case.fail_to_pass) > 12:
                typer.echo(f"  ... and {len(case.fail_to_pass) - 12} more")


@app.command("eval-recheck")
def eval_recheck(
    cases_file: Path = typer.Option(Path("evals/cases.json"), "--cases"),
) -> None:
    """Re-check the issue/PR pairing of built cases (GitHub API only, no sandbox)."""
    cases = load_cases(cases_file)
    flagged = 0
    with github_client(get_settings().github_token) as client:
        for c in cases:
            try:
                issue = fetch_issue(client, c.ref)
                pr = fetch_pr(client, c.ref, c.fix_pr)
            except GitHubError as e:
                typer.echo(f"ERROR     {c.id}: {e}")
                continue
            problem = pair_problem(issue.closed_at, pr.created_at)
            if problem:
                flagged += 1
                typer.echo(f"SUSPECT   {c.id}: {problem}")
            signals = feature_signals(issue.labels, pr.title)
            if signals:
                typer.echo(
                    f"REVIEW    {c.id}: looks like a feature request ({'; '.join(signals)})"
                )
    typer.echo(f"{flagged} suspect pair(s) among {len(cases)} case(s)")


def _dev_ids(path: Path = Path("evals/dev_ids.txt")) -> set[str]:
    if not path.exists():
        typer.echo(f"ERROR   {path} does not exist")
        raise typer.Exit(1)
    return {
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


@app.command("eval-run")
def eval_run(
    cases_file: Path = typer.Option(Path("evals/cases.json"), "--cases"),
    include_excluded: bool = typer.Option(
        False, "--include-excluded", help="Also run cases listed in evals/excluded.txt"
    ),
    trials: int = typer.Option(1, help="Runs per case (use 3+ for stable numbers)"),
    limit: int = typer.Option(0, help="Only the first N cases after filtering (0 = all)"),
    per_repo: int = typer.Option(0, "--per-repo", help="At most N cases from each repo"),
    split: str | None = typer.Option(
        None, "--split", help="'dev' or 'heldout' (dev ids are listed in evals/dev_ids.txt)"
    ),
    only: list[str] | None = typer.Option(None, "--only", help="Case id(s) to run"),
    label: str = typer.Option("run", help="Names this run in results and LangSmith"),
    resume: Path | None = typer.Option(
        None, "--resume", help="results.jsonl of an interrupted run: continue it"
    ),
    rerun_failed: Path | None = typer.Option(
        None, "--rerun-failed",
        help="results.jsonl of an earlier run: only the cases with an unresolved run in it",
    ),
) -> None:
    """Run the agent on every case and grade it with the maintainers' tests."""
    if not cases_file.exists():
        typer.echo(f"ERROR   {cases_file} does not exist (run eval-build first)")
        raise typer.Exit(1)
    if split not in (None, "dev", "heldout"):
        typer.echo("ERROR   --split must be 'dev' or 'heldout'")
        raise typer.Exit(1)

    settings = get_settings()
    cases = load_cases(cases_file)
    if not include_excluded:
        skip = _excluded_ids()
        cases = [c for c in cases if c.id not in skip]
    if rerun_failed:
        unresolved = {r.case_id for r in load_results(rerun_failed) if not r.resolved}
        cases = [c for c in cases if c.id in unresolved]
    if only:
        cases = [c for c in cases if c.id in only]
    if split:
        dev = _dev_ids()
        cases = [c for c in cases if (c.id in dev) == (split == "dev")]
    if per_repo:
        taken: dict[str, int] = {}
        picked = []
        for c in cases:
            if taken.get(c.repo, 0) < per_repo:
                picked.append(c)
                taken[c.repo] = taken.get(c.repo, 0) + 1
        cases = picked
    if limit:
        cases = cases[:limit]
    try:
        for c in cases:
            check_allowed(c.ref, settings.allowed_repo_list)
    except RepoNotAllowed as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    if not cases:
        typer.echo("ERROR   no cases selected")
        raise typer.Exit(1)

    if resume:
        results_file = resume
        done = {
            (r.case_id, r.trial) for r in load_results(resume) if r.outcome != "model_error"
        }
    else:
        out_dir = Path(".repodoctor/evals") / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(
            json.dumps({
                "label": label, "trials": trials, "split": split,
                "cases": [c.id for c in cases],
                "llm_explore": settings.llm_explore, "llm_fix": settings.llm_fix,
                "llm_fallbacks": settings.fallback_specs,
                "max_attempts": settings.max_attempts,
                "repro_max_attempts": settings.repro_max_attempts,
            }, indent=2),
            encoding="utf-8",
        )
        results_file = out_dir / "results.jsonl"
        done = set()
    typer.echo(f"{len(cases)} case(s) x {trials} trial(s). Results: {results_file}\n")
    run_eval(
        cases, settings, trials=trials, label=label, results_file=results_file,
        done=done, log=typer.echo,
    )
    typer.echo("\n" + summarize(load_results(results_file)))


@app.command("eval-report")
def eval_report(
    results_file: Path,
    show: str | None = typer.Option(
        None, "--show", help="Case id: print its diff, failure details and the real fix's PR"
    ),
) -> None:
    """Print the summary of a finished (or interrupted) eval run."""
    results = load_results(results_file)
    if not show:
        typer.echo(summarize(results))
        return
    for r in (r for r in results if r.case_id == show):
        typer.echo(f"== {r.case_id} trial {r.trial}: {r.outcome} ==")
        if r.detail:
            typer.echo(r.detail)
        typer.echo(f"Maintainers' fix: {r.gold_pr}\n")
        typer.echo(f"Explorer chose: {', '.join(r.explored_files or r.relevant_files) or '-'}")
        if r.explored_files and r.relevant_files != r.explored_files:
            typer.echo(f"Fixer's context at the end: {', '.join(r.relevant_files)}")
        typer.echo(f"Hypothesis: {r.hypothesis or '-'}\n")
        typer.echo(f"Time by step: {r.timings or '-'}")
        if r.context_omitted:
            typer.echo(f"Not shown to the fixer (budget): {', '.join(r.context_omitted)}")
        for h in r.history:
            typer.echo(f"--- attempt {h['attempt']} ---")
            typer.echo(f"reasoning: {h['reasoning']}")
            for e in h["edits"]:
                typer.echo(f"edit {e['path']}:\n  old: {e['old']!r}\n  new: {e['new']!r}")
            typer.echo(f"feedback: {h['feedback']}")
        typer.echo("\n" + (r.diff or "(no diff)"))


@app.command("eval-failures")
def eval_failures(
    results_files: list[Path],
    include_excluded: bool = typer.Option(False, "--include-excluded"),
) -> None:
    """Why runs failed: how each fix attempt ended, per unresolved run and overall."""
    skip = set() if include_excluded else _excluded_ids()
    results = [r for f in results_files for r in load_results(f) if r.case_id not in skip]
    unresolved_kinds: Counter = Counter()
    resolved_kinds: Counter = Counter()
    rows: list[str] = []
    for r in results:
        kinds = [attempt_kind(h.get("feedback") or "") for h in r.history]
        failed = [k for k in kinds if k != "passed"]
        (resolved_kinds if r.resolved else unresolved_kinds).update(failed)
        if not r.resolved:
            path = " > ".join(kinds) or r.detail[:70] or "-"
            rows.append(f"{r.case_id:<38} t{r.trial} {r.outcome:<14} {path}")
    names = sorted(
        set(unresolved_kinds) | set(resolved_kinds),
        key=lambda k: -(unresolved_kinds[k] + resolved_kinds[k]),
    )
    typer.echo(f"{'a fix attempt ended with':<26} {'unresolved runs':>16} {'resolved runs':>14}")
    for k in names:
        typer.echo(f"{k:<26} {unresolved_kinds[k]:>16} {resolved_kinds[k]:>14}")
    typer.echo("\n" + "\n".join(rows))

if __name__ == "__main__":
    app()
