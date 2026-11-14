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
    case_id,
    is_test_path,
    load_cases,
    load_specs,
    parse_case_flags,
    removed_line_count,
    save_cases,
    save_specs,
)
from repodoctor.evals.discover import discover
from repodoctor.evals.gold import gold_test_source
from repodoctor.evals.run import (
    attempt_kind,
    compare,
    is_infrastructure_failure,
    load_results,
    run_eval,
    summarize,
)
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
    fetch_file_text,
    fetch_issue,
    fetch_pr,
    fetch_pr_text,
    github_client,
    list_pr_files,
)
from repodoctor.tools.issue_ref import (
    InvalidIssueUrl,
    IssueRef,
    RepoNotAllowed,
    check_allowed,
    parse_issue_url,
)
from repodoctor.tools.pr_package import (
    PackageError,
    approve,
    blocking_findings,
    load_package,
    prepare_package,
    render_review,
)
from repodoctor.tools.repackage import repackage as repackage_package
from repodoctor.tools.repo import (
    GitError,
    ProjectInfo,
    clone_issue_repo,
    detect_project,
    remove_dir,
    run_git,
)
from repodoctor.tools.repo_config import ignored_paths, with_ignores
from repodoctor.tools.sandbox import Sandbox, SandboxError, ensure_image, image_exists
from repodoctor.tools.triage import TriageResult, run_triage, scan_repo

# Loads .env into os.environ so LangSmith picks up LANGSMITH_* variables
load_dotenv()

app = typer.Typer(no_args_is_help=True, help="RepoDoctor: autonomous issue fixer")

RUN_ERRORS = (
    RunnerError, GitHubError, ModelConfigError, SandboxError, GitError, ValueError,
)


def _code_version() -> str:
    try:
        sha = run_git("rev-parse", "--short", "HEAD").strip()
        dirty = bool(run_git("status", "--porcelain").strip())
    except GitError:
        return "unknown"
    return sha + ("+uncommitted changes" if dirty else "")


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
    return dest, with_ignores(detect_project(dest), ignored_paths(ref.full_name))


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
    if final.get("edited_tests"):
        typer.echo(
            "NOTE    this change modifies existing test files, review them: "
            + ", ".join(final["edited_tests"])
        )
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
def smoke(
    url: str,
    durations: int = typer.Option(0, "--durations", help="Also print the N slowest tests"),
    per_file: bool = typer.Option(
        False, "--per-file", help="Run every test file separately and print its time"
    ),
    per_file_timeout: int = typer.Option(90, "--per-file-timeout"),
) -> None:
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
            if per_file:
                for path, secs, status, last in sb.time_test_files(info, per_file_timeout):
                    typer.echo(f"{secs:6.0f}s  {status:<8} {path}")
                    if last:
                        typer.echo(f"          stuck at: {last[:150]}")
                return
            extra = ["--durations", str(durations)] if durations else None
            r, report = sb.run_suite(info, args=extra, timeout=s.sandbox_timeout_s)
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
    if durations:
        typer.echo(r.output)
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
        final = run_fix(
            FixRequest(repo_dir, title, body, list(test or [])), settings, progress=typer.echo
        )
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
    prepare_pr: bool = typer.Option(
        False, "--prepare-pr",
        help="After a successful run, write a pull-request package and review it (no push)",
    ),
    ignore_triage: bool = typer.Option(
        False, "--ignore-triage",
        help="Run the agent even if triage says skip (preparing a PR still needs a pass)",
    ),
) -> None:
    """Run the fix loop on a real GitHub issue (repo must be in ALLOWED_REPOS)."""
    ref = _resolve_issue(url)
    settings = get_settings()
    if prepare_pr and replay:
        typer.echo(
            "ERROR   --prepare-pr cannot be combined with --replay (replays are for evaluation)"
        )
        raise typer.Exit(1)
    if not settings.github_token:
        typer.echo("WARN    GITHUB_TOKEN not set: GitHub allows only 60 API requests/hour")
    if not replay:
        result = _run_triage(ref, settings)
        _print_triage(result)
        if result.report.verdict == "skip" and not ignore_triage:
            typer.echo(
                "\nTriage says skip: nothing was run. --ignore-triage runs the agent anyway."
            )
            raise typer.Exit(1)
        typer.echo("")
    try:
        final, intake = run_fix_issue(
            ref, settings, replay=replay, targets=list(test or []), fix_pr=fix_pr,
            log=typer.echo, progress=typer.echo,
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

    if prepare_pr:
        fresh = _run_triage(ref, settings)
        if fresh.report.verdict == "skip":
            _print_triage(fresh)
            typer.echo("ERROR   triage says skip now: no package prepared")
            raise typer.Exit(1)
        try:
            pkg = prepare_package(
                ref=ref, issue=fresh.issue, final=final, workdir=intake.workdir,
                settings=settings, triage=fresh.report, repo_info=fresh.repo,
                policy_hits=fresh.policy_hits,
                ai_threads=fresh.ai_threads,
            )
        except (PackageError, GitError) as e:
            typer.echo(f"ERROR   {e}")
            raise typer.Exit(1) from e
        typer.echo(f"\nPackage written to {pkg}\n")
        _review_flow(pkg, settings)

@app.command("eval-discover")
def eval_discover(
    repo: str,
    limit: int = typer.Option(8, help="Stop after this many NEW candidates"),
    max_prs: int = typer.Option(200, help="How many recent merged PRs to scan"),
    out: Path = typer.Option(Path("evals/specs.json"), "--out"),
) -> None:
    """Find new (issue, fixing PR) candidates in a repo and add them to the specs file."""
    settings = get_settings()
    owner, _, name = repo.partition("/")
    specs = load_specs(out) if out.exists() else []
    known = {(s.repo, s.issue) for s in specs}
    try:
        check_allowed(IssueRef(owner, name, 0), settings.allowed_repo_list)
        with github_client(settings.github_token) as client:
            found = discover(client, repo, limit=limit, max_prs=max_prs, exclude=known)
    except (RepoNotAllowed, GitHubError) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    save_specs(out, specs + found)
    for s in found:
        typer.echo(f"  + {s.repo}#{s.issue}  (fixed by PR #{s.fix_pr})")
    typer.echo(f"{len(found)} new candidate(s); {len(specs) + len(found)} in {out}")


@app.command("eval-build")
def eval_build(
    specs_file: Path = typer.Option(Path("evals/specs.json"), "--specs"),
    repo: list[str] | None = typer.Option(
        None, "--repo", help="Only build specs of this repo (repeatable)"
    ),
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
    refresh_missing: bool = typer.Option(
        False, "--refresh-missing",
        help="Rebuild held-out cases built before 'which required tests are new' was recorded",
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
    if refresh_missing:
        keep = _dev_ids() | _excluded_ids()
        stale = {c.id for c in cases if c.new_pass_to_pass is None and c.id not in keep}
        cases = [c for c in cases if c.id not in stale]
        done = {c.id for c in cases}
        typer.echo(
            f"Rebuilding {len(stale)} held-out case(s) to record which required tests are new"
        )
    rejected_file = out.with_suffix(".rejected.json")
    rejected: dict[str, str] = (
        json.loads(rejected_file.read_text(encoding="utf-8")) if rejected_file.exists() else {}
    )
    with github_client(settings.github_token) as client:
        for spec in load_specs(specs_file):
            if repo and spec.repo not in repo:
                continue
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


FLAGS_FILE = Path("evals/case_flags.txt")
DEV_FILE = Path("evals/dev_ids.txt")


def _case_flags() -> dict[str, str]:
    if not FLAGS_FILE.exists():
        return {}
    return parse_case_flags(FLAGS_FILE.read_text(encoding="utf-8"))


def _excluded_ids() -> set[str]:
    return {cid for cid, tag in _case_flags().items() if tag in ("spec", "infra")}


def _scope_ids() -> set[str]:
    return {cid for cid, tag in _case_flags().items() if tag == "scope"}


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
    """Re-check built cases (GitHub API only): pairing, feature signals, edited tests."""
    cases = load_cases(cases_file)
    flagged = changed_tests = 0
    with github_client(get_settings().github_token) as client:
        for c in cases:
            try:
                issue = fetch_issue(client, c.ref)
                pr = fetch_pr(client, c.ref, c.fix_pr)
                files = list_pr_files(client, c.ref, c.fix_pr)
            except GitHubError as e:
                typer.echo(f"ERROR     {c.id}: {e}")
                continue
            problem = pair_problem(issue.closed_at, pr.created_at)
            if problem:
                flagged += 1
                typer.echo(f"SUSPECT    {c.id}: {problem}")
            signals = feature_signals(issue.labels, pr.title)
            if signals:
                typer.echo(
                    f"REVIEW     {c.id}: looks like a feature request ({'; '.join(signals)})"
                )
            edited = {
                f.path: removed_line_count(f.patch)
                for f in files
                if f.path.endswith(".py") and is_test_path(f.path)
            }
            edited = {p: n for p, n in edited.items() if n}
            if edited:
                changed_tests += 1
                detail = ", ".join(f"{p}: {n}" for p, n in edited.items())
                typer.echo(f"TESTCHANGE {c.id}: the PR edits existing test lines ({detail})")
            if c.new_pass_to_pass:
                names = ", ".join(t.partition("::")[2] for t in c.new_pass_to_pass[:3])
                typer.echo(
                    f"NEWP2P     {c.id}: {len(c.new_pass_to_pass)} required test(s) are "
                    f"absent from the base repository ({names})"
                )
    typer.echo(
        f"{flagged} suspect pair(s); {changed_tests} case(s) edit existing tests; "
        f"{len(cases)} checked"
    )


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
    list_only: bool = typer.Option(
        False, "--list", help="Only print the selected cases and the run count; run nothing"
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
    if list_only:
        flags = _case_flags()
        for c in cases:
            tag = f"  [{flags[c.id]}]" if c.id in flags else ""
            typer.echo(f"{c.id}{tag}")
        typer.echo(f"{len(cases)} case(s) x {trials} trial(s) = {len(cases) * trials} runs")
        return

    if resume:
        results_file = resume
        done = {
            (r.case_id, r.trial) for r in load_results(resume) if not is_infrastructure_failure(r)
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
                "allow_test_updates": settings.allow_test_updates,
                "fix_context_chars": settings.fix_context_chars,
                "fix_context_rounds": settings.fix_context_rounds,
                "run_timeout_s": settings.run_timeout_s,
                "repodoctor_commit": _code_version(),
                "repo_overrides": (
                    json.loads(Path("evals/repo_overrides.json").read_text(encoding="utf-8"))
                    if Path("evals/repo_overrides.json").exists() else {}
                ),
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
    typer.echo("\n" + summarize(load_results(results_file), scope_ids=_scope_ids()))


@app.command("eval-compare")
def eval_compare(
    a: list[Path] = typer.Option(..., "--a", help="results.jsonl file(s) of arm A"),
    b: list[Path] = typer.Option(..., "--b", help="results.jsonl file(s) of arm B"),
) -> None:
    """Side-by-side comparison of two sets of runs on the same cases."""
    typer.echo(compare(
        [r for f in a for r in load_results(f)], [r for f in b for r in load_results(f)]
    ))


@app.command("eval-gold")
def eval_gold(
    cid: str = typer.Argument(..., metavar="CASE_ID"),
    cases_file: Path = typer.Option(Path("evals/cases.json"), "--cases"),
    limit: int = typer.Option(6, help="Maximum number of tests to print"),
    test_id: list[str] | None = typer.Option(
        None, "--test-id", help="Print these tests (ids as shown in failure reports)"
    ),
    new_p2p: bool = typer.Option(
        False, "--new-p2p",
        help="Print the required tests the PR itself added: the agent cannot see them",
    ),
) -> None:
    """Print the source of the maintainers' tests (what the grader checks)."""
    case = next((c for c in load_cases(cases_file) if c.id == cid), None)
    if case is None:
        typer.echo(f"ERROR   unknown case {cid}")
        raise typer.Exit(1)
    if test_id:
        ids = list(test_id)
    elif new_p2p:
        if case.new_pass_to_pass is None:
            typer.echo("ERROR   not recorded for this case: run eval-build --refresh-missing")
            raise typer.Exit(1)
        ids = case.new_pass_to_pass
    else:
        ids = case.fail_to_pass
    try:
        with github_client(get_settings().github_token) as client:
            texts = {
                p: fetch_file_text(client, case.ref, p, case.head_sha) for p in case.run_targets
            }
    except GitHubError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    shown: set[tuple[str, str]] = set()
    for tid in ids:
        found = gold_test_source(tid, texts)
        if found is None or found in shown:
            continue
        shown.add(found)
        if len(shown) > limit:
            typer.echo(f"... more tests not shown (--limit {limit})")
            break
        typer.echo(f"--- {found[0]}  ({tid}) ---\n{found[1]}\n")


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
        typer.echo(summarize(results, scope_ids=_scope_ids()))
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
        if r.outcome == "explore_failed":
            typer.echo("Explorer tool calls:\n  " + "\n  ".join(r.explore_trace or ["-"]))
        typer.echo(f"Time by step: {r.timings or '-'}")
        typer.echo(f"Slowest model calls: {r.slowest_calls or '-'}")
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


def _run_triage(ref: IssueRef, settings: Settings) -> TriageResult:
    try:
        with github_client(settings.github_token) as client:
            return run_triage(client, ref, me=settings.github_username)
    except GitHubError as e:
        typer.echo(f"ERROR   triage failed: {e}")
        raise typer.Exit(1) from e


def _print_triage(result: TriageResult) -> None:
    typer.echo(f"TRIAGE: {result.report.verdict.upper()}")
    for f in result.report.findings:
        typer.echo(f"  [{f.level:<4}] {f.message}")
    for hit in result.policy_hits[:5]:
        typer.echo(f"  policy: {hit}")
    for t in result.ai_threads[:3]:
        typer.echo(f"  ai-thread: {t}")


def _review_flow(pkg: Path, settings: Settings) -> None:
    meta = load_package(pkg)
    patch = (pkg / "change.patch").read_text(encoding="utf-8")
    pr_text = (pkg / "pr.md").read_text(encoding="utf-8")
    typer.echo(render_review(meta, patch, pr_text))
    if blocking_findings(meta):
        typer.echo("\nThis package cannot be approved: see the [skip] findings above.")
        raise typer.Exit(1)
    typer.echo(
        f"\nEdit {pkg / 'pr.md'} now if you want to change the text. Approval records a hash "
        "of the patch and the text, and any later edit invalidates it."
    )
    answer = typer.prompt(
        f"Type the repository name ({meta['repo']}) to approve, or press Enter to stop",
        default="", show_default=False,
    )
    if answer.strip() != meta["repo"]:
        typer.echo("Not approved.")
        return
    acknowledged = True
    if meta["policy_hits"] or meta.get("ai_threads"):
        reply = typer.prompt(
            "The project's docs or recent threads mention AI or automated contributions "
            "(see above). Type 'read' to confirm you read them",
            default="", show_default=False,
        )
        acknowledged = reply.strip().lower() == "read"
        if not acknowledged:
            typer.echo("Not approved.")
            return
    try:
        approve(pkg, policy_acknowledged=acknowledged)
    except PackageError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    typer.echo(f"Approved: {pkg}")
    typer.echo("There is no submit command yet, so nothing was pushed.")


@app.command()
def triage(url: str) -> None:
    """Check whether an issue is a reasonable target (GitHub API only, no model calls)."""
    result = _run_triage(_resolve_issue(url), get_settings())
    _print_triage(result)
    raise typer.Exit(1 if result.report.verdict == "skip" else 0)


@app.command()
def review(
    package: Path = typer.Argument(..., exists=True, file_okay=False, help="A package folder"),
) -> None:
    """Review a prepared pull-request package and approve it (nothing is pushed)."""
    _review_flow(package, get_settings())


@app.command("repackage")
def repackage_cmd(
    package: Path = typer.Argument(..., exists=True, file_okay=False, help="A package folder"),
    test: str | None = typer.Option(
        None, "--test", help="The new test id, if you moved or renamed the test"
    ),
) -> None:
    """After editing files in the package's clone: rebuild the commit and patch and re-verify."""
    settings = get_settings()
    typer.echo("Re-verifying in the sandbox (install, then three test runs) ...")
    try:
        meta = repackage_package(package, settings, test_id=test)
    except (PackageError, GitError, SandboxError) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    v = meta["verification"]
    typer.echo(f"  the test fails without the source change: {v['test_fails_without_fix']}")
    typer.echo(f"  the test passes with it:                  {v['test_passes_with_fix']}")
    typer.echo(f"  tests that newly fail:                    {len(v['new_failures'])}")
    typer.echo("Any earlier approval was cleared.\n")
    _review_flow(package, settings)


@app.command("triage-scan")
def triage_scan(
    repo: str,
    limit: int = typer.Option(10, help="Show at most this many candidates"),
    scan: int = typer.Option(30, help="How many recent open, unassigned issues to examine"),
    show_skipped: bool = typer.Option(False, "--show-skipped", help="Also list skipped issues"),
) -> None:
    """List a repo's open issues with triage verdicts (GitHub API only, no model calls)."""
    settings = get_settings()
    owner, _, name = repo.partition("/")
    try:
        check_allowed(IssueRef(owner, name, 0), settings.allowed_repo_list)
        with github_client(settings.github_token) as client:
            result = scan_repo(client, repo, me=settings.github_username, scan=scan)
    except (RepoNotAllowed, GitHubError) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e

    typer.echo(f"{repo} (default branch {result.repo.default_branch}): "
               f"repository checks {result.repo_report.verdict.upper()}")
    for t in result.ai_threads[:3]:
        typer.echo(f"  ai-thread: {t}")
    for f in result.repo_report.findings:
        typer.echo(f"  [{f.level:<4}] {f.message}")
    candidates = [r for r in result.rows if r.verdict != "skip"][:limit]
    skipped = [r for r in result.rows if r.verdict == "skip"]
    typer.echo("")
    for r in candidates:
        labels = f"  ({', '.join(r.labels)})" if r.labels else ""
        typer.echo(f"{r.verdict:<5} #{r.number}  {r.title[:70]}{labels}")
        for note in r.notes:
            typer.echo(f"        - {note}")
    if not candidates:
        typer.echo("No candidates among the issues examined.")
    typer.echo(f"\n{len(candidates)} candidate(s); {len(skipped)} skipped.")
    if show_skipped:
        for r in skipped:
            typer.echo(f"skip  #{r.number}  {r.title[:70]}")
            for note in r.notes:
                typer.echo(f"        - {note}")
    if candidates:
        typer.echo(
            "\nNext: uv run repodoctor fix-issue "
            f"https://github.com/{repo}/issues/{candidates[0].number} --prepare-pr"
        )


if __name__ == "__main__":
    app()
