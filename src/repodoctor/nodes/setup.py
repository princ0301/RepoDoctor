from dataclasses import asdict

from repodoctor.deps import Deps
from repodoctor.state import RepoDoctorState
from repodoctor.tools.repo import head_sha, list_untracked
from repodoctor.tools.sandbox import ExecResult


def _why(r: ExecResult) -> str:
    return "timeout" if r.timed_out else f"exit {r.exit_code}"


def make_setup(deps: Deps):
    def setup(state: RepoDoctorState) -> dict:
        sb, s = deps.sandbox, deps.settings

        r = sb.install(deps.info)
        if not r.ok:
            return {"error": f"Dependency install failed ({_why(r)}):\n{r.output}"}
        sb.lock_network()

        result, baseline = sb.run_suite(deps.info, timeout=s.sandbox_timeout_s)
        if baseline is None:
            return {
                "error": f"The baseline test run produced no report ({_why(result)}):\n"
                f"{result.output}"
            }

        out: dict = {
            "baseline": asdict(baseline),
            "untracked_baseline": sorted(list_untracked(deps.workdir)),
            "base_sha": head_sha(deps.workdir),
            "attempts": 0,
            "history": [],
        }

        targets = list(state.get("target_tests") or [])
        if targets:  # user-supplied tests: they must fail on the unmodified code
            t_res, _ = sb.run_suite(deps.info, args=targets, timeout=s.sandbox_timeout_s)
            if t_res.ok:
                return {"error": "The target test(s) already pass on the unmodified code."}
            if t_res.timed_out or t_res.exit_code != 1:
                return {
                    "error": f"The target test(s) could not run ({_why(t_res)}). "
                    f"Check the test id:\n{t_res.output}"
                }
            out["target_failure"] = t_res.output
        return out

    return setup