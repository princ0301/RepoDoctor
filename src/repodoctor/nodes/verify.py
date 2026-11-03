from repodoctor.deps import Deps
from repodoctor.state import RepoDoctorState
from repodoctor.tools.testreport import SuiteReport, new_failures


def make_verify(deps: Deps):
    def verify(state: RepoDoctorState) -> dict:
        s = deps.settings
        history = list(state.get("history", []))

        def finish(success: bool, feedback: str, regressed: list[str] | None = None) -> dict:
            if history:
                history[-1] = {**history[-1], "feedback": feedback}
            return {
                "success": success, "feedback": feedback, "history": history,
                "regressed_tests": regressed or [],
            }

        if state.get("patch_errors"):
            return finish(
                False, "Your edits could not be applied:\n" + "\n".join(state["patch_errors"])
            )

        # 1. Cheap check first: the target tests.
        res, _ = deps.sandbox.run_suite(
            deps.info, args=list(state["target_tests"]), timeout=s.sandbox_timeout_s
        )
        if not res.ok:
            label = "timed out" if res.timed_out else "still fail"
            return finish(
                False, f"The target test(s) {label} after your edits. Output:\n{res.output}"
            )

        # 2. Full suite, compared against the baseline (pre-existing failures are ignored).
        res, report = deps.sandbox.run_suite(deps.info, timeout=s.sandbox_timeout_s)
        if report is None:
            return finish(False, f"The full test run produced no report:\n{res.output}")
        regressions = new_failures(SuiteReport(**state["baseline"]), report)
        if regressions:
            lines = [f"- {k}: {v}" for k, v in list(regressions.items())[:10]]
            return finish(
                False, "Your edits broke tests that passed before:\n" + "\n".join(lines),
                list(regressions),
            )

        return finish(True, "The target test(s) pass and no tests regressed.")

    return verify
