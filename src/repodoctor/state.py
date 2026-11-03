from typing import TypedDict


class RepoDoctorState(TypedDict, total=False):
    # input
    issue_title: str
    issue_body: str
    target_tests: list[str]      # pytest node ids that must pass after the fix

    # setup
    baseline: dict               # SuiteReport as a dict
    target_failure: str          # output of the target tests on unmodified code
    untracked_baseline: list[str]

    base_sha: str                # commit before the agent changed anything
    repro_file: str              # generated test file (when the agent wrote the test)
    repro_attempts: int
    full_diff: str               # base..working tree: generated test + fix

    # explore
    relevant_files: list[str]
    explored_files: list[str]
    hypothesis: str

    # fix/verify loop
    attempts: int
    history: list[dict]
    patch_errors: list[str]
    context_omitted: list[str]
    regressed_tests: list[str]
    edited_tests: list[str]
    diff: str
    feedback: str
    success: bool

    # terminal failure (setup/model problems, not "the fix didn't work")
    error: str

    timings: dict
