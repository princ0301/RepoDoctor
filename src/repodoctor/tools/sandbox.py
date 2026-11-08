"""Docker sandbox: runs untrusted repo code in a locked-down container.

Uses the docker CLI (no SDK). Lifecycle:
start -> install (network on) -> lock_network -> run tests (network off) -> destroy.
"""

import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.testreport import SuiteReport, parse_junit, parse_outcomes

IMAGE = "repodoctor-sandbox:latest"
PIP_CACHE_VOLUME = "repodoctor-pip-cache"
INSTALL_TIMEOUT_S = 420
PIP_ENV = "export PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=5; "
BUILD_TIMEOUT_S = 1200
MAX_LINES = 200
MAX_CHARS = 20_000
JUNIT_PATH = "/tmp/repodoctor-junit.xml"  # inside the container, never in the repo
_DOCKERFILE = Path(__file__).resolve().parents[3] / "docker" / "sandbox.Dockerfile"


class SandboxError(RuntimeError):
    """Docker is unavailable, or a docker command failed."""


@dataclass
class ExecResult:
    exit_code: int
    output: str          # truncated: safe to hand to an LLM
    timed_out: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def truncate_output(text: str, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS) -> str:
    """Keep the tail: pytest puts the failure summary at the end."""
    lines = text.splitlines()
    dropped = max(0, len(lines) - max_lines)
    out = "\n".join(lines[-max_lines:])
    truncated = dropped > 0
    if len(out) > max_chars:
        out = out[-max_chars:]
        truncated = True
    if not truncated:
        return out
    note = f"[... {dropped} earlier lines truncated ...]" if dropped else "[... truncated ...]"
    return f"{note}\n{out}"


def _docker(*args: str, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise SandboxError("docker was not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"docker {args[0]} timed out after {timeout}s") from e
    if check and p.returncode != 0:
        raise SandboxError(f"docker {' '.join(args[:2])} failed: {p.stderr.strip()}")
    return p


def docker_available() -> bool:
    try:
        return _docker("info", timeout=20, check=False).returncode == 0
    except SandboxError:
        return False


def image_exists() -> bool:
    return _docker("image", "inspect", IMAGE, check=False).returncode == 0


def ensure_image(rebuild: bool = False) -> None:
    if not rebuild and image_exists():
        return
    if not _DOCKERFILE.is_file():
        raise SandboxError(f"Dockerfile not found: {_DOCKERFILE}")
    _docker(
        "build", "-t", IMAGE, "-f", str(_DOCKERFILE), str(_DOCKERFILE.parent),
        timeout=BUILD_TIMEOUT_S,
    )


def _container_state(name: str) -> str:
    p = _docker(
        "inspect", "-f",
        "status={{.State.Status}} exit_code={{.State.ExitCode}} "
        "oom_killed={{.State.OOMKilled}}",
        name, check=False,
    )
    return p.stdout.strip() or p.stderr.strip() or "unknown"


def build_install_commands(root: Path, info: ProjectInfo) -> list[str]:
    """Shell commands that install the project and its test dependencies."""
    cmds = [f"pip install -r {shlex.quote(r)}" for r in info.requirement_files]
    if any((root / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
        target = "." + (f"[{','.join(info.extras)}]" if info.extras else "")
        cmds.append(f"pip install -e {shlex.quote(target)}")
    cmds += [f"pip install --group {shlex.quote(g)}" for g in info.dependency_groups]
    cmds.append("pip install pytest")  # no-op if already installed
    return cmds


class Sandbox:
    def __init__(
        self,
        workdir: Path,
        *,
        memory: str = "2g",
        cpus: float = 2.0,
        image: str = IMAGE,
    ) -> None:
        self.workdir = workdir.resolve()
        self.memory = memory
        self.cpus = cpus
        self.image = image
        self.name: str | None = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.destroy()

    def start(self) -> None:
        if "," in str(self.workdir):
            raise SandboxError("workdir path must not contain commas")
        ensure_image()
        self.name = f"repodoctor-{uuid.uuid4().hex[:8]}"
        _docker(
            "run", "-d", "--init", "--name", self.name,
            "--memory", self.memory,
            "--cpus", str(self.cpus),
            "--pids-limit", "512",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,source={self.workdir},target=/work",
            "--mount", f"type=volume,source={PIP_CACHE_VOLUME},target=/root/.cache/pip",
            "-w", "/work",
            self.image, "sleep", "infinity",
        )

    def _require_running(self) -> None:
        if not self.name:
            raise SandboxError("sandbox is not started")
        p = _docker("inspect", "-f", "{{.State.Running}}", self.name, check=False)
        if p.stdout.strip() != "true":
            raise SandboxError(
                f"container {self.name} is not running ({_container_state(self.name)})"
            )

    def run(self, cmd: str, timeout: int) -> ExecResult:
        self._require_running()
        start = time.monotonic()
        try:
            p = subprocess.run(
                ["docker", "exec", self.name, "timeout", "-k", "5", str(timeout),
                 "bash", "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 30,   # host-side backstop
            )
        except FileNotFoundError as e:
            raise SandboxError("docker was not found on PATH") from e
        except subprocess.TimeoutExpired:
            _docker("kill", self.name, check=False)
            return ExecResult(-1, "[host-side timeout: container killed]", True,
                              time.monotonic() - start)
        if p.returncode != 0 and p.stdout.startswith("Error response from daemon"):
            raise SandboxError(p.stdout.strip())  # docker itself failed, not the command
        # 124 = `timeout` fired; 137 = it had to SIGKILL (could also be an OOM kill)
        return ExecResult(
            exit_code=p.returncode,
            output=truncate_output(p.stdout),
            timed_out=p.returncode in (124, 137),
            duration_s=time.monotonic() - start,
        )

    def read_text(self, path: str) -> str | None:
        """Untruncated file read from inside the container (None if missing)."""
        self._require_running()
        p = _docker("exec", self.name or "", "cat", path, check=False)
        return p.stdout if p.returncode == 0 else None

    def install(
        self, info: ProjectInfo, *, timeout: int = INSTALL_TIMEOUT_S, attempts: int = 2
    ) -> ExecResult:
        cmd = PIP_ENV + " && ".join(build_install_commands(self.workdir, info))
        result = self.run(cmd, timeout)
        for _ in range(attempts - 1):   # a stalled download is usually transient: retry
            if result.ok or not result.timed_out:
                return result
            result = self.run(cmd, timeout)
        return result

    def lock_network(self) -> None:
        """Cut the container off from the network. Call after install."""
        _docker("network", "disconnect", "bridge", self.name or "")

    def run_tests(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> ExecResult:
        """Plain run: raw (truncated) output only."""
        cmd = " ".join(shlex.quote(a) for a in [*info.test_command, *(args or [])])
        return self.run(cmd, timeout)

    def _run_junit(
        self, info: ProjectInfo, args: list[str] | None, timeout: int
    ) -> tuple[ExecResult, str | None]:
        argv = [*info.test_command, f"--junitxml={JUNIT_PATH}", *(args or [])]
        cmd = f"rm -f {JUNIT_PATH}; " + " ".join(shlex.quote(a) for a in argv)
        result = self.run(cmd, timeout)
        return result, self.read_text(JUNIT_PATH)

    def run_suite(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> tuple[ExecResult, SuiteReport | None]:
        """Run tests; the report is None if pytest produced none (crash, timeout)."""
        result, xml = self._run_junit(info, args, timeout)
        try:
            return result, (parse_junit(xml) if xml else None)
        except ValueError:
            return result, None

    def run_outcomes(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> tuple[ExecResult, dict[str, str] | None]:
        """Run tests; returns test id -> outcome (None if pytest produced no report)."""
        result, xml = self._run_junit(info, args, timeout)
        try:
            return result, (parse_outcomes(xml) if xml else None)
        except ValueError:
            return result, None

    def destroy(self) -> None:
        if self.name:
            _docker("rm", "-f", self.name, check=False)
            self.name = None
