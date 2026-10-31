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

IMAGE = "repodoctor-sandbox:latest"
INSTALL_TIMEOUT_S = 900
BUILD_TIMEOUT_S = 1200
MAX_LINES = 200
MAX_CHARS = 20_000
_DOCKERFILE = Path(__file__).resolve().parents[3] / "docker" / "sandbox.Dockerfile"


class SandboxError(RuntimeError):
    """Docker is unavailable, or docker command failed."""

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
            "run", "-d", "--name", self.name,
            "--memory", self.memory,
            "--cpus", str(self.cpus),
            "--pids-limit", "512",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,source={self.workdir},target=/work",
            "-w", "/work",
            self.image, "sleep", "infinity",
        )

    def run(self, cmd: str, timeout: int) -> ExecResult:
        if not self.name:
            raise SandboxError("sandbox is not started")
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
        # 124 = `timeout` fired; 137 = it had to SIGKILL (could also be an OOM kill)
        return ExecResult(
            exit_code=p.returncode,
            output=truncate_output(p.stdout),
            timed_out=p.returncode in (124, 137),
            duration_s=time.monotonic() - start,
        )

    def install(self, info: ProjectInfo) -> ExecResult:
        cmds = build_install_commands(self.workdir, info)
        return self.run(" && ".join(cmds), INSTALL_TIMEOUT_S)

    def lock_network(self) -> None:
        """Cut the container off from the network. Call after install."""
        _docker("network", "disconnect", "bridge", self.name or "")

    def run_tests(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> ExecResult:
        cmd = " ".join(shlex.quote(a) for a in [*info.test_command, *(args or [])])
        return self.run(cmd, timeout)

    def destroy(self) -> None:
        if self.name:
            _docker("rm", "-f", self.name, check=False)
            self.name = None