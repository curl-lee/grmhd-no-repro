from pathlib import Path
import re


WINDOWS_ACTIVE_PATTERN = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|Users[\\/]Administrator|Windows 11|PowerShell|cmd\.exe|wsl\.exe|junction)",
    re.IGNORECASE,
)


def test_active_files_have_no_windows_paths_or_commands():
    roots = [Path("src"), Path("scripts"), Path("configs"), Path("tests")]
    files = [Path("README_GRMHD_REPRO.md"), Path("pyproject.toml")]
    for root in roots:
        files.extend(
            path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml", ".md"}
        )
    findings = []
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if WINDOWS_ACTIVE_PATTERN.search(line):
                findings.append(f"{path}:{line_number}:{line}")
    assert not findings, "Active Windows remnants:\n" + "\n".join(findings)
