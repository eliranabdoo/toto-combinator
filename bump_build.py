import sys
import subprocess
from pathlib import Path

try:
    import toml
except ImportError:
    print("Please install toml: poetry add --dev toml")
    sys.exit(1)

def bump_version(version: str, part: str) -> str:
    major, minor, patch = map(int, version.split('.'))
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("part must be one of: major, minor, patch")
    return f"{major}.{minor}.{patch}"

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"major", "minor", "patch"}:
        print("Usage: poetry run python bump_build.py [major|minor|patch]")
        sys.exit(1)
    part = sys.argv[1]
    pyproject_path = Path("pyproject.toml")
    data = toml.load(pyproject_path)
    old_version = data["tool"]["poetry"]["version"]
    new_version = bump_version(old_version, part)
    data["tool"]["poetry"]["version"] = new_version
    pyproject_path.write_text(toml.dumps(data), encoding="utf-8")
    print(f"Bumped version: {old_version} -> {new_version}")

    exe_name = f"קומבינציות_טוטו_{new_version}.exe"
    subprocess.run([
        "poetry", "run", "pyinstaller",
        "--onefile", "--name", exe_name, "app.py"
    ], check=True)
    print(f"Built exe: dist/{exe_name}")

if __name__ == "__main__":
    main()
