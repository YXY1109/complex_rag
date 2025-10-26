#!/usr/bin/env python3
"""
测试运行脚本
用于运行单元测试、集成测试并生成覆盖率报告
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    if description:
        print(f"📋 {description}")
    print(f"🔧 Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )

        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("📤 STDERR:")
            print(result.stderr)

        return result.returncode == 0, result.stdout, result.stderr

    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False, "", str(e)


def install_test_dependencies():
    """安装测试依赖"""
    print("📦 Installing test dependencies...")

    requirements_file = project_root / "tests" / "requirements.txt"
    if requirements_file.exists():
        success, stdout, stderr = run_command([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], "Installing test dependencies")

        if success:
            print("✅ Test dependencies installed successfully")
        else:
            print("⚠️  Warning: Could not install test dependencies")
            print(stderr)
    else:
        print("⚠️  Test requirements file not found")


def check_test_environment():
    """检查测试环境"""
    print("🔍 Checking test environment...")

    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("⚠️  Warning: Python 3.8+ is recommended for testing")

    # 检查必要的包
    required_packages = ["pytest", "pytest-asyncio", "pytest-cov"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")

    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        return False

    print("✅ Test environment is ready")
    return True


def run_unit_tests(verbose=False, coverage=False):
    """运行单元测试"""
    print("🧪 Running unit tests...")

    cmd = [sys.executable, "-m", "pytest"]

    # 基础参数
    cmd.extend([
        "tests/unit",
        "-v" if verbose else "-q",
        "--tb=short"
    ])

    # 覆盖率参数
    if coverage:
        cmd.extend([
            "--cov=interfaces",
            "--cov=infrastructure",
            "--cov=services",
            "--cov=api",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ])

    # 异步测试支持
    cmd.extend(["--asyncio-mode=auto"])

    success, stdout, stderr = run_command(cmd, "Running unit tests")

    if success:
        print("✅ Unit tests passed")
    else:
        print("❌ Unit tests failed")

    return success, stdout, stderr


def run_integration_tests(verbose=False):
    """运行集成测试"""
    print("🔗 Running integration tests...")

    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend([
        "tests/integration",
        "-v" if verbose else "-q",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

    success, stdout, stderr = run_command(cmd, "Running integration tests")

    if success:
        print("✅ Integration tests passed")
    else:
        print("❌ Integration tests failed")

    return success, stdout, stderr


def run_all_tests(verbose=False, coverage=False):
    """运行所有测试"""
    print("🚀 Running all tests...")

    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend([
        "tests",
        "-v" if verbose else "-q",
        "--tb=short"
    ])

    if coverage:
        cmd.extend([
            "--cov=interfaces",
            "--cov=infrastructure",
            "--cov=services",
            "--cov=api",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ])

    cmd.extend(["--asyncio-mode=auto"])

    success, stdout, stderr = run_command(cmd, "Running all tests")

    if success:
        print("✅ All tests passed")
    else:
        print("❌ Some tests failed")

    return success, stdout, stderr


def run_specific_tests(test_path, verbose=False):
    """运行特定测试"""
    print(f"🎯 Running specific tests: {test_path}")

    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend([
        test_path,
        "-v" if verbose else "-q",
        "--tb=short",
        "--asyncio-mode=auto"
    ])

    success, stdout, stderr = run_command(cmd, f"Running tests in {test_path}")

    if success:
        print(f"✅ Tests in {test_path} passed")
    else:
        print(f"❌ Tests in {test_path} failed")

    return success, stdout, stderr


def generate_coverage_report():
    """生成覆盖率报告"""
    print("📊 Generating coverage report...")

    # 确保覆盖率数据存在
    success, stdout, stderr = run_command([
        sys.executable, "-m", "coverage", "report"
    ], "Coverage report")

    if not success:
        print("⚠️  No coverage data found, running tests with coverage...")
        run_all_tests(coverage=True)

    # 生成HTML报告
    success, stdout, stderr = run_command([
        sys.executable, "-m", "coverage", "html"
    ], "Generating HTML coverage report")

    if success:
        html_path = project_root / "htmlcov" / "index.html"
        if html_path.exists():
            print(f"✅ HTML coverage report generated: {html_path}")
            print(f"🌐 Open in browser: file://{html_path}")
        else:
            print("⚠️  HTML report file not found")
    else:
        print("❌ Failed to generate HTML coverage report")


def lint_code():
    """代码质量检查"""
    print("🔍 Running code quality checks...")

    # 运行flake8
    success, stdout, stderr = run_command([
        sys.executable, "-m", "flake8",
        "interfaces",
        "infrastructure",
        "services",
        "api",
        "--max-line-length=120",
        "--ignore=E203,W503"
    ], "Running flake8")

    if success:
        print("✅ Code style check passed")
    else:
        print("⚠️  Code style issues found")

    return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Test runner for Complex RAG project")
    parser.add_argument(
        "command",
        choices=["unit", "integration", "all", "coverage", "lint", "check", "specific"],
        help="Command to run"
    )
    parser.add_argument(
        "--path",
        help="Specific test path (used with 'specific' command)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--cov", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing test dependencies"
    )

    args = parser.parse_args()

    print("Complex RAG Test Runner")
    print(f"Project root: {project_root}")
    print(f"Python: {sys.version}")
    print()

    # 检查环境
    if not check_test_environment():
        if not args.skip_install:
            install_test_dependencies()
            if not check_test_environment():
                print("❌ Environment setup failed")
                return 1
        else:
            print("❌ Environment check failed")
            return 1

    success = True

    try:
        if args.command == "check":
            # 只检查环境
            pass

        elif args.command == "unit":
            success, _, _ = run_unit_tests(verbose=args.verbose, coverage=args.coverage)

        elif args.command == "integration":
            success, _, _ = run_integration_tests(verbose=args.verbose)

        elif args.command == "all":
            success, _, _ = run_all_tests(verbose=args.verbose, coverage=args.coverage)

        elif args.command == "coverage":
            success, _, _ = run_all_tests(verbose=args.verbose, coverage=True)
            if success:
                generate_coverage_report()

        elif args.command == "lint":
            success = lint_code()

        elif args.command == "specific":
            if not args.path:
                print("❌ Error: --path is required for 'specific' command")
                return 1
            success, _, _ = run_specific_tests(args.path, verbose=args.verbose)

        else:
            print(f"❌ Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    if success:
        print("\n🎉 Tests completed successfully!")
        return 0
    else:
        print("\n💥 Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())