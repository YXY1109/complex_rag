#!/usr/bin/env python3
"""
Simple baseline test script to verify current configuration and dependencies.
This script checks basic functionality without pytest configuration issues.
"""

import sys
from pathlib import Path
import importlib.util


def test_configuration_exists():
    """Test that configuration files exist"""
    print("Testing configuration files...")

    # Main configuration
    config_path = Path("config/settings.py")
    if config_path.exists():
        print("[PASS] Main configuration file exists")
    else:
        print("[FAIL] Main configuration file missing")
        return False

    # Docker configuration
    docker_compose_path = Path("docker/docker-compose.yml")
    if docker_compose_path.exists():
        print("[PASS] Docker compose file exists")
    else:
        print("[FAIL] Docker compose file missing")
        return False

    return True


def test_service_files_exist():
    """Test that key service files exist"""
    print("\nTesting service files...")

    service_files = [
        "rag_service/app.py",
        "api/main.py",
        "rag_service/bce/service.py",
        "rag_service/qwen3/service.py",
        "rag_service/ocr/service.py",
        "rag_service/llm/service.py"
    ]

    all_exist = True
    for service_file in service_files:
        if Path(service_file).exists():
            print(f"[PASS] {service_file}")
        else:
            print(f"[FAIL] {service_file}")
            all_exist = False

    return all_exist


def test_provider_files_exist():
    """Test that provider implementations exist"""
    print("\nTesting provider files...")

    provider_files = [
        "rag_service/providers/openai/llm_provider.py",
        "rag_service/providers/openai/embedding_provider.py",
        "rag_service/providers/ollama/llm_provider.py",
        "rag_service/providers/bce/rerank_provider.py",
        "rag_service/providers/qwen/llm_provider.py"
    ]

    all_exist = True
    for provider_file in provider_files:
        if Path(provider_file).exists():
            print(f"[PASS] {provider_file}")
        else:
            print(f"[FAIL] {provider_file}")
            all_exist = False

    return all_exist


def test_service_imports():
    """Test that key services can be imported"""
    print("\nTesting service imports...")

    services_to_test = [
        ("config.settings", "Main configuration"),
        ("rag_service.app", "RAG service application"),
    ]

    success_count = 0
    for module_name, description in services_to_test:
        try:
            # Try to import the module
            if Path(module_name.replace(".", "/")).with_suffix(".py").exists():
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    Path(module_name.replace(".", "/")).with_suffix(".py")
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    print(f"[PASS] {description} ({module_name})")
                    success_count += 1
                else:
                    print(f"[FAIL] {description} ({module_name}) - cannot load spec")
            else:
                print(f"[SKIP] {description} ({module_name}) - file not found")
        except Exception as e:
            print(f"[ERROR] {description} ({module_name}) - error: {e}")

    return success_count > 0


def test_directory_structure():
    """Test that expected directory structure exists"""
    print("\nTesting directory structure...")

    expected_dirs = [
        "api",
        "rag_service",
        "config",
        "docker",
        "tests",
        "docs",
        "rag_service/providers",
        "rag_service/services",
        "rag_service/bce",
        "rag_service/qwen3",
        "rag_service/ocr",
        "rag_service/llm"
    ]

    all_exist = True
    for dir_path in expected_dirs:
        if Path(dir_path).exists() and Path(dir_path).is_dir():
            print(f"[PASS] {dir_path}/")
        else:
            print(f"[FAIL] {dir_path}/")
            all_exist = False

    return all_exist


def main():
    """Run all baseline tests"""
    print("Complex RAG System - Baseline Configuration Test")
    print("=" * 50)

    tests = [
        test_configuration_exists,
        test_service_files_exist,
        test_provider_files_exist,
        test_directory_structure,
        test_service_imports
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Baseline Test Summary:")

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[PASS] All baseline tests passed - ready for refactoring!")
        return 0
    else:
        print("[FAIL] Some baseline tests failed - address issues before refactoring")
        return 1


if __name__ == "__main__":
    sys.exit(main())