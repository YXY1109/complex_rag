from datetime import datetime

import pytest

from src.utils.common import chinese_to_pinyin, get_collection_name, get_now_time, run_seconds, truncate_filename


@pytest.mark.parametrize("text, expected", [("你好", "nihao"), ("测试", "ceshi"), ("", "")])
def test_chinese_to_pinyin(text, expected):
    result = chinese_to_pinyin(text)
    assert result == expected


@pytest.mark.parametrize("format_str", ["%Y-%m-%d", "%H:%M:%S"])
def test_get_now_time(format_str):
    result = get_now_time(format_str)
    assert isinstance(result, str)


def test_run_seconds():
    result = run_seconds(2)
    assert isinstance(result, datetime)


@pytest.mark.parametrize(
    "filename, max_length, expected_type", [("short_name.txt", 50, str), ("a" * 60 + ".txt", 50, str)]
)
def test_truncate_filename(filename, max_length, expected_type):
    result = truncate_filename(filename, max_length)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize("user_id, expected_prefix", [(123, "knowledge_name_"), (456, "knowledge_name_")])
def test_get_collection_name(user_id, expected_prefix):
    result = get_collection_name(user_id)
    assert result.startswith(expected_prefix)
