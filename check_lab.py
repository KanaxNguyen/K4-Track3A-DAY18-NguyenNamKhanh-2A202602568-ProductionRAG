"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import math
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def check_eval_report(path: str) -> bool:
    """Require actual, complete measurements rather than default scores."""
    if not check_file(path):
        return False
    if not check_json(path, ["aggregate", "num_questions", "per_question"]):
        return False
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    with open("test_set.json", encoding="utf-8") as f:
        expected = len(json.load(f))
    aggregate = report["aggregate"]
    rows = report["per_question"]
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    def valid_score(value):
        return isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1
    complete = (aggregate.get("evaluation_status") == "completed"
                and report["num_questions"] == expected and len(rows) == expected
                and all(valid_score(aggregate.get(k)) for k in metrics)
                and all(valid_score(row.get(k)) for row in rows for k in metrics))
    if complete:
        print(f"  ✅ {path} — {expected} câu hỏi có đủ 4 điểm RAGAS thật")
    else:
        print(f"  ❌ {path} — RAGAS chưa hoàn tất hoặc thiếu điểm đo")
    return complete


def run_tests() -> tuple[int, int, bool]:
    """Run pytest and require an exit code of zero without collection errors."""
    try:
        import re
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q",
             "--junitxml=reports/pytest_results.xml"],
            capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace", check=False
        )
        summary = result.stdout + result.stderr
        m_pass = re.search(r"(\d+)\s+passed", summary)
        m_fail = re.search(r"(\d+)\s+failed", summary)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        m_errors = re.search(r"(\d+)\s+errors?", summary)
        errors = int(m_errors.group(1)) if m_errors else 0
        total = passed + failed + errors
        return passed, total, result.returncode == 0 and total > 0
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0, False


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if not check_eval_report("reports/ragas_report.json"):
        errors += 1
    if not check_eval_report("reports/naive_baseline_report.json"):
        errors += 1

    # 3. Analysis
    print("\n📝 Analysis:")
    if not check_file("analysis/failure_analysis.md"):
        errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections.extend([f"{ref_dir}/{f}" for f in os.listdir(ref_dir)
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])
    if os.path.isdir("analysis"):
        reflections.extend([f"analysis/{f}" for f in os.listdir("analysis")
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])

    if reflections:
        for r in set(reflections):
            print(f"  ✅ {r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân (đặt tại {ref_dir}/reflection_[HọTên].md hoặc analysis/reflection_[HọTên].md)")
        errors += 1

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")
        errors += 1

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total, tests_ok = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if tests_ok else '❌'} {passed}/{total} tests passed ({pct:.0f}%)")
    else:
        print("  ⚠️  Không chạy được tests")
    if not tests_ok:
        errors += 1

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)
    return errors == 0


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
