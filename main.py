"""
Mini NPU Simulator
==================
MAC(Multiply-Accumulate) 연산으로 패턴이 Cross(+)인지 X인지 판별하는 콘솔 프로그램.

- 외부 라이브러리 없이 표준 라이브러리(json, time, re, os, sys)만 사용한다.
- Python 3.8 이상에서 동작한다.

실행: python main.py            (data.json은 main.py와 같은 폴더)
      python main.py other.json (모드 2에서 다른 JSON 파일을 분석하고 싶을 때)
"""

import json
import os
import re
import sys
import time

# ----------------------------------------------------------------------
# 공통 설정값
# ----------------------------------------------------------------------
EPSILON = 1e-9          # 두 점수의 차이가 이보다 작으면 "동점"으로 본다
REPEAT = 10             # 성능 측정 반복 횟수
BENCH_SIZES = [3, 5, 13, 25]
LINE = "#" + "-" * 39

CROSS = "Cross"         # 프로그램 내부 표준 라벨 1
X = "X"                 # 프로그램 내부 표준 라벨 2
UNDECIDED = "UNDECIDED"


def section(title):
    """'# [1] 필터 입력' 같은 구분선 제목을 출력한다."""
    print()
    print(LINE)
    print("# " + title)
    print(LINE)


# ======================================================================
# 1단계: 데이터 구조 (n×n 2차원 리스트)
# ----------------------------------------------------------------------
# 패턴과 필터는 "리스트 안의 리스트"로 저장한다.
#   matrix[i][j]  → i번째 줄, j번째 칸의 값
# ======================================================================
def create_matrix(n, fill=0.0):
    """n×n 크기의 행렬을 fill 값으로 채워서 만든다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치의 값을 읽는다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치에 값을 저장한다."""
    matrix[row][col] = value


def check_square(matrix, n):
    """
    matrix가 숫자로 된 n×n 행렬인지 검사한다.
    문제가 없으면 None, 문제가 있으면 '이유 문자열'을 돌려준다.
    (모드 2에서 프로그램을 멈추지 않고 FAIL 사유로 쓰기 위해 예외 대신 문자열을 반환)
    """
    if not isinstance(matrix, list):
        return "행렬이 리스트 형식이 아님"
    if len(matrix) != n:
        return "행 수 불일치 (기대 {}, 실제 {})".format(n, len(matrix))
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            return "{}행이 리스트 형식이 아님".format(i)
        if len(row) != n:
            return "{}행의 열 수 불일치 (기대 {}, 실제 {})".format(i, n, len(row))
        for value in row:
            # bool은 int의 하위 타입이라 따로 걸러낸다 (true/false는 숫자로 보지 않음)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "{}행에 숫자가 아닌 값 포함: {!r}".format(i, value)
    return None


def format_matrix(matrix):
    """행렬을 보기 좋은 문자열로 만든다 (저장 확인용)."""
    lines = []
    for row in matrix:
        lines.append("  " + " ".join("{:g}".format(v) for v in row))
    return "\n".join(lines)


# ======================================================================
# 2단계: MAC 연산 (이 과제의 핵심)
# ----------------------------------------------------------------------
# 같은 위치끼리 곱하고(Multiply), 그 결과를 계속 더한다(Accumulate).
# 이중 for문 → N×N 번 곱셈/덧셈 → 시간 복잡도 O(N²)
# ======================================================================
def mac(pattern, filt):
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


# [보너스 1] 2차원 → 1차원(길이 N²) 변환 후 MAC
def flatten(matrix):
    """[[a, b], [c, d]] → [a, b, c, d]"""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_flat(pattern_flat, filter_flat):
    """1차원 배열끼리 MAC. 반복문이 한 겹이고, 행 리스트를 거치는 이중 인덱싱이 없다."""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ======================================================================
# 3단계: 판정 (epsilon 기반 동점 처리)
# ----------------------------------------------------------------------
# 컴퓨터의 소수(float)는 오차가 있다. 예) 0.1 + 0.2 = 0.30000000000000004
# 그래서 "==" 나 ">" 로 바로 비교하지 않고,
# 차이가 EPSILON 보다 작으면 "동점(판정 불가)"으로 처리한다.
# ======================================================================
def decide(score_a, score_b, label_a="A", label_b="B"):
    if abs(score_a - score_b) < EPSILON:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


# ======================================================================
# 4단계: 라벨 정규화
# ----------------------------------------------------------------------
# 데이터 곳곳의 표기('+', 'cross', 'x', 'X' ...)를
# 프로그램 내부 표준 라벨(Cross, X) 두 가지로 통일한다.
# ======================================================================
LABEL_MAP = {
    "+": CROSS,
    "cross": CROSS,
    "x": X,
}


def normalize_label(raw):
    """표준 라벨(Cross/X)을 돌려준다. 해석할 수 없으면 None."""
    if raw is None:
        return None
    return LABEL_MAP.get(str(raw).strip().lower())


# ======================================================================
# [보너스 2] 패턴 생성기
# ----------------------------------------------------------------------
# N을 주면 N×N 십자가 / X 패턴을 자동으로 만든다.
# 모드 1의 필터 자동 입력과 성능 분석(3×3 등)에 재활용한다.
# ======================================================================
def make_cross(n):
    """가운데 행과 가운데 열이 1인 십자가 패턴."""
    m = create_matrix(n)
    mid = n // 2
    for k in range(n):
        set_value(m, mid, k, 1.0)
        set_value(m, k, mid, 1.0)
    return m


def make_x(n):
    """두 대각선이 1인 X 패턴."""
    m = create_matrix(n)
    for k in range(n):
        set_value(m, k, k, 1.0)
        set_value(m, k, n - 1 - k, 1.0)
    return m


# ======================================================================
# 7단계: 성능 측정
# ----------------------------------------------------------------------
# - 입력/출력/파일 읽기는 빼고 "MAC 함수 호출 구간"만 측정한다.
# - REPEAT(10)회 반복한 뒤 평균(ms)을 낸다.
# ======================================================================
ROUNDS = 5             # "10회 평균" 측정을 몇 번 반복할지 (잡음 제거용)


def measure_ms(func, *args):
    """
    func(*args)를 REPEAT(10)회 실행한 평균 시간(ms)을 잰다.
    μs 단위의 짧은 연산은 그 순간 OS가 다른 일을 하면 값이 크게 튄다.
    그래서 '10회 평균'을 ROUNDS(5)번 측정하고, 방해를 가장 덜 받은 최솟값을 쓴다.
    """
    func(*args)  # 워밍업 1회 (첫 호출의 잡음 제거, 측정에는 포함하지 않음)
    best = None
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(REPEAT):
            func(*args)
        end = time.perf_counter()
        avg = (end - start) / REPEAT * 1000.0
        if best is None or avg < best:
            best = avg
    return best


def print_benchmark(sizes):
    """크기별 평균 시간 표 + (보너스) 2D/1D 비교 표를 출력한다."""
    print("{:<8}{:>16}{:>12}".format("크기", "평균 시간(ms)", "연산 횟수"))
    print("-" * 40)
    rows = []
    for n in sizes:
        pattern = make_cross(n)
        filt = make_x(n)
        ms_2d = measure_ms(mac, pattern, filt)
        p_flat, f_flat = flatten(pattern), flatten(filt)   # 변환 시간은 측정에서 제외
        ms_1d = measure_ms(mac_flat, p_flat, f_flat)
        rows.append((n, ms_2d, ms_1d))
        print("{:<10}{:>14.4f}{:>12}".format("{}×{}".format(n, n), ms_2d, n * n))

    print()
    print("[보너스] 2D vs 1D(평탄화) 비교 (동일 입력, 평균/{}회)".format(REPEAT))
    print("{:<8}{:>12}{:>12}{:>10}".format("크기", "2D(ms)", "1D(ms)", "1D/2D"))
    print("-" * 44)
    for n, ms_2d, ms_1d in rows:
        ratio = ms_1d / ms_2d if ms_2d > 0 else 0.0
        print("{:<10}{:>12.4f}{:>12.4f}{:>10.2f}".format(
            "{}×{}".format(n, n), ms_2d, ms_1d, ratio))


# ======================================================================
# 5단계: 모드 1 - 사용자 입력 (3×3)
# ======================================================================
def read_matrix(name, n):
    """
    n×n 행렬을 한 줄씩 입력받는다.
    - 숫자 개수가 n개가 아니거나, 숫자로 바꿀 수 없으면 안내 후 그 줄을 다시 입력받는다.
    """
    print("{} ({}줄 입력, 공백 구분)".format(name, n))
    matrix = []
    while len(matrix) < n:
        line = input()
        parts = line.split()
        if len(parts) != n:
            print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요. "
                  "({}번째 줄 다시 입력)".format(n, len(matrix) + 1))
            continue
        try:
            row = [float(p) for p in parts]
        except ValueError:
            print("입력 형식 오류: 숫자가 아닌 값이 있습니다. "
                  "({}번째 줄 다시 입력)".format(len(matrix) + 1))
            continue
        matrix.append(row)
    return matrix


def ask_choice(prompt, choices):
    """choices 중 하나가 입력될 때까지 반복해서 묻는다."""
    while True:
        answer = input(prompt).strip()
        if answer in choices:
            return answer
        print("잘못된 선택입니다. {} 중에서 입력하세요.".format("/".join(choices)))


def run_user_mode():
    n = 3

    # [1] 필터 입력
    section("[1] 필터 입력")
    how = ask_choice("필터 입력 방식 (1: 직접 입력, 2: 패턴 생성기 A=Cross, B=X): ",
                     ["1", "2"])
    if how == "1":
        filter_a = read_matrix("필터 A", n)
        print()
        filter_b = read_matrix("필터 B", n)
    else:
        filter_a, filter_b = make_cross(n), make_x(n)

    # 저장 확인
    print()
    print("✓ 필터 저장 완료")
    print("필터 A:")
    print(format_matrix(filter_a))
    print("필터 B:")
    print(format_matrix(filter_b))

    # [2] 패턴 입력
    section("[2] 패턴 입력")
    how = ask_choice("패턴 입력 방식 (1: 직접 입력, 2: 생성기 Cross, 3: 생성기 X): ",
                     ["1", "2", "3"])
    if how == "1":
        pattern = read_matrix("패턴", n)
    else:
        pattern = make_cross(n) if how == "2" else make_x(n)
        print("생성된 패턴:")
        print(format_matrix(pattern))

    # [3] MAC 연산 + 판정
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed = (measure_ms(mac, pattern, filter_a) + measure_ms(mac, pattern, filter_b)) / 2
    result = decide(score_a, score_b, "A", "B")

    section("[3] MAC 결과")
    print("A 점수: {!r}".format(score_a))
    print("B 점수: {!r}".format(score_b))
    print("연산 시간(평균/{}회): {:.4f} ms".format(REPEAT, elapsed))
    if result == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < {:g})".format(EPSILON))
    else:
        print("판정: {}".format(result))

    # [4] 성능 분석 (3×3)
    section("[4] 성능 분석 (평균/{}회)".format(REPEAT))
    print_benchmark([3])


# ======================================================================
# [보너스 2] 모드 3 - 패턴 생성기
# ----------------------------------------------------------------------
# 크기 N을 입력받아 N×N Cross / X 패턴을 만들고,
# 생성된 패턴으로 판정 확인 + 성능 분석까지 수행한다.
# ======================================================================
MIN_N, MAX_N, PRINT_LIMIT = 3, 99, 25


def read_size():
    """홀수 N(3~99)을 입력받는다. 십자가/X의 '정중앙 칸'이 있으려면 N이 홀수여야 한다."""
    while True:
        text = input("크기 N 입력 ({}~{} 사이 홀수): ".format(MIN_N, MAX_N)).strip()
        try:
            n = int(text)
        except ValueError:
            print("입력 형식 오류: 정수를 입력하세요.")
            continue
        if n < MIN_N or n > MAX_N or n % 2 == 0:
            print("입력 범위 오류: {}~{} 사이의 홀수를 입력하세요.".format(MIN_N, MAX_N))
            continue
        return n


def run_generator_mode():
    # [1] 크기 입력 + 패턴 생성
    section("[1] 패턴 생성")
    n = read_size()
    cross, x = make_cross(n), make_x(n)
    if n <= PRINT_LIMIT:
        print("Cross 패턴 ({0}×{0}):".format(n))
        print(format_matrix(cross))
        print("X 패턴 ({0}×{0}):".format(n))
        print(format_matrix(x))
    else:
        print("✓ {0}×{0} Cross / X 패턴 생성 완료 (크기가 커서 출력 생략)".format(n))

    # [2] 생성된 패턴을 필터로도 재활용해 판정 확인
    section("[2] 생성 패턴 판정 확인 (필터 = 생성된 Cross / X)")
    for name, pattern, expected in (("Cross 패턴", cross, CROSS), ("X 패턴", x, X)):
        score_cross = mac(pattern, cross)
        score_x = mac(pattern, x)
        verdict = decide(score_cross, score_x, CROSS, X)
        status = "PASS" if verdict == expected else "FAIL"
        print("--- {} ---".format(name))
        print("Cross 점수: {!r} | X 점수: {!r}".format(score_cross, score_x))
        print("판정: {} | expected: {} | {}".format(verdict, expected, status))

    # [3] 성능 분석: 기준 크기 + 입력한 N
    section("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    sizes = sorted(set(BENCH_SIZES + [n]))
    print_benchmark(sizes)


# ======================================================================
# 6단계: 모드 2 - data.json 분석
# ======================================================================
KEY_PATTERN = re.compile(r"^size_(\d+)_(\d+)$")


def load_filters(data):
    """
    data['filters']에서 size_N 필터를 읽어 {N: {'Cross': 행렬, 'X': 행렬}} 형태로 만든다.
    잘못된 필터는 건너뛰고 errors {N: 사유}에 기록한다 (프로그램은 계속 진행).
    """
    filters = {}
    errors = {}
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        print("✗ 'filters' 항목이 없거나 형식이 잘못되었습니다.")
        return filters, errors

    for key, pair in raw_filters.items():
        m = re.match(r"^size_(\d+)$", str(key))
        if not m:
            print("✗ {}: 필터 키 형식 오류 (size_N 이어야 함)".format(key))
            continue
        n = int(m.group(1))
        if not isinstance(pair, dict):
            errors[n] = "필터 {} 형식 오류".format(key)
            print("✗ {}".format(errors[n]))
            continue

        normalized = {}
        problem = None
        for label_key, matrix in pair.items():
            label = normalize_label(label_key)          # 'cross' → Cross, 'x' → X
            if label is None:
                problem = "알 수 없는 필터 라벨 '{}'".format(label_key)
                break
            reason = check_square(matrix, n)
            if reason:
                problem = "{} {} 필터: {}".format(key, label, reason)
                break
            normalized[label] = matrix
        if problem is None and (CROSS not in normalized or X not in normalized):
            problem = "{}에 Cross/X 필터가 모두 있어야 함".format(key)

        if problem:
            errors[n] = problem
            print("✗ {}".format(problem))
        else:
            filters[n] = normalized
            print("✓ {:<7} 필터 로드 완료 (Cross, X)".format(key))
    return filters, errors


def analyze_case(key, case, filters, filter_errors):
    """
    패턴 하나를 분석한다.
    반환: dict(cross, x, verdict, expected, passed, reason)
    어떤 문제가 생겨도 예외를 밖으로 던지지 않고 FAIL + 사유로 돌려준다.
    """
    result = {"cross": None, "x": None, "verdict": None,
              "expected": None, "passed": False, "reason": ""}

    # (1) 키에서 N 추출: size_13_2 → 13
    m = KEY_PATTERN.match(str(key))
    if not m:
        result["reason"] = "[스키마] 키 형식 오류 (size_N_idx 이어야 함)"
        return result
    n = int(m.group(1))

    if not isinstance(case, dict):
        result["reason"] = "[스키마] 패턴 항목이 객체 형식이 아님"
        return result

    # (2) expected 라벨 정규화: '+' → Cross, 'x' → X
    expected = normalize_label(case.get("expected"))
    result["expected"] = expected
    if expected is None:
        result["reason"] = "[라벨] expected 값을 해석할 수 없음: {!r}".format(
            case.get("expected"))
        return result

    # (3) 해당 크기의 필터 선택
    if n not in filters:
        if n in filter_errors:
            result["reason"] = "[스키마] size_{} 필터 오류: {}".format(n, filter_errors[n])
        else:
            result["reason"] = "[스키마] size_{} 필터가 없음".format(n)
        return result

    # (4) 패턴 크기 검증 (필터와 같은 n×n인지)
    pattern = case.get("input")
    reason = check_square(pattern, n)
    if reason:
        if "불일치" in reason:
            result["reason"] = "[크기] 패턴과 필터 크기 불일치 - {}".format(reason)
        else:
            result["reason"] = "[스키마] 패턴 데이터 오류 - {}".format(reason)
        return result

    # (5) MAC 연산 + 판정
    score_cross = mac(pattern, filters[n][CROSS])
    score_x = mac(pattern, filters[n][X])
    verdict = decide(score_cross, score_x, CROSS, X)
    result.update(cross=score_cross, x=score_x, verdict=verdict)

    if verdict == expected:
        result["passed"] = True
    elif verdict == UNDECIDED:
        result["reason"] = "[수치] 동점(UNDECIDED) - |Cross-X|={:.3g} < {:g}".format(
            abs(score_cross - score_x), EPSILON)
    else:
        result["reason"] = "[판정] 판정 {} ≠ expected {}".format(verdict, expected)
    return result


def run_json_mode(path):
    # 파일 읽기 (I/O) - 파일이 없거나 JSON 문법이 틀려도 프로그램이 죽지 않게 처리
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("오류: '{}' 파일을 찾을 수 없습니다.".format(path))
        return
    except json.JSONDecodeError as e:
        print("오류: JSON 형식이 올바르지 않습니다. ({})".format(e))
        return
    if not isinstance(data, dict):
        print("오류: JSON 최상위가 객체 형식이 아닙니다.")
        return

    # [1] 필터 로드
    section("[1] 필터 로드")
    filters, filter_errors = load_filters(data)

    # [2] 패턴 분석
    section("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("✗ 'patterns' 항목이 없거나 형식이 잘못되었습니다.")
        patterns = {}

    total, passed, failures = 0, 0, []
    for key, case in patterns.items():
        total += 1
        try:
            r = analyze_case(key, case, filters, filter_errors)
        except Exception as e:  # 예상 못 한 오류도 케이스 단위 FAIL로 처리
            r = {"cross": None, "x": None, "verdict": None, "expected": None,
                 "passed": False, "reason": "[예외] {}: {}".format(type(e).__name__, e)}

        print("--- {} ---".format(key))
        if r["cross"] is not None:
            print("Cross 점수: {!r}".format(r["cross"]))
            print("X 점수: {!r}".format(r["x"]))
            status = "PASS" if r["passed"] else "FAIL"
            print("판정: {} | expected: {} | {}".format(r["verdict"], r["expected"], status))
        else:
            print("FAIL - {}".format(r["reason"]))

        if r["passed"]:
            passed += 1
        else:
            failures.append((key, r["reason"]))

    # [3] 성능 분석
    section("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    print_benchmark(BENCH_SIZES)

    # [4] 결과 요약
    section("[4] 결과 요약")
    print("총 테스트: {}개".format(total))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(total - passed))
    if failures:
        print()
        print("실패 케이스:")
        for key, reason in failures:
            print("- {}: {}".format(key, reason))


# ======================================================================
# 실행 흐름: 모드 선택 → 모드별 실행
# ======================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "data.json")

    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 (보너스)")
    try:
        mode = ask_choice("선택: ", ["1", "2", "3"])
        if mode == "1":
            run_user_mode()
        elif mode == "2":
            run_json_mode(json_path)
        else:
            run_generator_mode()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")


if __name__ == "__main__":
    main()
