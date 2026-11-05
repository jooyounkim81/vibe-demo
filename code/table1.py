"""
Table 1: 치료방법(treatment)별 대상자 특성표
- 입력: data/appendicitis_synthetic.csv
- 출력: output/table1.csv (UTF-8 BOM, 엑셀에서 바로 열림), output/data_issues.csv
- 연속형: 중앙값 (Q1–Q3), 범주형: n (%)  ※ % 분모 = 해당 변수 결측 제외 인원
실행: python3 code/table1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "appendicitis_synthetic.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# 허용 범위 (연구 정의에 맞게 조정 가능)
AGE_RANGE = (0, 110)      # 세
LOS_RANGE = (0, 365)      # 일 (음수 불가)

TREAT_ORDER = ["LA", "OA", "NOM"]

# ---------------------------------------------------------------------------
# 1. 데이터 읽기 (파일 첫머리 BOM 처리, 빈칸은 결측으로)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, encoding="utf-8-sig", skipinitialspace=True)
df = df.replace(r"^\s*$", np.nan, regex=True)
print(f"전체 대상자: {len(df)}명, 변수 {df.shape[1]}개\n")

# ---------------------------------------------------------------------------
# 2. 이상값 점검
# ---------------------------------------------------------------------------
issues = []


def flag(mask, var, rule):
    for _, r in df.loc[mask].iterrows():
        issues.append({"patient_id": r["patient_id"], "variable": var,
                       "value": r[var], "rule": rule})
    return int(mask.sum())


def age_group_of(a):
    if pd.isna(a):
        return np.nan
    return "<18" if a < 18 else ("18-64" if a < 65 else ">=65")


n_age = flag((df["age"] < AGE_RANGE[0]) | (df["age"] > AGE_RANGE[1]),
             "age", f"범위 밖 ({AGE_RANGE[0]}–{AGE_RANGE[1]}세)")
n_los = flag((df["los_days"] < LOS_RANGE[0]) | (df["los_days"] > LOS_RANGE[1]),
             "los_days", f"범위 밖 ({LOS_RANGE[0]}–{LOS_RANGE[1]}일)")
n_grp = flag(df["age"].map(age_group_of).ne(df["age_group"]) & df["age"].notna(),
             "age_group", "age와 age_group 불일치")
n_blank = {}
for c in df.columns:
    n_blank[c] = flag(df[c].isna(), c, "빈칸(결측)")
n_dup = int(df["patient_id"].duplicated().sum())

print("[이상값 점검 결과]")
print(f"  나이 범위 밖        : {n_age}건")
print(f"  재원일수 범위 밖    : {n_los}건")
print(f"  age_group 불일치    : {n_grp}건")
print(f"  빈칸(결측) 합계     : {sum(n_blank.values())}건 "
      f"{ {k: v for k, v in n_blank.items() if v} }")
print(f"  patient_id 중복     : {n_dup}건")
issues = pd.DataFrame(issues)
issues.to_csv(OUT / "data_issues.csv", index=False, encoding="utf-8-sig")
print(issues.to_string(index=False), "\n")

# 처리 원칙: 행은 삭제하지 않고, 불가능한 값만 결측으로 바꾼 뒤 결측 행으로 표시
bad_age = (df["age"] < AGE_RANGE[0]) | (df["age"] > AGE_RANGE[1])
df.loc[bad_age, ["age", "age_group"]] = np.nan   # 나이가 틀리면 연령군도 신뢰 불가
bad_los = (df["los_days"] < LOS_RANGE[0]) | (df["los_days"] > LOS_RANGE[1])
df.loc[bad_los, "los_days"] = np.nan

# ---------------------------------------------------------------------------
# 3. Table 1
# ---------------------------------------------------------------------------
VARS = [
    # (변수, 표시이름, 유형, 범주순서/라벨)
    ("age", "Age, years", "cont", None),
    ("age_group", "Age group", "cat", {"<18": "<18", "18-64": "18–64", ">=65": "≥65"}),
    ("sex", "Sex", "cat", {"M": "Male", "F": "Female"}),
    ("comorbidity", "Comorbidity", "bin", None),
    ("complicated", "Complicated appendicitis", "bin", None),
    ("year", "Year of diagnosis", "cat", None),
    ("los_days", "Length of stay, days", "cont", None),
    ("antibiotic_days", "Antibiotic duration, days", "cont", None),
    ("analgesic_days", "Analgesic duration, days", "cont", None),
    ("complication", "Complication", "bin", None),
    ("readmit_30d", "30-day readmission", "bin", None),
    ("failure_1y", "Treatment failure within 1 year", "bin", None),
]

groups = [("Overall", df)] + [(t, df[df["treatment"] == t]) for t in TREAT_ORDER]


def med_iqr(s):
    s = s.dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:g} ({q1:g}–{q3:g})"


def n_pct(n, d):
    return f"{n} ({100 * n / d:.1f})" if d else "0 (0.0)"


rows = [["Characteristic"] + [f"{g} (n={len(d)})" for g, d in groups]]
for var, label, kind, levels in VARS:
    if kind == "cont":
        rows.append([f"{label}, median (IQR)"] + [med_iqr(d[var]) for _, d in groups])
    elif kind == "bin":
        rows.append([f"{label}, n (%)"] +
                    [n_pct(int((d[var] == 1).sum()), d[var].notna().sum()) for _, d in groups])
    else:
        rows.append([f"{label}, n (%)"] + [""] * len(groups))
        lv = levels or {k: str(k) for k in sorted(df[var].dropna().unique())}
        for k, lab in lv.items():
            rows.append([f"  {lab}"] +
                        [n_pct(int((d[var] == k).sum()), d[var].notna().sum()) for _, d in groups])
    miss = [int(d[var].isna().sum()) for _, d in groups]
    if any(miss):
        rows.append(["  Missing, n"] + [str(m) for m in miss])

table1 = pd.DataFrame(rows[1:], columns=rows[0])
table1.to_csv(OUT / "table1.csv", index=False, encoding="utf-8-sig")
print("[Table 1]")
print(table1.to_csv(sep="\t", index=False))
