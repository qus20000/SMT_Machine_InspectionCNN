import os
import re
import pandas as pd

# 우선순위 정의: C -> FID -> R -> 그 외 알파벳순
PREFIX_ORDER = {"C": 0, "FID": 1, "R": 2}

def split_filename_parts(filename: str):
    """
    파일명을 (board_num, prefix, number, is_defect) 형태로 분리
    예: 'BOARD1_C1.png' -> (1, 'C', 1, False)
        'BOARD15_C1_D.png' -> (15, 'C', 1, True)
    """
    board_match = re.search(r"BOARD(\d+)", filename)
    board_num = int(board_match.group(1)) if board_match else 999

    is_defect = filename.endswith("_D.png")

    comp_match = re.search(r"_([A-Za-z]+)(\d+)(?:_D)?\.png", filename)
    if comp_match:
        prefix = comp_match.group(1)
        number = int(comp_match.group(2))
    else:
        prefix = ""
        number = 10**12

    return board_num, prefix, number, is_defect

def prefix_rank(prefix: str) -> int:
    """
    접두사 정렬 순위: C -> FID -> R -> 알파벳순
    """
    if prefix in PREFIX_ORDER:
        return PREFIX_ORDER[prefix]
    return 100 + ord(prefix[0]) if prefix else 999

def sort_df(df: pd.DataFrame, fname_col: str) -> pd.DataFrame:
    """
    파일명 열을 규칙대로 파싱하여 정렬 키를 만든 뒤 정렬한다.
    정렬 우선순위:
      1) BOARD 번호
      2) Defect 여부 (정상 먼저)
      3) 접두사 순위(C/FID/R/알파벳)
      4) 숫자
      5) 원본 순서
    """
    filename_parts = df[fname_col].astype(str).apply(split_filename_parts)

    df2 = df.copy()
    df2["_board_num"] = [p[0] for p in filename_parts]
    df2["_prefix"]    = [p[1] for p in filename_parts]
    df2["_number"]    = [p[2] for p in filename_parts]
    df2["_is_defect"] = [p[3] for p in filename_parts]
    df2["_prefix_rank"] = df2["_prefix"].apply(prefix_rank)
    df2["_original"]  = range(len(df2))

    df2 = df2.sort_values(
        by=["_board_num", "_is_defect", "_prefix_rank", "_number", "_original"],
        ascending=[True, True, True, True, True]
    )

    return df2.drop(columns=[
        "_board_num", "_prefix", "_number", "_is_defect",
        "_prefix_rank", "_original"
    ])

def _resolve_overwrite_check(user_flag):
    """
    덮어쓰기 정책 결정을 일원화한다.
    우선순위:
      1) 환경변수 EXCELSORTER_OVERWRITE
         - "1/true/on/yes"  -> 자동 덮어쓰기 (프롬프트 없음) -> overwrite_check=False
         - "0/false/off/no" -> 프롬프트로 확인           -> overwrite_check=True
      2) 함수 인자 overwrite_check 가 None이 아니면 그 값 사용
      3) 기본값(False): 자동 덮어쓰기(프롬프트 없음)
    """
    env = os.environ.get("EXCELSORTER_OVERWRITE", "").strip().lower()
    if env in ("1", "true", "on", "yes"):
        return False
    if env in ("0", "false", "off", "no"):
        return True
    if user_flag is not None:
        return bool(user_flag)
    return False

def sort_excel(df: pd.DataFrame,
               filename_column: str = None,
               output_path: str = None,
               overwrite_check: bool | None = None):
    """
    데이터프레임을 정렬하여 반환하거나 저장한다.

    Args:
        df: 정렬할 데이터프레임
        filename_column: 파일명 열 이름. None이면 자동 감지
        output_path: 저장 경로. None이면 저장하지 않고 정렬된 DF만 반환
        overwrite_check: None이면 환경변수 정책을 우선 적용, 없으면 기본 False(자동 덮어쓰기)
                         True면 덮어쓰기 전 사용자 확인 프롬프트 표시

    Returns:
        정렬된 데이터프레임
    """
    # 파일명 컬럼 자동 감지
    if filename_column is None:
        for col in df.columns:
            if df[col].astype(str).str.contains(r"\.png$", regex=True).any():
                filename_column = col
                break
        if not filename_column:
            raise RuntimeError("파일명 컬럼을 찾지 못했습니다.")

    sorted_df = sort_df(df, filename_column)

    if output_path:
        # 덮어쓰기 정책 결정
        ow_check = _resolve_overwrite_check(overwrite_check)

        # 저장은 임시파일에 먼저 쓴 뒤 원자적 교체(os.replace)
        tmp_path = output_path + ".tmp"
        if os.path.exists(output_path) and ow_check:
            # 사용자 확인 프롬프트
            resp = input(f"\n{os.path.basename(output_path)} 파일이 이미 존재합니다. 덮어쓰시겠습니까? (Y/N): ").strip().upper()
            if resp != "Y":
                print(f"저장 취소됨: {output_path}")
                return sorted_df

        # 엑셀 저장
        sorted_df.to_excel(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        print(f"파일 저장됨: {output_path}")

    return sorted_df

if __name__ == "__main__":
    print("이 모듈은 import 하여 사용하세요.")
