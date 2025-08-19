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
    # BOARD 번호 추출
    board_match = re.search(r"BOARD(\d+)", filename)
    board_num = int(board_match.group(1)) if board_match else 999
    
    # Defect 여부 확인
    is_defect = filename.endswith("_D.png")
    
    # 소자 타입과 번호 추출
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
    접두사 정렬 순위 반환: C -> FID -> R -> 알파벳순
    """
    if prefix in PREFIX_ORDER:
        return PREFIX_ORDER[prefix]
    # 그 외는 사전순 + 뒤쪽에 배치
    return 100 + ord(prefix[0]) if prefix else 999

def sort_df(df: pd.DataFrame, fname_col: str) -> pd.DataFrame:
    # 파일명을 파트별로 분리
    filename_parts = df[fname_col].astype(str).apply(split_filename_parts)
    
    df2 = df.copy()
    
    # 정렬을 위한 컬럼 추가
    df2["_board_num"] = [parts[0] for parts in filename_parts]
    df2["_prefix"] = [parts[1] for parts in filename_parts]
    df2["_number"] = [parts[2] for parts in filename_parts]
    df2["_is_defect"] = [parts[3] for parts in filename_parts]
    df2["_prefix_rank"] = df2["_prefix"].apply(prefix_rank)
    df2["_original"] = range(len(df2))

    # 정렬 수행:
    # 1. BOARD 번호
    # 2. Defect 여부 (정상 먼저)
    # 3. 소자 타입 순위
    # 4. 소자 번호
    # 5. 원본 순서
    df2 = df2.sort_values(
        by=["_board_num", "_is_defect", "_prefix_rank", "_number", "_original"],
        ascending=[True, True, True, True, True]
    )
    
    # 임시 컬럼 제거
    return df2.drop(columns=[
        "_board_num", "_prefix", "_number", "_is_defect",
        "_prefix_rank", "_original"
    ])

def sort_excel(df: pd.DataFrame, filename_column: str = None, output_path: str = None, overwrite_check: bool = True):
    """
    데이터프레임을 정렬하여 반환하거나 저장합니다.
    
    Args:
        df: 정렬할 데이터프레임
        filename_column: 파일명이 있는 열 이름. None이면 자동 감지
        output_path: 저장할 파일 경로. None이면 저장하지 않고 데이터프레임만 반환
        overwrite_check: True면 파일이 이미 존재할 경우 덮어쓰기 여부를 물어봄
    
    Returns:
        정렬된 데이터프레임
    """
    # 파일명 컬럼 찾기
    if filename_column is None:
        for col in df.columns:
            if df[col].astype(str).str.contains(r"\.png$", regex=True).any():
                filename_column = col
                break
        if not filename_column:
            raise RuntimeError("파일명 컬럼을 찾지 못했습니다.")

    sorted_df = sort_df(df, filename_column)
    
    if output_path:
        # 파일 존재 여부 확인 및 덮어쓰기 체크
        if os.path.exists(output_path) and overwrite_check:
            response = input(f"\n{os.path.basename(output_path)} 파일이 이미 존재합니다. 덮어쓰시겠습니까? (Y/N): ").strip().upper()
            if response != 'Y':
                print(f"저장 취소됨: {output_path}")
                return sorted_df

        sorted_df.to_excel(output_path, index=False)
        print(f"파일 저장됨: {output_path}")
    
    return sorted_df

if __name__ == "__main__":
    print("이 모듈은 직접 실행하지 않고 다른 스크립트에서 import하여 사용합니다.")
