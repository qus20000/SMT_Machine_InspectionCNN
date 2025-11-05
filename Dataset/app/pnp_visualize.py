# pnp_visualize.py
import os, re, argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------- Robust reader ----------------
def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    with open(path, "rb") as f:
        head = f.read(4096)

    def try_read(encs, seps):
        for enc in encs:
            for sep in seps:
                try:
                    return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                except Exception:
                    pass
        return None

    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff") or head.count(b"\x00") > 50:
        df = try_read(("utf-16", "utf-16-le", "utf-16-be"), ("\t", ",", ";", "|"))
        if df is not None:
            return df

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception:
            df = try_read((enc,), (",", ";", "\t", "|"))
            if df is not None:
                return df

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if len(df.columns) == 1 and any("\x00" in str(c) for c in df.columns):
            df2 = try_read(("utf-16", "utf-16-le"), ("\t", ","))
            if df2 is not None:
                return df2
        return df
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise SystemExit(f"파일을 읽지 못했습니다: {path}\n{e}")

# ---------------- normalize ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def key(s):
        s = str(s).replace("\ufeff","").replace("\xa0"," ").replace("\x00","")
        s = re.sub(r"\s+","", s.strip().lower())
        s = s.replace("-", "").replace("_","").replace("(","").replace(")","")
        return s
    mapping = {
        "designator":"Designator", "refdes":"Designator", "reference":"Designator",
        "midx":"Mid X","x":"Mid X","centerx":"Mid X","posx":"Mid X","xmm":"Mid X",
        "midy":"Mid Y","y":"Mid Y","centery":"Mid Y","posy":"Mid Y","ymm":"Mid Y",
        "footprint":"Footprint","package":"Footprint",
        # defects
        "status":"Status","result":"Status","defect":"Status","ng":"Status","ok":"Status",
        "defectprob":"Score",
        "file":"File","filename":"File","image":"File","img":"File","imagename":"File","path":"File",
        # predictedLabel -> Pred
        "predictedlabel":"Pred","predicted_label":"Pred","label":"Pred",
    }
    ren = {}
    for c in df.columns:
        k = key(c)
        if k in mapping:
            ren[c] = mapping[k]
    df = df.rename(columns=ren)
    df.columns = [str(c).replace("\x00","") for c in df.columns]
    return df

def to_float(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\x00","", regex=False)
    s = s.str.replace("\u2212","-", regex=False)
    s = s.str.replace(",",".", regex=False)
    s = s.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# ---------------- designator helpers ----------------
_DESIG_RE = re.compile(r"([A-Za-z]+)\s*0*([0-9]+)")
def canonical_designator(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\x00", "").upper()
    m = _DESIG_RE.findall(t)
    if not m:
        m2 = re.findall(r"[A-Z]+[0-9]+", t)
        if not m2:
            return ""
        t = m2[-1]
        m = _DESIG_RE.findall(t)
        if not m:
            return t
    prefix, num = m[-1]
    try:
        num_i = int(num)
    except Exception:
        num_i = int(re.sub(r"\D", "", num) or "0")
    return f"{prefix}{num_i}"

def choose_designator_col(df: pd.DataFrame) -> str | None:
    for name in ["Designator","RefDes","Reference"]:
        if name in df.columns:
            return name
    best = (None, -1e9)
    for c in df.columns:
        s = df[c].astype(str)
        pat  = s.str.fullmatch(r"[A-Za-z]+\s*0*\d+").fillna(False).mean()
        anti = s.str.contains(r"0603|0805|1206|KR|X7R|BB\d+", case=False, regex=True).fillna(False).mean()
        score = pat - 0.5*anti
        if score > best[1]:
            best = (c, score)
    return best[0]

def get_designators_from_pnp(df: pd.DataFrame) -> np.ndarray:
    s_idx = pd.Index(df.index).astype(str).map(canonical_designator)
    idx_ratio = (s_idx != "").mean()
    dcol = choose_designator_col(df)
    s_col = df[dcol].astype(str).map(canonical_designator) if dcol else pd.Series([], dtype=str)
    col_ratio = (s_col != "").mean() if len(s_col) else -1
    if idx_ratio >= 0.8 and idx_ratio >= col_ratio:
        print(f"[debug] using index as designator (ratio={idx_ratio:.2f})")
        return s_idx.to_numpy()
    else:
        print(f"[debug] using column as designator: {dcol} (ratio={col_ratio:.2f})")
        return s_col.to_numpy()

# ---------------- package sizes ----------------
PKG_SIZE = {
    "c0603": (1.6, 0.8), "r0603": (1.6, 0.8),
    "c0805": (2.0, 1.25), "r0805": (2.0, 1.25),
    "c1206": (3.2, 1.6),  "r1206": (3.2, 1.6),
}
def norm_pkg(fp: str) -> str:
    if not isinstance(fp, str):
        return ""
    t = re.sub(r"\s+","", fp.strip().lower())
    for k in PKG_SIZE.keys():
        if k in t:
            return k
    if "0603" in t: return "c0603"
    if "0805" in t: return "c0805"
    if "1206" in t: return "c1206"
    return ""

# --- helper: Footprint/Device에서 패키지(C0603/R0805 …) 뽑기 ---
def extract_pkg_from_values(fp_val: str, dev_val: str):
    """
    Footprint 또는 Device 문자열에서 C0603/R0805 같은 패키지명을 추출.
    반환: (표시용 라벨(대문자), 크기계산용 키(소문자) or "")
    둘 다 못 찾으면 (원본 Footprint 문자열, norm_pkg(fp) 결과)
    """
    def _pick(s):
        if not s:
            return None
        s = str(s)
        m = re.search(r'([CR])\s*0?(\d{3,4})', s, flags=re.I)
        if m:
            label = f"{m.group(1).upper()}{m.group(2).zfill(4)}"   # C0603
            key   = f"{m.group(1).lower()}{m.group(2).zfill(4)}"   # c0603
            return (label, key)
        return None

    got = _pick(fp_val)
    if got:
        return got
    got = _pick(dev_val)
    if got:
        return got
    fp_txt = "" if fp_val is None else str(fp_val)
    return (fp_txt, (norm_pkg(fp_txt) or ""))

# ---------------- drawing ----------------
def add_rect_icon(fig, x, y, color, pkg_key, edge="white", opacity=0.9):
    if pkg_key not in PKG_SIZE or pkg_key == "":
        L, W = (1.6, 0.8)
    else:
        L, W = PKG_SIZE[pkg_key]
    halfL, halfW = L/2.0, W/2.0
    fig.add_shape(
        type="rect",
        x0=x-halfL, x1=x+halfL, y0=y-halfW, y1=y+halfW,
        line=dict(color=edge, width=1),
        fillcolor=color, opacity=opacity, layer="above"
    )

def make_figure(*,
                x_raw, y_raw,         # CSV Mid X / Mid Y 원본
                designator,
                pkg_key,
                pkg_label,
                is_ng,
                title: str) -> go.Figure:

    # 1) CSV 원본(툴팁에 그대로 표시)
    x = np.asarray(x_raw, dtype=float)   # Mid X: +값
    y = np.asarray(y_raw, dtype=float)   # Mid Y: -값

    # 2) 화면 표시용 좌표 (위=0, 아래로 내려갈수록 눈금이 내려가도록 y만 부호 뒤집음)
    xs = x
    ys = -y   # 화면에 찍을 때만 뒤집음

    print("[debug] RAW  X min/max:", float(np.nanmin(x)), float(np.nanmax(x)))
    print("[debug] RAW  Y min/max:", float(np.nanmin(y)), float(np.nanmax(y)))
    print("[debug] DISP X min/max:", float(np.nanmin(xs)), float(np.nanmax(xs)))
    print("[debug] DISP Y min/max:", float(np.nanmin(ys)), float(np.nanmax(ys)))

    fig = go.Figure()
    ok = ~is_ng

    # OK
    cd_ok = np.column_stack([designator[ok], pkg_label[ok], x[ok], y[ok]])  # 툴팁: 원본값
    fig.add_trace(go.Scattergl(
        x=xs[ok], y=ys[ok], mode="markers", name=f"OK ({ok.sum()})",
        marker=dict(size=6, opacity=0),
        customdata=cd_ok,
        hovertemplate=(
            "Designator: %{customdata[0]}<br>"
            "Pkg: %{customdata[1]}<br>"
            "X: %{customdata[2]:.3f} mm<br>"    # CSV Mid X (양수)
            "Y: %{customdata[3]:.3f} mm<extra></extra>"  # CSV Mid Y (음수)
        ),
    ))

    # NG
    cd_ng = np.column_stack([designator[is_ng], pkg_label[is_ng], x[is_ng], y[is_ng]])
    fig.add_trace(go.Scattergl(
        x=xs[is_ng], y=ys[is_ng], mode="markers", name=f"NG ({is_ng.sum()})",
        marker=dict(size=6, opacity=0),
        customdata=cd_ng,
        hovertemplate=(
            "Designator: %{customdata[0]}<br>"
            "Pkg: %{customdata[1]}<br>"
            "X: %{customdata[2]:.3f} mm<br>"
            "Y: %{customdata[3]:.3f} mm<extra></extra>"
        ),
    ))

    # 패키지 네모도 '화면 좌표'(xs, ys)에 그림
    for xi, yi, k, ng in zip(xs, ys, pkg_key, is_ng):
        add_rect_icon(fig, xi, yi, "red" if ng else "lime", k)

    # 축 설정: 좌상단(0,0), 오른쪽 +X, 아래로 갈수록 값이 커지게 보이지만(실제 CSV Y는 음수),
    # autorange='reversed'로 0이 위에 오도록 뒤집음
    ymax_disp = float(np.nanmax(-y_raw))   

    step = 5.0  # 눈금 간격
    ticks = np.arange(0.0, np.ceil(ymax_disp/step)*step + 0.1, step)
    ticktext = ["0"] + [f"-{int(v)}" if abs(v - int(v)) < 1e-9 else f"-{v:g}" for v in ticks[1:]]

    fig.update_xaxes( #
        title_text="X (mm; origin at top-left)",
        range=[0.0, float(np.nanmax(xs))],
        showgrid=True, gridwidth=1, gridcolor="#444",
        zeroline=False, linecolor="white", mirror=True,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=ticktext,
        title_text="Y (mm; top=0, down negative)",
        range=[0.0, ymax_disp],           # 위=0, 아래=양수
        autorange="reversed",             # 화면상 위가 작은 값이 되도록 뒤집기. EasyEDA 상에서 우측아래로 디자인하면 필요한 옵션
        showgrid=True, gridwidth=1, gridcolor="#444",
        zeroline=False, linecolor="white", mirror=True,
        scaleanchor="x", scaleratio=1,
        tickformat="~g",
    )

    fig.update_layout(
        title=title, paper_bgcolor="black", plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="black", bordercolor="white", borderwidth=1),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="PnP Plotly Viewer (robust UTF-16 CSV support)")
    ap.add_argument("--save-html", default="pnp_view.html")
    ap.add_argument("--title", default="PnP Viewer (package-sized crosses)")
    ap.add_argument("--layer", default="all", choices=["T","B","all"])
    ap.add_argument("--footprint", help="Filter by footprint substring (e.g., C0603)")
    args = ap.parse_args()

    if not os.path.exists(args.pnp):
        raise SystemExit(f"PnP 파일을 찾을 수 없습니다: {args.pnp}")

    # PnP 읽기/정리
    pnp = normalize_columns(read_table(args.pnp))
    need = ["Designator","Mid X","Mid Y","Footprint"]
    if not all(c in pnp.columns for c in need):
        raise SystemExit(f"필수 컬럼 누락: {need} (현재: {list(pnp.columns)})")

    if args.layer!="all" and "Layer" in pnp.columns:
        pnp = pnp[pnp["Layer"].astype(str).str.upper().str.startswith(args.layer)]
    if args.footprint:
        pnp = pnp[pnp["Footprint"].astype(str).str.contains(args.footprint, case=False, na=False)]
    if len(pnp)==0:
        raise SystemExit("표시할 PnP 데이터가 없습니다.")

    # 좌표(원본) -> 툴팁에도 그대로 사용
    x0 = to_float(pnp["Mid X"]).to_numpy()
    y0 = to_float(pnp["Mid Y"]).to_numpy()

# X는 +, Y는 −가 정상인데 반대로 들어오면 자동 교정
    if np.nanmin(x0) < 0 and np.nanmin(y0) > 0:
        print("[fix] detected swapped columns -> use Mid Y as X, Mid X as Y")
        x_raw, y_raw = y0, x0   # 바꿔서 사용
    else:
        x_raw, y_raw = x0, y0

    print(f"[main] RAW X min/max: {np.nanmin(x_raw)} {np.nanmax(x_raw)}")
    print(f"[main] RAW Y min/max: {np.nanmin(y_raw)} {np.nanmax(y_raw)}")

    # Designator
    des = get_designators_from_pnp(pnp)

    # 패키지 라벨/키
    fp_ser  = pnp["Footprint"].astype("string").fillna("").str.replace("\x00", "", regex=False)
    dev_ser = (pnp["Device"].astype("string").fillna("") if "Device" in pnp.columns
               else pd.Series([""]*len(pnp), dtype="string"))
    pairs = [extract_pkg_from_values(fp_ser.iat[i], dev_ser.iat[i]) for i in range(len(pnp))]
    pkg_label = np.array([p[0] for p in pairs], dtype=object)   # 표기용 (C0603/R0805...)
    pkg_key   = np.array([p[1] for p in pairs], dtype=object)   # 크기용 (c0603/r0805...)

    # defects -> NG mask
    is_ng = np.zeros(len(des), dtype=bool)
    if args.defects and os.path.exists(args.defects):
        defs_raw = read_table(args.defects)
        df = normalize_columns(defs_raw.copy())
        if "Pred" in df.columns:
            pred = pd.to_numeric(df["Pred"], errors="coerce").fillna(0).astype(int)
            if "Designator" in df.columns:
                dcol = df["Designator"].astype(str)
            elif "File" in df.columns:
                dcol = df["File"].astype(str)
            else:
                dcol = pd.Series([], dtype=str)
            pred_map = {}
            for d, p in zip(dcol, pred):
                d_std = canonical_designator(str(d))
                if d_std:
                    pred_map[d_std] = max(pred_map.get(d_std, 0), int(p))
            is_ng = np.array([pred_map.get(d, 0) == 1 for d in des], dtype=bool)
            print(f"[debug] defects rows: {len(df)}, unique des in defects: {len(pred_map)}")
            print(f"[debug] matched NG on board: {is_ng.sum()} / {len(des)}")
        else:
            print("[debug] 'Pred' 컬럼을 찾지 못했습니다. normalize_columns 매핑을 확인하세요.")
    elif args.defects:
        print(f"[debug] defects path not found: {args.defects}")

    # 그림 생성 (원본 좌표만 사용)
    fig = make_figure(
        x_raw=x_raw, y_raw=y_raw,   # <- 교정된 원본 그대로 전달
        designator=des,
        pkg_key=pkg_key,
        pkg_label=pkg_label,
        is_ng=is_ng,
        title=args.title,
    )

    out = args.save_html if args.save_html.endswith(".html") else args.save_html + ".html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"저장 완료: {out}")

if __name__ == "__main__":
    main()
