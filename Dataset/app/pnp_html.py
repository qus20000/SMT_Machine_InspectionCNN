# pnp_html.py
# - 네가 주었던 pnp_visualize.py의 로직을 그대로 옮겨와서
#   GUI에서 사용할 수 있도록 함수화(build_boardmap_html)만 추가/정리
# - 초기에는 모든 부품을 중립색(회색)으로 표시
# - PySide6에서 JS 함수 PNP.setState("R75", 1) 로 실시간 색상 갱신 가능
# - Plotly는 inline 포함(오프라인에서도 동작)

import os, re, json
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

    # UTF-16/널문자 탐지
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff") or head.count(b"\x00") > 50:
        df = try_read(("utf-16", "utf-16-le", "utf-16-be"), ("\t", ",", ";", "|"))
        if df is not None:
            return df

    # 일반 CSV
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
        "midx":"Mid X","x":"Mid X","centerx":"Mid X","posx":"Mid X","xmm":"Mid X","centerxmm":"Mid X",
        "midy":"Mid Y","y":"Mid Y","centery":"Mid Y","posy":"Mid Y","ymm":"Mid Y","centerymm":"Mid Y",
        "footprint":"Footprint","package":"Footprint",
        # 기타 흔한 칼럼들도 보존(필수는 아님)
        "device":"Device","layer":"Layer",
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
    s = s.str.replace("\u2212","-", regex=False)  # 유니코드 마이너스
    s = s.str.replace(",",".", regex=False)       # 소수점 콤마 → 점
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
        # print(f"[debug] using index as designator (ratio={idx_ratio:.2f})")
        return s_idx.to_numpy()
    else:
        # print(f"[debug] using column as designator: {dcol} (ratio={col_ratio:.2f})")
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

def extract_pkg_from_values(fp_val: str, dev_val: str):
    """Footprint 또는 Device 문자열에서 C0603/R0805 같은 패키지명을 추출.
       반환: (표시용 라벨(대문자), 크기계산용 키(소문자) or "")"""
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
                title: str,
                neutral_color: str) -> go.Figure:

    # 1) CSV 원본(툴팁에 그대로 표시)
    x = np.asarray(x_raw, dtype=float)   # Mid X: +값
    y = np.asarray(y_raw, dtype=float)   # Mid Y: -값

    # 2) 화면 표시용 좌표 (위=0, 아래로 내려갈수록 눈금이 내려가도록 y만 부호 뒤집음)
    xs = x
    ys = -y   # 화면에 찍을 때만 뒤집음

    fig = go.Figure()

    # hover를 위해 투명 포인트 한 레이어 깔아둠
    cd_all = np.column_stack([designator, pkg_label, x, y])
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers", name="ALL",
        marker=dict(size=6, opacity=0),
        customdata=cd_all,
        hovertemplate=(
            "Designator: %{customdata[0]}<br>"
            "Pkg: %{customdata[1]}<br>"
            "X: %{customdata[2]:.3f} mm<br>"
            "Y: %{customdata[3]:.3f} mm<extra></extra>"
        ),
    ))

    # 도형(패키지 네모)은 초기엔 전부 중립색
    for xi, yi, k in zip(xs, ys, pkg_key):
        add_rect_icon(fig, xi, yi, neutral_color, k)

    # 축 설정
    ymax_disp = float(np.nanmax(-y_raw))
    step = 5.0
    ticks = np.arange(0.0, np.ceil(ymax_disp/step)*step + 0.1, step)
    ticktext = ["0"] + [f"-{int(v)}" if abs(v - int(v)) < 1e-9 else f"-{v:g}" for v in ticks[1:]]

    fig.update_xaxes(
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
        range=[0.0, ymax_disp],
        autorange="reversed",
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

# ---------------- public API ----------------
def build_boardmap_html(*,
                        pnp_path: str,
                        out_html: str,
                        title: str,
                        color_ok: str,
                        color_ng: str,
                        color_neutral: str) -> str:
    """
    GUI에서 호출. 초기에는 color_neutral로 그리며,
    JS 측에서 PNP.setState('R75', 1) 식으로 OK/NG 갱신.
    """
    if not os.path.exists(pnp_path):
        raise FileNotFoundError(f"PnP 파일을 찾을 수 없습니다: {pnp_path}")

    # PnP 읽기/정리
    pnp = normalize_columns(read_table(pnp_path))
    need = ["Designator","Mid X","Mid Y","Footprint"]
    if not all(c in pnp.columns for c in need):
        raise RuntimeError(f"필수 컬럼 누락: {need} (현재: {list(pnp.columns)})")

    # 좌표(원본)
    x0 = to_float(pnp["Mid X"]).to_numpy()
    y0 = to_float(pnp["Mid Y"]).to_numpy()

    # X는 +, Y는 −가 정상인데 반대로 들어오면 자동 교정
    if np.nanmin(x0) < 0 and np.nanmin(y0) > 0:
        # print("[fix] detected swapped columns -> use Mid Y as X, Mid X as Y")
        x_raw, y_raw = y0, x0
    else:
        x_raw, y_raw = x0, y0

    # Designator
    des = get_designators_from_pnp(pnp)

    # 패키지 라벨/키
    fp_ser  = pnp["Footprint"].astype("string").fillna("").str.replace("\x00", "", regex=False)
    dev_ser = (pnp["Device"].astype("string").fillna("") if "Device" in pnp.columns
               else pd.Series([""]*len(pnp), dtype="string"))
    pairs = [extract_pkg_from_values(fp_ser.iat[i], dev_ser.iat[i]) for i in range(len(pnp))]
    pkg_label = np.array([p[0] for p in pairs], dtype=object)
    pkg_key   = np.array([p[1] for p in pairs], dtype=object)

    # 그림 생성 (초기 전부 회색)
    fig = make_figure(
        x_raw=x_raw, y_raw=y_raw,
        designator=des, pkg_key=pkg_key, pkg_label=pkg_label,
        title=title, neutral_color=color_neutral,
    )

    # designator -> shape index 매핑 (shapes는 fig.layout.shapes 순서)
    index = {str(d): i for i, d in enumerate(des)}

    # HTML 생성 (Plotly를 inline으로 포함: WebEngine 오프라인 호환)
    html = fig.to_html(include_plotlyjs="inline", full_html=True)

    # 라이브 업데이트용 JS 삽입
    inject = f"""
<script>
(function(){{
  const OK="{color_ok}";
  const NG="{color_ng}";
  window.PNP = {{
    indexByDes: {json.dumps(index)},
    setState: function(des, pred) {{
      if (!des) return;
      const key = String(des).toUpperCase();
      const i = this.indexByDes[key];
      if (i === undefined) return;
      const col = (Number(pred) === 1) ? NG : OK;
      const patch = {{}};
      patch[`shapes[${{i}}].fillcolor`] = col;
      Plotly.relayout(document.querySelector('div.plotly-graph-div'), patch);
    }}
  }};
}})();
</script>
"""
    html = html.replace("</body>", inject + "</body>")

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
