# pnp_html.py
# - pnp_visualize.py의 로직을 그대로 옮겨와서
#   GUI에서 사용할 수 있도록 함수화(build_boardmap_html)만 추가/정리
# - 초기에는 모든 부품을 중립색(회색)으로 표시
# - PySide6에서 JS 함수 PNP.setState("R75", 1) 로 실시간 색상 갱신 가능
# - Plotly는 inline 포함(오프라인에서도 동작)

import os, re, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64
from PIL import Image


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
    s = s.str.replace(",",".", regex=False)       # 소수점 콤마 -> 점
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
    # 추가(20251105): LED 및 IC 계열 패키지
    "led0603": (1.6, 0.8), "led0805": (2.0, 1.25), "led1206": (3.2, 1.6),
    "sop8": (4.9, 3.8), "soic8": (4.9, 3.8),
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
    """Footprint 또는 Device 문자열에서 C0603/R0805/LED0805/SOP-8 등 패키지명을 추출.
       반환: (표시용 라벨(대문자), 크기계산용 키(소문자) or "")"""
    def _pick(s):
        if not s:
            return None
        s = str(s).strip()

        # 1) 일반 SMD (R/C/LED0603 등)
        m = re.search(r'([A-Z]+)\s*0?(\d{3,4})', s, flags=re.I)
        if m:
            label = f"{m.group(1).upper()}{m.group(2).zfill(4)}"   # 예: LED0805
            key   = f"{m.group(1).lower()}{m.group(2).zfill(4)}"   # 예: led0805
            return (label, key)

        # 2) IC 패키지 (SOP-8, SOIC-8 등) 20251105 추가
        m = re.search(r'(sop|soic|tssop|qfn|qfp)[\-\s_]?(\d+)', s, flags=re.I)
        if m:
            label = f"{m.group(1).upper()}-{m.group(2)}"
            key   = f"{m.group(1).lower()}{m.group(2)}"            # 예: sop8
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
    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    y_min = float(np.nanmin(ys))
    y_max = float(np.nanmax(ys))
    pad = 5.0  # 주변 여백(mm) – 필요하면 줄이거나 늘려도 됨

     # 2) X축은 양끝 1%를 outlier 로 보고 잘라서 "가운데만" 보이게
    xs_valid = xs[np.isfinite(xs)]
    if len(xs_valid):
        q1, q99 = np.percentile(xs_valid, [1, 99])  # 1% ~ 99%
        pad = 5.0                                   # 양쪽 여유 (mm)
        x_min = float(q1 - pad)
        x_max = float(q99 + pad)
    else:
        # 혹시라도 이상하면 기존 방식으로 fallback
        x_min = 0.0
        x_max = float(np.nanmax(xs))

    fig.update_xaxes(
        title_text="",
        range=[float(np.nanmin(xs)) - 1, float(np.nanmax(xs)) + 1],
        showgrid=True, gridwidth=1, gridcolor="#444",
        zeroline=False, linecolor="white", mirror=True,
    )
    fig.update_yaxes(
        title_text="",
        range=[float(np.nanmin(ys)) - 1, float(np.nanmax(ys)) + 1],
        autorange="reversed",
        showgrid=True, gridwidth=1, gridcolor="#444",
        zeroline=False, linecolor="white", mirror=True,
        scaleanchor="x", scaleratio=1,
        tickformat="~g",
    )

    fig.update_layout(
        title=title,                  # 제목은 그대로 두고 싶으면 유지
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=False,
        margin=dict(l=5, r=5, t=20, b=5),
    )
    
    return fig

# ---------------- public API ----------------
def build_boardmap_html(*,
                        pnp_path: str,
                        out_html: str,
                        title: str,
                        color_ok: str,
                        color_ng: str,
                        bg_image_path: str = None,
                        color_neutral: str) -> str:
                        


    if not os.path.exists(pnp_path):
        raise FileNotFoundError(f"PnP 파일을 찾을 수 없습니다: {pnp_path}")

    pnp = normalize_columns(read_table(pnp_path))
    need = ["Designator", "Mid X", "Mid Y", "Footprint"]
    if not all(c in pnp.columns for c in need):
        # Footprint가 없으면 우리가 추출해둔 pkg라벨로 대체
        if "Footprint" not in pnp.columns and "Device" in pnp.columns:
            pnp["Footprint"] = pnp["Device"]
        else:
            raise RuntimeError(f"필수 컬럼 누락: {need} (현재: {list(pnp.columns)})")

    x0 = to_float(pnp["Mid X"]).to_numpy()
    y0 = to_float(pnp["Mid Y"]).to_numpy()

    if np.nanmin(x0) < 0 and np.nanmin(y0) > 0:
        x_raw, y_raw = y0, x0
    else:
        x_raw, y_raw = x0, y0

    des = get_designators_from_pnp(pnp)

    fp_ser = pnp["Footprint"].astype("string").fillna("").str.replace("\x00", "", regex=False)
    dev_ser = (pnp["Device"].astype("string").fillna("")
            if "Device" in pnp.columns
            else pd.Series([""] * len(pnp), dtype="string"))
    pairs = [extract_pkg_from_values(fp_ser.iat[i], dev_ser.iat[i]) for i in range(len(pnp))]
    pkg_label = np.array([p[0] for p in pairs], dtype=object)
    pkg_key = np.array([p[1] for p in pairs], dtype=object)

 
    # ------------------------------------------------

    fig = make_figure(
        x_raw=x_raw,
        y_raw=y_raw,
        designator=des,
        pkg_key=pkg_key,
        pkg_label=pkg_label,
        title=title,
        neutral_color=color_neutral,
    )

     # -------------------------------------------------
    #  배경 보드 이미지 붙이기 (1mm 당 19.85px 기준) (2025/11/06 추가)
    # -------------------------------------------------
    bg_b64 = None
    board_mm_w = None
    board_mm_h = None

    if bg_image_path is not None and os.path.exists(bg_image_path):
        try:
            # 1) 이미지 픽셀 크기 읽기
            img = Image.open(bg_image_path)
            px_w, px_h = img.size   # (width, height) in pixels

            # EasyEDA Png Export를 통해 추출된 이미지와 mm 간 축적계산값
            px_per_mm = 19.85
            board_mm_w = px_w / px_per_mm
            board_mm_h = px_h / px_per_mm

            # 2) base64 로 인코딩해서 HTML 안에 embed
            with open(bg_image_path, "rb") as f:
                bg_b64 = base64.b64encode(f.read()).decode("ascii")

            print(f"[CHK] using board bg: {bg_image_path}, "
                  f"{px_w}x{px_h}px -> {board_mm_w:.2f} x {board_mm_h:.2f} mm")
        except Exception as e:
            print(f"[WARN] bg load failed: {bg_image_path} ({e})")
            bg_b64 = None

    # 3) 실제로 Plotly 그림에 배경을 깔기
    if bg_b64 is not None and board_mm_w is not None and board_mm_h is not None:
        # (0,0)을 보드 왼쪽 위라고 가정하고, mm 단위로 전체 보드를 덮게 함
        fig.add_layout_image(
            dict(
                source="data:image/png;base64," + bg_b64,
                xref="x",
                yref="y",
                x=0.0,                  # 왼쪽
                y=0.0,                  # 위쪽
                sizex=board_mm_w,       # 가로 길이 (mm)
                sizey=board_mm_h,       # 세로 길이 (mm)
                sizing="stretch",
                layer="below",          # 소자 박스보다 뒤에 배경이미지를 배치
                name="pcb_bg"
            )
        )

        # 축을 보드 전체 크기에 맞춰서 잡기
        fig.update_xaxes(
            range=[0.0, board_mm_w],
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            range=[0.0, board_mm_h],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )

    index = {str(d): i for i, d in enumerate(des)}

    html = fig.to_html(include_plotlyjs="inline", full_html=True)

    inject = f"""
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
(function() {{
  // 색 정의
  const OK = "{color_ok}";
  const NG = "{color_ng}";
  const NEUTRAL = "{color_neutral}";
  const PLOT_DIV_SELECTOR = "div.plotly-graph-div";

  // 나중에 plotly_click에서 쓸 Qt 객체
  let qtBoard = null;

  // 전역 PNP 객체
  window.PNP = {{
    indexByDes: {json.dumps(index)},
    stateByDes: {{}},     // des -> "ok" | "ng" | "neutral"
    bgVisible: true,   //  배경 현재 상태 저장

    setState: function(des, pred) {{
      try {{
        if (!des) return;
        const key = String(des).toUpperCase();
        const idx = this.indexByDes[key];
        if (idx === undefined) return;

        const div = document.querySelector(PLOT_DIV_SELECTOR);
        if (!div || !window.Plotly) return;

        const col = (Number(pred) === 1) ? NG : OK;
        const st  = (Number(pred) === 1) ? "ng" : "ok";
        this.stateByDes[key] = st;

        const patch = {{}};
        patch[`shapes[${{idx}}].fillcolor`] = col;
        window.Plotly.relayout(div, patch);
      }} catch (err) {{
        console.log("[PNP.setState] error:", err);
      }}
    }},

    resetAll: function() {{
      try {{
        const div = document.querySelector(PLOT_DIV_SELECTOR);
        if (!div || !window.Plotly) return;

        const patch = {{}};
        for (const [des, idx] of Object.entries(this.indexByDes)) {{
          patch[`shapes[${{idx}}].fillcolor`] = NEUTRAL;
        }}
        window.Plotly.relayout(div, patch);

        // 상태도 초기화
        this.stateByDes = {{}};
      }} catch (err) {{
        console.log("[PNP.resetAll] error:", err);
      }}
    }},
    
    setBgVisible: function(on) {{
      try {{
        const div = document.querySelector(PLOT_DIV_SELECTOR);
        if (!div || !window.Plotly || !div._fullLayout) return;

        const images = div._fullLayout.images || [];
        const opacity = on ? 1.0 : 0.0;
        const patch = {{}}; 

        
        for (let i = 0; i < images.length; i++) {{
            const im = images[i];
            if (im.name === "pcb_bg") {{
                patch[`images[${{i}}].opacity`] = opacity;}}
    }}
    
        window.Plotly.relayout(div, patch);
        this.bgVisible = vis;
      }} catch (err) {{
        console.log("[PNP.setBgVisible] error:", err);
      }}
    }}
  }};

  // 여기서 Qt WebChannel 초기화
  //   Python 쪽에서 ui_main.py 안에서
  //     channel.registerObject("qtBoard", self._board_bridge)
  //   해놨으니까 이름은 그대로 "qtBoard"
  new QWebChannel(qt.webChannelTransport, function(channel) {{
    qtBoard = channel.objects.qtBoard;
    // console.log("Qt board connected:", !!qtBoard);
  }});

  // Plotly div가 만들어진 뒤에 hover / click 붙이기
  function setupPlotEvents() {{
    const div = document.querySelector(PLOT_DIV_SELECTOR);
    if (!div || !window.Plotly) {{
      // 아직 안 만들어졌으면 조금 있다가 다시
      setTimeout(setupPlotEvents, 300);
      return;
    }}

    // 마우스 올렸을 때 툴팁 색상 바꾸기
    div.on('plotly_hover', function(ev) {{
      try {{
        const pt = ev.points && ev.points[0];
        if (!pt || !pt.customdata) return;
        const des = String(pt.customdata[0] || "").toUpperCase();
        const st  = (window.PNP.stateByDes && window.PNP.stateByDes[des]) || "neutral";

        // 기본 파랑
        let bg = "rgba(120,160,255,0.85)";
        if (st === "ok") {{
          bg = "rgba(120,255,120,0.85)";     // 연두
        }} else if (st === "ng") {{
          bg = "rgba(255,120,120,0.85)";     // 연빨
        }}

        window.Plotly.relayout(div, {{
          "hoverlabel.bgcolor": bg,
          "hoverlabel.font.color": "black"
        }});
      }} catch (err) {{
        console.log("[hover] error:", err);
      }}
    }});

    // 보드 클릭 -> Qt 로 보내기
    div.on('plotly_click', function(ev) {{
      try {{
        const pt = ev.points && ev.points[0];
        if (!pt || !pt.customdata) return;
        const des = String(pt.customdata[0] || "").toUpperCase();

        // 여기서 Python 슬롯 호출
        if (qtBoard && typeof qtBoard.onBoardClick === "function") {{
          qtBoard.onBoardClick(des);
        }}
      }} catch (err) {{
        console.log("[click] error:", err);
      }}
    }});
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", setupPlotEvents);
  }} else {{
    setupPlotEvents();
  }}
}})();
</script>
"""
    html = html.replace("</body>", inject + "</body>")



    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    return out_html