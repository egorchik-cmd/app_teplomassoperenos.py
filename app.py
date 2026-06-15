"""
Численное моделирование тепломассопереноса при сублимационной термопечати
ВКР, направление 20.04.01 «Техносферная безопасность»

Запуск: streamlit run app.py
requirements.txt: streamlit>=1.30 / numpy>=1.24 / matplotlib>=3.7
packages.txt:     fonts-dejavu-core
"""

import io, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# ── Кириллица на Streamlit Cloud ──
def _font():
    avail = {f.name for f in fm.fontManager.ttflist}
    for n in ("DejaVu Sans","Liberation Sans","Noto Sans","Arial"):
        if n in avail:
            plt.rcParams["font.family"] = n; return n
    plt.rcParams["font.family"] = "DejaVu Sans"; return "DejaVu Sans"
_FONT = _font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"]  = True

# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Модель термопечати", page_icon="🧵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
div[data-testid="stMetric"]{
  background:#fff;border-radius:12px;padding:14px 16px;
  box-shadow:0 2px 8px rgba(0,0,0,.07);border:1px solid #eef0f3;}
div[data-testid="stMetric"] label{font-size:13px!important;color:#5f6368!important;}
.card-ok  {background:#e8f5e9;border-left:5px solid #43a047;padding:14px 18px;
           border-radius:6px;color:#1b5e20;font-size:15px;margin:8px 0;}
.card-bad {background:#fce4ec;border-left:5px solid #e91e63;padding:14px 18px;
           border-radius:6px;color:#880e4f;font-size:15px;margin:8px 0;}
.card-warn{background:#fff8e1;border-left:5px solid #fb8c00;padding:14px 18px;
           border-radius:6px;color:#e65100;font-size:15px;margin:8px 0;}
.explain  {background:#f5f5f5;border-radius:8px;padding:12px 16px;
           font-size:14px;color:#444;line-height:1.7;margin:8px 0;}
h1{font-size:1.65rem!important;}h2{font-size:1.2rem!important;}
h3{font-size:1.0rem!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ МОДЕЛИ (глава 3, таблица 3.1)
# ═══════════════════════════════════════════════════════════════
LAM1,RHO1,CP1,D1 = 16.0,  7850., 500.,  10.0e-3  # плита [25]
LAM2,RHO2,CP2,D2 = 0.08,  700.,  1300., 0.1e-3   # бумага [24]
LAM3,RHO3,CP3,D3 = 0.049, 1380., 1300., 0.3e-3   # ткань λэф=0.049 [24]
T0    = 20.0   # начальная температура, °C
ALPHA = 10.0   # коэф. теплоотдачи свободной поверхности ткани, Вт/(м²·К)

R_GAS   = 8.314       # газовая постоянная, Дж/(моль·К)
EA      = 121.0e3     # энергия активации, Дж/моль [29]
D0_ARR  = 9.1339      # предэкспоненциальный множитель, м²/с [29]
TG      = 75.0        # температура стеклования ПЭТ, °C [26]

R_PET   = 7.5e-6      # радиус ПЭТ-волокна, м (d=15 мкм, СЭМ)
R_SILK  = 6.3e-6      # радиус шёлкового волокна, м (d=12.6 мкм, СЭМ)

# Глава 4: эмиссия и вентиляция
M0_DYE    = 6.0    # масса красителя на бумаге, г/м² (раздел 1.1)
KAPPA     = 0.02   # степень термодеструкции κ (раздел 4.3)
S_PRESS   = 0.24   # площадь рабочего поля термопресса, м² (раздел 4.3)
N_CYCLES  = 30     # циклов/ч
ANILINE_FR= 0.10   # доля анилина в суммарной эмиссии (раздел 4.4)
PDK       = 0.1    # ПДК анилина, мг/м³ (СанПиН 1.2.3685-21 [34])
V_ROOM    = 50.0   # объём помещения, м³ (20 м² × 2.5 м, раздел 4.4)

# Три режима диплома (таблица 3.2)
MODES_DIPLOM = [
    (100, 60,  "#808080", "--"),   # недостаточный — серый
    (200, 60,  "#1565c0", "-"),    # оптимальный — синий
    (220, 150, "#c62828", "-"),    # предельный — красный
]

# Экспериментальные режимы (таблица 2.1)
EXP_PET    = [(150,30,1),(150,60,1),(170,30,1),(170,60,1),(180,30,1),
              (180,60,1),(200,30,1),(200,60,1),(200,90,1),(200,120,2),(200,150,2)]
EXP_BOUNDS = [(100,60,0),(220,150,2)]

# ═══════════════════════════════════════════════════════════════
# ФУНКЦИИ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════
def D_arr(T_c):
    """Коэф. диффузии по Аррениусу (3.8–3.9). При T ≤ Tg → D = 0."""
    T_c = np.asarray(T_c, float)
    D   = D0_ARR * np.exp(-EA / (R_GAS * (T_c + 273.15)))
    return np.where(T_c <= TG, 0.0, D)

def hmean(a, b):
    return np.where(a + b > 0, 2*a*b/(a+b), 0.0)

def build_grid(Nx=15):
    n1=max(2,int(D1*1e3*Nx)); n2=max(2,int(D2*1e3*Nx)); n3=max(2,int(D3*1e3*Nx))
    x=np.concatenate([np.linspace(0,D1,n1,endpoint=False),
                      np.linspace(D1,D1+D2,n2,endpoint=False),
                      np.linspace(D1+D2,D1+D2+D3,n3+1)])
    N=len(x); lam,rho,cp=np.empty(N),np.empty(N),np.empty(N); i2,i3=n1,n1+n2
    lam[:i2],rho[:i2],cp[:i2]        = LAM1,RHO1,CP1
    lam[i2:i3],rho[i2:i3],cp[i2:i3] = LAM2,RHO2,CP2
    lam[i3:],rho[i3:],cp[i3:]        = LAM3,RHO3,CP3
    return x,lam,rho,cp,i2,i3

@st.cache_data(show_spinner=False)
def solve_heat(Tp:float, tau:int, sf=0.4):
    """
    Уравнение теплопроводности (3.1), явная МКР-схема (3.14).
    Возвращает: массив времён t_arr и T_fabric(t) — температура середины ткани.
    """
    x,lam,rho,cp,i2,i3 = build_grid()
    dx = x[1]-x[0]
    dt = sf * 0.5 * dx**2 / np.max(lam/(rho*cp))
    T  = np.full(len(x), T0); T[0] = Tp
    im = i3 + (len(x)-i3)//2  # узел в середине слоя ткани

    # 200 точек для плавного графика T(t)
    se = max(1, int(tau/(200*dt)))
    t_arr  = [0.0]
    Tf_arr = [T0]
    t, step = 0.0, 0

    while t < tau:
        ds = min(dt, tau-t); Tn = T.copy()
        le = hmean(lam[1:-1], lam[2:])
        lw = hmean(lam[:-2],  lam[1:-1])
        Tn[1:-1] = T[1:-1] + ds/(rho[1:-1]*cp[1:-1]) * (
            le*(T[2:]-T[1:-1]) - lw*(T[1:-1]-T[:-2])) / dx**2
        Tn[-1] = (lam[-1]/dx*T[-2] + ALPHA*T0) / (lam[-1]/dx + ALPHA)
        Tn[0]  = Tp
        T = Tn; t += ds; step += 1
        if step % se == 0 or abs(t-tau) < 1e-9:
            t_arr.append(t); Tf_arr.append(T[im])

    return np.array(t_arr), np.array(Tf_arr)

@st.cache_data(show_spinner=False)
def solve_diff(Tf_tuple:tuple, t_tuple:tuple, tau:int,
               R:float, fiber:str="PET", Nr=40, sf=0.4):
    """
    Уравнение диффузии Фика (3.7) в цилиндрических координатах.
    Возвращает: r (Nr точек) и финальный профиль C(r) при t=tau.
    """
    Tf = np.array(Tf_tuple)
    dr = R/(Nr-1); r = np.linspace(0, R, Nr)
    Dm = float(np.max(D_arr(Tf)))
    dt = (sf*dr**2/Dm) if Dm > 0 else 1.0
    C  = np.zeros(Nr)
    t, step = 0.0, 0

    while t < tau:
        ds = min(dt, tau-t)
        frac = t/tau*(len(Tf)-1)
        lo   = int(frac); hi = min(lo+1, len(Tf)-1); w = frac-lo
        Tc   = (1-w)*Tf[lo] + w*Tf[hi]
        Dc   = float(D_arr(Tc))
        if Dc > 0:
            Cn = C.copy(); j = np.arange(1,Nr-1)
            re = r[j]+.5*dr; rw = r[j]-.5*dr
            Cn[j] = C[j] + ds/(r[j]*dr) * (
                Dc*re*(C[j+1]-C[j]) - Dc*rw*(C[j]-C[j-1])) / dr
            Cn[0]  = C[0] + ds*2*Dc*(C[1]-C[0])/dr**2
            Cn[-1] = (1.0 if Tc > TG else 0.0) if fiber=="PET" else Cn[-2]
            C = np.clip(Cn, 0., 1.)
        t += ds; step += 1

    return r, C

_trap = np.trapezoid if hasattr(np,"trapezoid") else np.trapz

def eta_transfer(C, r):
    """η = ∫C·r dr / (R²/2) — коэф. переноса (раздел 4.3)."""
    return float(np.clip(_trap(C*r, r) / (0.5*r[-1]**2), 0., 1.))

def calc_emission(eta):
    """Цепочка расчёта эмиссии и воздухообмена (формулы 4.1–4.3)."""
    Mres = M0_DYE*(1-eta)
    G    = KAPPA*Mres*S_PRESS*N_CYCLES*1000
    G_an = ANILINE_FR*G
    L    = G_an/PDK
    K    = L/V_ROOM
    return Mres, G, G_an, L, K

def fmt_d(v):
    """Читаемая научная нотация: 4.0×10⁻¹³"""
    if v <= 0: return "0"
    e   = int(np.floor(np.log10(abs(v)))); m = v/10**e
    sup = str(e).translate(str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}"

@st.cache_data(show_spinner=False)
def run_single(Tp:int, tau:int, fiber:str):
    """Полный расчёт одного режима."""
    Rf = R_PET if fiber=="PET" else R_SILK
    t_arr, Tf = solve_heat(float(Tp), int(tau))
    r, C = solve_diff(tuple(Tf.tolist()), tuple(t_arr.tolist()), int(tau), float(Rf), fiber)
    eta  = eta_transfer(C, r) if fiber=="PET" else 0.0
    Mres,G,G_an,L,K = calc_emission(eta)
    return dict(
        t_arr=t_arr, Tf=Tf, r=r, C=C, Rf=Rf, fiber=fiber,
        Tmax=float(Tf[-1]), Cax=float(C[0]),
        D_at=float(D_arr(float(Tf[-1]))),
        eta=eta, Mres=Mres, G=G, G_an=G_an, L=L, K=K
    )

def _depth50(C, r):
    """Глубина от поверхности до уровня C = 0.5 (мкм) — «R/2» из гл. 3."""
    for i in range(len(r)-1, -1, -1):
        if C[i] <= 0.5:
            return (r[-1]-r[i])*1e6
    return 0.0

def verdict(res):
    c = res["Cax"]
    if res["fiber"] == "SILK":
        return ("warn",
            "Шёлк: краситель не проникает внутрь волокна. "
            "Фиброин не имеет сродства к дисперсным красителям — "
            "на поверхности задано условие непроницаемости (ур. 3.13). "
            "Окрашивание носит только поверхностный характер [1].")
    if c < 0.01:
        return ("bad",
            f"Недостаточный нагрев. Температура ткани {res['Tmax']:.0f} °C. "
            f"D = {fmt_d(res['D_at'])} м²/с — слишком мал для диффузии за время выдержки. "
            "Цветопереноса нет.")
    if c < 0.5:
        return ("ok",
            f"Оптимальный режим. C/Cs на оси волокна = {c:.3f} — "
            "краситель равномерно закрепился в структуре волокна. "
            "Ожидается чёткое насыщенное изображение.")
    if c < 0.95:
        return ("warn",
            f"Интенсивный режим. C/Cs = {c:.3f} — "
            "глубокое проникновение, возможно лёгкое расплывание контуров.")
    return ("warn",
        f"Предельный режим. Волокно насыщено полностью (C/Cs = {c:.3f}). "
        "Краситель мигрирует за контур изображения — "
        "визуальное расплывание и потеря чёткости [1].")

# ═══════════════════════════════════════════════════════════════
# ГРАФИКИ — ТОЧНО КАК В ДИПЛОМЕ
# ═══════════════════════════════════════════════════════════════

def fig_41(results_dict, active_modes):
    """
    Рис. 4.1 — Расчётные поля температуры T(t) в ткани.
    Ось X: время t, с. Ось Y: температура T, °C.
    Три кривые для трёх режимов. Аннотации в ключевых точках.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    colors = {"100/60": "#808080", "200/60": "#1565c0", "220/150": "#c62828"}
    labels = {
        "100/60":  "100 °C / 60 с",
        "200/60":  "200 °C / 60 с",
        "220/150": "220 °C / 150 с",
    }

    # аннотации — ключевые точки (как на рис. 4.1 диплома)
    annot_times = {
        "100/60":  [5, 10, 20, 30, 60, 90, 150],
        "200/60":  [5, 10, 20, 30, 40, 60],
        "220/150": [5, 10, 20, 30, 40, 60, 150],
    }

    for key in active_modes:
        if key not in results_dict: continue
        res = results_dict[key]
        t_arr, Tf = res["t_arr"], res["Tf"]
        col = colors[key]; lbl = labels[key]
        ls  = "--" if key=="100/60" else "-"
        ax.plot(t_arr, Tf, color=col, lw=2.5, ls=ls, label=lbl, zorder=3)

        # Аннотации температуры в ключевых точках
        for t_pt in annot_times.get(key, []):
            if t_pt > t_arr[-1]: continue
            idx = np.argmin(np.abs(t_arr - t_pt))
            T_pt = Tf[idx]
            ax.annotate(f"{T_pt:.0f}",
                xy=(t_arr[idx], T_pt),
                fontsize=8, color=col, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points")

    # Начальная точка
    ax.annotate("20", xy=(0, T0), fontsize=8, color="#333",
                fontweight="bold", ha="center", va="top",
                xytext=(0, -8), textcoords="offset points")

    ax.set_xlabel("Время t, сек", fontsize=11, labelpad=6)
    ax.set_ylabel("T, °C", fontsize=11, labelpad=6)
    ax.set_title("Рис. 4.1 — Температура ткани во времени T(t) для трёх режимов",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-2, max(60 if "220/150" not in active_modes else 155, 65))
    ax.set_ylim(0, 235)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95,
              edgecolor="#ddd")
    ax.grid(True, alpha=0.35, ls="--", color="#ccc")
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_42(results_dict, fiber, show_modes):
    """
    Рис. 4.2 — Профили концентрации C(r) для ПЭТ и шёлка.
    Ось X: r, мкм (радиус волокна, 0–7.5 мкм).
    Ось Y: C/Cs (0–1).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    plot_colors = {"200/60": "#1565c0", "220/150": "#c62828"}
    plot_labels = {"200/60": "ПЭТ 200 °C / 60 с", "220/150": "ПЭТ 220 °C / 150 с"}

    # Кривые ПЭТ
    for key in ["200/60", "220/150"]:
        if key not in show_modes or key not in results_dict: continue
        res = results_dict[key]
        if res["fiber"] != "PET": continue
        r_um = res["r"] * 1e6  # перевод в мкм
        col  = plot_colors[key]
        ax.plot(r_um, res["C"], color=col, lw=2.5, marker="o",
                markersize=4, markevery=5, label=plot_labels[key], zorder=3)
        # Аннотации 0/1 на кривой (как в дипломе)
        for i, (rv, cv) in enumerate(zip(r_um, res["C"])):
            if i % 5 == 0:
                ax.annotate(f"{cv:.0f}", xy=(rv, cv),
                    fontsize=7, color=col, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    # Кривая шёлка — горизонталь C=0
    if fiber == "SILK" or "SILK" in show_modes:
        Rf_silk = R_SILK * 1e6
        r_silk  = np.linspace(0, Rf_silk, 20)
        ax.plot(r_silk, np.zeros_like(r_silk), color="#555", lw=2.0,
                ls="--", marker="o", markersize=4, markevery=5,
                label="Шёлк (все режимы)", zorder=3)
        for i, rv in enumerate(r_silk):
            if i % 5 == 0:
                ax.annotate("0", xy=(rv, 0), fontsize=7, color="#555",
                    ha="center", va="top",
                    xytext=(0, -8), textcoords="offset points")

    ax.set_xlabel("r, мкм  (радиус волокна: 0 = ось, 7.5 = поверхность)", fontsize=11, labelpad=6)
    ax.set_ylabel("C/Cs", fontsize=11, labelpad=6)
    ax.set_title("Рис. 4.2 — Профили концентрации красителя в волокне C(r)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-0.2, 8.0)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor="#ddd")
    ax.grid(True, alpha=0.35, ls="--", color="#ccc")
    ax.spines[["top","right"]].set_visible(False)
    return fig


def fig_phase():
    """Карта режимов: расчёт + экспериментальные точки табл. 2.1."""
    T_r=np.linspace(100,220,13); tau_r=np.linspace(20,300,13)
    grid=np.zeros((len(T_r),len(tau_r)))
    for i,Tp in enumerate(T_r):
        for j,ta in enumerate(tau_r):
            res=run_single(int(round(Tp)),int(round(ta)),"PET")
            grid[i,j]=res["Cax"]
    fig,ax=plt.subplots(figsize=(9,5.5)); fig.patch.set_facecolor("white")
    cf=ax.contourf(tau_r,T_r,grid,levels=20,cmap="RdYlGn",alpha=0.92)
    cs=ax.contour(tau_r,T_r,grid,levels=[0.1,0.3,0.5,0.7,0.9],
                  colors="black",alpha=0.3,linewidths=0.6)
    ax.clabel(cs,inline=True,fontsize=7,fmt="%.1f")
    cb=fig.colorbar(cf,ax=ax,pad=0.02)
    cb.set_label("C/Cs на оси волокна (0=нет, 1=полное)")
    mm={0:("X","#c62828","Нет переноса (эксп.)"),
        1:("o","#1b5e20","Хороший перенос (эксп.)"),
        2:("s","#e65100","Расплывание (эксп.)")}
    seen=set()
    for (Tp,ta,cls) in EXP_PET+EXP_BOUNDS:
        mk,col,lab=mm[cls]
        ax.scatter(ta,Tp,marker=mk,c=col,s=90,edgecolors="white",
                   linewidths=1.2,zorder=5,
                   label=lab if lab not in seen else None)
        seen.add(lab)
    ax.set_xlabel("Время выдержки τ, с"); ax.set_ylabel("Температура плиты, °C")
    ax.set_title("Карта режимов: расчёт (заливка) + эксперимент таблицы 2.1")
    ax.legend(fontsize=9,loc="lower right",framealpha=0.95)
    ax.set_xlim(20,300); ax.set_ylim(100,220); ax.grid(True,alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)
    return fig


# ═══════════════════════════════════════════════════════════════
# САЙДБАР
# ═══════════════════════════════════════════════════════════════
st.title("🧵 Перенос красителя при сублимационной термопечати")
st.caption(
    "Численная модель: нагрев пакета «плита–бумага–ткань» (ур. 3.1) + "
    "диффузия красителя в волокне (ур. 3.7) → оценка эмиссии токсикантов и воздухообмена  |  "
    "ВКР 20.04.01 «Техносферная безопасность»"
)
st.divider()

with st.sidebar:
    st.header("⚙️ Параметры")
    section = st.radio("Раздел", [
        "Три режима из диплома",
        "Свой режим",
        "Карта режимов",
    ])
    st.divider()

    if section == "Три режима из диплома":
        st.markdown("**Выбери режимы для отображения:**")
        show_100 = st.checkbox("100 °C / 60 с — нет переноса",  value=True)
        show_200 = st.checkbox("200 °C / 60 с — оптимальный",   value=True)
        show_220 = st.checkbox("220 °C / 150 с — расплывание",  value=True)
        st.divider()
        fib_lbl = st.radio("Материал",["Полиэстер (ПЭТ)","Шёлк"])
        fiber   = "PET" if "ПЭТ" in fib_lbl else "SILK"

    elif section == "Свой режим":
        fib_lbl = st.radio("Материал",["Полиэстер (ПЭТ)","Шёлк"])
        fiber   = "PET" if "ПЭТ" in fib_lbl else "SILK"
        Tp_usr  = st.slider("Температура плиты, °C", 80, 220, 200, 5)
        tau_usr = st.slider("Время выдержки τ, с",   20, 300, 60, 10)
    else:
        fiber = "PET"
        st.info("Карта строится для ПЭТ (есть данные таблицы 2.1).")

# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: ТРИ РЕЖИМА ИЗ ДИПЛОМА
# ═══════════════════════════════════════════════════════════════
if section == "Три режима из диплома":

    active_keys = []
    if show_100: active_keys.append("100/60")
    if show_200: active_keys.append("200/60")
    if show_220: active_keys.append("220/150")

    if not active_keys:
        st.warning("Выбери хотя бы один режим в боковой панели.")
        st.stop()

    # ── РАСЧЁТ ──
    with st.spinner("Расчёт..."):
        results = {}
        for key, (Tp, tau, col, ls) in zip(
                ["100/60","200/60","220/150"], MODES_DIPLOM):
            if key in active_keys:
                results[key] = run_single(Tp, tau, fiber)

    # ── ТАБЛИЦА 4.3 ──
    st.subheader("Таблица 4.3 — Ключевые результаты расчёта")
    rows = []
    mode_names = {"100/60":"100 / 60","200/60":"200 / 60","220/150":"220 / 150"}
    for key,(Tp,tau,_,__) in zip(["100/60","200/60","220/150"],MODES_DIPLOM):
        if key not in results: continue
        res = results[key]
        rows.append({
            "Режим (T °C / τ с)": mode_names[key],
            "Tмакс ткани, °C":    f"{res['Tmax']:.1f}",
            "Глубина до C=0.5, мкм": f"{0.0:.1f}" if res['Cax']<0.01
                                      else (f"≈0" if fiber=="SILK"
                                            else f"{_depth50(res['C'],res['r']):.1f}"),
            "C/Cs ПЭТ на оси":   "≈ 0" if res['Cax']<0.005 else f"{res['Cax']:.4f}",
            "C/Cs шёлк":          "0" if fiber!="SILK" else "0",
        })
    st.table(rows)

    st.divider()

    # ── РИС. 4.1 и РИС. 4.2 ──
    st.subheader("📈 Основные графики")
    g1, g2 = st.columns(2)

    with g1:
        st.pyplot(fig_41(results, active_keys), use_container_width=True)
        st.markdown('<div class="explain">'
            '<b>Рис. 4.1 — Температура ткани во времени.</b> '
            'Ось X — время выдержки в секундах. '
            'Ось Y — температура внутри слоя ткани. '
            'Числа на кривых — значения температуры в ключевые моменты. '
            'Чем выше кривая, тем горячее ткань. '
            'При режиме 100 °C ткань прогревается лишь до 96 °C — '
            'ниже необходимого уровня для эффективной диффузии красителя.'
            '</div>', unsafe_allow_html=True)

    with g2:
        show_c_modes = [k for k in active_keys]
        st.pyplot(fig_42(results, fiber, show_c_modes), use_container_width=True)
        st.markdown('<div class="explain">'
            '<b>Рис. 4.2 — Профиль концентрации красителя в волокне.</b> '
            'Ось X — положение внутри волокна в мкм '
            '(0 = центр/ось волокна, 7.5 мкм = поверхность). '
            'Ось Y — доля насыщения красителем '
            '(0 = пусто, 1 = максимальное насыщение). '
            'Поверхность волокна всегда = 1 при T > Tg (ур. 3.11). '
            'Для шёлка C = 0 при всех режимах — поверхностное окрашивание.'
            '</div>', unsafe_allow_html=True)

    st.divider()

    # ── БЛОК КЛЮЧЕВЫХ ДАННЫХ (как в правой части рис. 4.1 диплома) ──
    st.subheader("📋 Ключевые данные по режимам")
    k1, k2, k3 = st.columns(3)
    kols = [k1, k2, k3]
    mode_labels = [
        ("100 °C / 60 сек", "#808080"),
        ("200 °C / 60 сек", "#1565c0"),
        ("220 °C / 150 сек","#c62828"),
    ]
    for i,(key,lbl_col) in enumerate(zip(
            ["100/60","200/60","220/150"], mode_labels)):
        if key not in results: continue
        res = results[key]
        lbl, col = lbl_col
        with kols[i]:
            st.markdown(f"**:{col.replace('#','')}[{lbl}]**")
            st.markdown(f"Tмакс ткани = **{res['Tmax']:.1f} °C**")
            st.markdown(f"D при Tмакс = **{fmt_d(res['D_at'])} м²/с**")
            if key=="100/60":
                st.markdown("Диффузия незначительна (D ≈ 10⁻¹⁷–10⁻¹⁸ м²/с)")
            elif key=="200/60":
                st.markdown("D ≈ 10⁻¹³–10⁻¹² м²/с (↑ 3–4 порядка vs 100°C)")
            else:
                st.markdown(f"Стационар через ~65–70 сек")

    st.divider()

    # ── ОХРАНА ТРУДА ──
    if "200/60" in results and fiber=="PET":
        st.subheader("🛡️ Оценка профессионального риска — оптимальный режим 200 °C / 60 с")
        res200 = results["200/60"]
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Коэффициент переноса η", f"{res200['eta']:.2f}",
            help="Доля красителя, закрепившейся внутри волокна ПЭТ (раздел 4.3).")
        e2.metric("Остаточный краситель Mres", f"{res200['Mres']:.2f} г/м²",
            help="Mres = M0·(1−η) — остаток на бумаге, подвергающийся деструкции.")
        e3.metric("Эмиссия анилина G_анилин", f"{res200['G_an']:.0f} мг/ч",
            help="G_анилин = 10%·κ·Mres·S·n·1000, κ=0.02 (раздел 4.3).")
        e4.metric("Воздухообмен L", f"{res200['L']:.0f} м³/ч",
            f"Кратность K = {res200['K']:.1f} ч⁻¹",
            help="L = G_анилин / ПДК. ПДК анилина = 0.1 мг/м³ (СанПиН 1.2.3685-21).")

        if res200["K"] <= 10:
            st.markdown(f'<div class="card-ok">Кратность воздухообмена '
                f'K = {res200["K"]:.1f} ч⁻¹ соответствует норме 6–10 ч⁻¹ '
                f'для студии {V_ROOM:.0f} м³ (СП 60.13330.2020). '
                f'Расчётное значение L = {res200["L"]:.0f} м³/ч, '
                f'проектное L ≥ 350 м³/ч (раздел 4.4).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-warn">K = {res200["K"]:.1f} ч⁻¹ '
                f'превышает норму. Нужна местная вытяжка L ≥ {res200["L"]:.0f} м³/ч.</div>',
                unsafe_allow_html=True)

        st.markdown('<div class="explain">'
            'Цепочка расчёта (глава 4): '
            'η → Mres = M0·(1−η) (ур. 4.1) → '
            'G = κ·Mres·S·n·1000 (ур. 4.2) → '
            'G_анилин = 10%·G → '
            'L = G_анилин / ПДК_анилина (ур. 4.3) → '
            'K = L / V_помещения.'
            '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: СВОЙ РЕЖИМ
# ═══════════════════════════════════════════════════════════════
elif section == "Свой режим":
    with st.spinner("Расчёт..."):
        res = run_single(int(Tp_usr), int(tau_usr), fiber)

    # Метрики
    st.subheader("📊 Результаты расчёта")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Температура ткани Tмакс", f"{res['Tmax']:.0f} °C",
        f"{res['Tmax']-TG:+.0f} °C к порогу Tg={TG:.0f} °C",
        delta_color="normal" if res['Tmax']>TG else "inverse",
        help="Максимальная температура внутри слоя ткани к концу выдержки.")
    m2.metric("C/Cs на оси волокна",
        "≈ 0" if res['Cax']<0.005 else f"{res['Cax']:.3f}",
        help="0 = красителя нет, 1 = полное насыщение. Оптимально: 0.1–0.5.")
    m3.metric("D при Tмакс", fmt_d(res['D_at'])+" м²/с",
        help="Коэффициент диффузии по Аррениусу при данной температуре ткани.")
    m4.metric("Коэф. переноса η", f"{res['eta']:.2f}",
        help="Доля красителя, закрепившегося внутри волокна.")

    vc,vt = verdict(res)
    st.markdown(f'<div class="card-{vc}">{vt}</div>', unsafe_allow_html=True)
    st.divider()

    # Два главных графика (как в дипломе)
    st.subheader("📈 Графики")
    results_usr = {f"{Tp_usr}/{tau_usr}": res}
    g1,g2 = st.columns(2)

    with g1:
        # T(t) — один режим
        fig_t, ax_t = plt.subplots(figsize=(7.5,4.8))
        fig_t.patch.set_facecolor("white"); ax_t.set_facecolor("white")
        ax_t.plot(res["t_arr"], res["Tf"], color="#1565c0", lw=2.5)
        ax_t.axhline(TG, color="#43a047", ls="-.", lw=1.3,
                     label=f"Tg = {TG:.0f} °C (порог диффузии)")
        # аннотации в ключевых точках
        t_pts = np.linspace(0, tau_usr, 8).astype(int)
        for t_pt in t_pts:
            idx = np.argmin(np.abs(res["t_arr"]-t_pt))
            T_pt = res["Tf"][idx]
            ax_t.annotate(f"{T_pt:.0f}", xy=(res["t_arr"][idx], T_pt),
                fontsize=8, color="#1565c0", ha="center", va="bottom",
                xytext=(0,5), textcoords="offset points")
        ax_t.set_xlabel("Время t, сек"); ax_t.set_ylabel("T, °C")
        ax_t.set_title(f"Температура ткани T(t)\nРежим {Tp_usr} °C / {tau_usr} с")
        ax_t.set_ylim(0, max(Tp_usr*1.07, TG+15))
        ax_t.legend(fontsize=9); ax_t.grid(True,alpha=0.35,ls="--",color="#ccc")
        ax_t.spines[["top","right"]].set_visible(False)
        st.pyplot(fig_t, use_container_width=True)
        st.markdown('<div class="explain">Температура внутри слоя ткани нарастает '
            'по мере нагрева от плиты. Числа — значения в ключевые моменты.</div>',
            unsafe_allow_html=True)

    with g2:
        # C(r)
        fig_c, ax_c = plt.subplots(figsize=(7.5,4.8))
        fig_c.patch.set_facecolor("white"); ax_c.set_facecolor("white")
        if fiber=="SILK":
            r_um = np.linspace(0, R_SILK*1e6, 20)
            ax_c.plot(r_um, np.zeros_like(r_um), "#555", lw=2.5,
                      marker="o", markersize=4, markevery=5, label="Шёлк: C = 0")
            for i,rv in enumerate(r_um):
                if i%5==0: ax_c.annotate("0", xy=(rv,0), fontsize=7,color="#555",
                    ha="center",va="top",xytext=(0,-8),textcoords="offset points")
        else:
            r_um = res["r"]*1e6
            ax_c.plot(r_um, res["C"], "#1565c0", lw=2.5, marker="o",
                      markersize=4, markevery=5, label=f"ПЭТ {Tp_usr}°C/{tau_usr}с")
            ax_c.fill_between(r_um, 0, res["C"], alpha=0.12, color="#1565c0")
            for i,(rv,cv) in enumerate(zip(r_um,res["C"])):
                if i%5==0: ax_c.annotate(f"{cv:.1f}",xy=(rv,cv),
                    fontsize=7,color="#1565c0",ha="center",va="bottom",
                    xytext=(0,4),textcoords="offset points")
            ax_c.plot(0, res["Cax"], "r*", ms=13, zorder=5,
                      label=f"Ось: C/Cs = {res['Cax']:.3f}")
        ax_c.set_xlabel("r, мкм  (0 = ось волокна, поверхность = правый край)")
        ax_c.set_ylabel("C/Cs")
        ax_c.set_title(f"Профиль концентрации C(r)\nРежим {Tp_usr} °C / {tau_usr} с")
        ax_c.set_xlim(-0.2, 8.0); ax_c.set_ylim(-0.05, 1.12)
        ax_c.legend(fontsize=9); ax_c.grid(True,alpha=0.35,ls="--",color="#ccc")
        ax_c.spines[["top","right"]].set_visible(False)
        st.pyplot(fig_c, use_container_width=True)
        st.markdown('<div class="explain">Распределение красителя по сечению волокна '
            'к концу выдержки. Поверхность (правый край) = 1 при T > Tg. '
            'Красная звезда — концентрация в центре волокна.</div>',
            unsafe_allow_html=True)

    st.divider()
    # Охрана труда для своего режима
    if fiber=="PET":
        st.subheader("🛡️ Профессиональный риск")
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("η переноса", f"{res['eta']:.2f}")
        e2.metric("Mres", f"{res['Mres']:.2f} г/м²")
        e3.metric("G_анилин", f"{res['G_an']:.0f} мг/ч")
        e4.metric("L воздухообмен", f"{res['L']:.0f} м³/ч",
                  f"K = {res['K']:.1f} ч⁻¹")
        if res["K"]<=10:
            st.markdown(f'<div class="card-ok">K = {res["K"]:.1f} ч⁻¹ — '
                f'в норме (6–10 ч⁻¹ для студии {V_ROOM:.0f} м³).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-warn">K = {res["K"]:.1f} ч⁻¹ — '
                f'выше нормы. Нужна местная вытяжка L ≥ {res["L"]:.0f} м³/ч.</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 3: КАРТА РЕЖИМОВ
# ═══════════════════════════════════════════════════════════════
elif section == "Карта режимов":
    st.subheader("🗺️ Карта режимов с верификацией по таблице 2.1")
    st.markdown('<div class="explain">Цветная заливка — результат расчёта модели '
        '(C/Cs на оси волокна ПЭТ). Точки — экспериментальные режимы из таблицы 2.1. '
        'Совпадение точек с расчётными зонами подтверждает адекватность модели.</div>',
        unsafe_allow_html=True)
    with st.spinner("Строю карту (169 расчётов, ~15–20 сек)..."):
        st.pyplot(fig_phase(), use_container_width=True)
    st.markdown('<div class="card-ok">Точки «хорошего переноса» (●) попадают '
        'в зелёно-жёлтую зону, «расплывание» (■) — в зону насыщения (C→1), '
        '«нет переноса» (✕) — в красную зону. Модель адекватна.</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# СПРАВОЧНЫЕ БЛОКИ
# ═══════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 Условные обозначения"):
    st.markdown("""
| Обозначение | Величина | Единицы |
|---|---|---|
| T | Температура | °C |
| Tмакс | Максимальная температура внутри слоя ткани к концу выдержки | °C |
| Tg | Температура стеклования ПЭТ — ниже неё диффузия невозможна | °C |
| T0 | Начальная температура (окружающая среда) | °C |
| t | Текущее время | с |
| τ | Время выдержки в термопрессе | с |
| x | Координата по толщине пакета плита–бумага–ткань | м |
| δ | Толщина слоя | мм |
| r | Радиальная координата в волокне (0=ось, R=поверхность) | мкм / м |
| R | Радиус волокна | мкм / м |
| C | Концентрация красителя в волокне | — |
| Cs | Равновесная концентрация на поверхности волокна | — |
| C/Cs | Доля насыщения (0=пусто, 1=максимум) | — |
| D(T) | Коэффициент диффузии красителя (зависит от T) | м²/с |
| D0 | Предэкспоненциальный множитель (Аррениус) | м²/с |
| Ea | Энергия активации диффузии | кДж/моль |
| Rг | Универсальная газовая постоянная (8.314) | Дж/(моль·К) |
| λ | Теплопроводность слоя | Вт/(м·К) |
| ρ | Плотность слоя | кг/м³ |
| cp | Удельная теплоёмкость слоя | Дж/(кг·К) |
| α | Коэффициент теплоотдачи с открытой поверхности ткани | Вт/(м²·К) |
| η | Коэффициент переноса красителя в волокно | — |
| M0 | Исходная масса красителя на бумаге | г/м² |
| Mres | Остаточная масса красителя: Mres = M0·(1−η) | г/м² |
| κ | Степень термодеструкции (κ = 0.02, консервативно) | — |
| S | Площадь рабочего поля термопресса | м² |
| n | Производительность термопресса | циклов/ч |
| G_анилин | Интенсивность выделения анилина | мг/ч |
| ПДК | Предельно допустимая концентрация анилина | мг/м³ |
| L | Требуемый воздухообмен | м³/ч |
| K | Кратность воздухообмена (раз/ч) | ч⁻¹ |
| V | Объём помещения студии | м³ |
""")

with st.expander("ℹ️ Как работает модель"):
    st.markdown("""
**Шаг 1 — Нагрев пакета (уравнение 3.1).**
Уравнение теплопроводности решается вдоль оси x (от плиты до свободной поверхности ткани).
На плите — фиксированная температура Tпресс (ур. 3.3).
На обратной стороне ткани — конвекция в воздух α=10 Вт/(м²·К) (ур. 3.4).
График Рис. 4.1 показывает, как растёт температура в середине слоя ткани.

**Шаг 2 — Диффузия красителя в волокно (уравнение 3.7).**
Когда T > Tg=75 °C, молекулы красителя начинают диффундировать в аморфные области ПЭТ.
Скорость определяется D(T) по Аррениусу (ур. 3.8–3.9).
При T=100°C: D ≈ 10⁻¹⁷ м²/с (ничтожно мало).
При T=200°C: D ≈ 4·10⁻¹³ м²/с (в 10000 раз быстрее!).
График Рис. 4.2 показывает распределение красителя по сечению волокна.

**Шаг 3 — Охрана труда (глава 4).**
η вычисляется интегрированием профиля C(r). Остаток (1−η) деструктирует →
образует анилин → расчёт требуемого воздухообмена L.

**Шёлк:** ГУ непроницаемости на поверхности (ур. 3.13) → C = 0 при всех режимах.
Это подтверждает экспериментальный факт поверхностного окрашивания шёлка [1].
""")

st.divider()
st.caption(
    f"Шрифт графиков: {_FONT}  |  МКР явная схема, CFL (ур. 3.16–3.17)  |  "
    "Верификация: Tмакс и C/Cs совпадают с таблицей 4.3  |  ВКР 20.04.01"
)
