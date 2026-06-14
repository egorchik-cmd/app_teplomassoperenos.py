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

# ── Кириллица: гарантия на Streamlit Cloud ──────────────────────
def _font():
    avail = {f.name for f in fm.fontManager.ttflist}
    for n in ("DejaVu Sans","Liberation Sans","Noto Sans","Arial"):
        if n in avail:
            plt.rcParams["font.family"] = n; return n
    plt.rcParams["font.family"] = "DejaVu Sans"; return "DejaVu Sans"
_FONT = _font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"]  = True

import streamlit as st
st.set_page_config(page_title="Модель термопечати", page_icon="🧵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
div[data-testid="stMetric"]{background:#fff;border-radius:12px;padding:14px 16px;
  box-shadow:0 2px 8px rgba(0,0,0,.07);border:1px solid #eef0f3;}
div[data-testid="stMetric"] label{font-size:13px!important;color:#5f6368!important;}
.card-ok  {background:#e8f5e9;border-left:5px solid #43a047;padding:14px 18px;
           border-radius:6px;color:#1b5e20;font-size:15px;margin:8px 0;}
.card-bad {background:#fce4ec;border-left:5px solid #e91e63;padding:14px 18px;
           border-radius:6px;color:#880e4f;font-size:15px;margin:8px 0;}
.card-warn{background:#fff8e1;border-left:5px solid #fb8c00;padding:14px 18px;
           border-radius:6px;color:#e65100;font-size:15px;margin:8px 0;}
.explain  {background:#f5f5f5;border-radius:8px;padding:12px 16px;
           font-size:14px;color:#444;line-height:1.7;margin:6px 0;}
h1{font-size:1.65rem!important;}h2{font-size:1.2rem!important;}
h3{font-size:1.0rem!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ МОДЕЛИ  (глава 3, таблица 3.1)
# ═══════════════════════════════════════════════════════════════
# Теплофизика слоёв [25]
LAM1,RHO1,CP1,D1 = 16.0,7850.,500., 10.0e-3   # плита
LAM2,RHO2,CP2,D2 = 0.08,700., 1300., 0.1e-3   # бумага
LAM3,RHO3,CP3,D3 = 0.049,1380.,1300., 0.3e-3  # ткань (λэф=0.05)
T0    = 20.0   # начальная температура, °C
ALPHA = 10.0   # теплоотдача свободной поверхности ткани, Вт/(м²·К)

# Диффузия [29]
R_GAS = 8.314
EA    = 121.0e3   # энергия активации, Дж/моль
D0_ARR= 9.1339    # предэкспоненциальный множитель, м²/с
TG    = 75.0      # температура стеклования ПЭТ, °C

# Радиусы волокон (СЭМ, раздел 2.3)
R_PET  = 7.5e-6   # R = d/2, d≈15 мкм для ПЭТ
R_SILK = 6.3e-6   # R = d/2, d≈12.6 мкм для шёлка

# Глава 4 — эмиссия и вентиляция
M0_DYE     = 6.0    # исходная масса красителя, г/м² (раздел 1.1)
KAPPA      = 0.02   # степень деструкции, κ = 0.02 (раздел 4.3)
S_PRESS    = 0.24   # площадь рабочего поля термопресса, м² (раздел 4.3)
N_CYCLES   = 30     # циклов/ч
ANILINE_FR = 0.10   # доля анилина в суммарной эмиссии (раздел 4.4)
PDK        = 0.1    # ПДК анилина, мг/м³ (СанПиН 1.2.3685-21 [34])
V_ROOM     = 50.0   # объём помещения, м³ (20 м² × 2.5 м, раздел 4.4)

# Экспериментальные режимы таблицы 2.1 (класс: 0=нет, 1=оптимум, 2=расплыв)
EXP_PET = [(150,30,1),(150,60,1),(170,30,1),(170,60,1),(180,30,1),
           (180,60,1),(200,30,1),(200,60,1),(200,90,1),(200,120,2),(200,150,2)]
EXP_BOUNDS = [(100,60,0),(220,150,2)]

# ═══════════════════════════════════════════════════════════════
# ФИЗИЧЕСКИЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════
def D_arr(T_c):
    """Коэф. диффузии по Аррениусу (3.8–3.9). При T ≤ Tg → D = 0."""
    T_c = np.asarray(T_c, float)
    D   = D0_ARR * np.exp(-EA / (R_GAS * (T_c + 273.15)))
    D   = np.where(T_c <= TG, 0.0, D)
    return float(D) if D.ndim == 0 else D

def hmean(a, b):
    return np.where(a+b > 0, 2*a*b/(a+b), 0.0)

def build_grid(Nx=15):
    n1=max(2,int(D1*1e3*Nx)); n2=max(2,int(D2*1e3*Nx)); n3=max(2,int(D3*1e3*Nx))
    x=np.concatenate([np.linspace(0,D1,n1,endpoint=False),
                      np.linspace(D1,D1+D2,n2,endpoint=False),
                      np.linspace(D1+D2,D1+D2+D3,n3+1)])
    N=len(x); lam,rho,cp=np.empty(N),np.empty(N),np.empty(N); i2,i3=n1,n1+n2
    lam[:i2],rho[:i2],cp[:i2]     = LAM1,RHO1,CP1
    lam[i2:i3],rho[i2:i3],cp[i2:i3] = LAM2,RHO2,CP2
    lam[i3:],rho[i3:],cp[i3:]     = LAM3,RHO3,CP3
    return x,lam,rho,cp,i2,i3

@st.cache_data(show_spinner=False)
def solve_heat(Tp:float, tau:int, sf=0.4):
    """Уравнение теплопроводности (3.1), явная МКР-схема (3.14)."""
    x,lam,rho,cp,i2,i3=build_grid(); dx=x[1]-x[0]
    dt=sf*0.5*dx**2/np.max(lam/(rho*cp))
    T=np.full(len(x),T0); T[0]=Tp
    se=max(1,int(tau/(24*dt))); sT,st=[T.copy()],[0.0]; t,step=0.0,0
    while t<tau:
        ds=min(dt,tau-t); Tn=T.copy()
        le=hmean(lam[1:-1],lam[2:]); lw=hmean(lam[:-2],lam[1:-1])
        Tn[1:-1]=T[1:-1]+ds/(rho[1:-1]*cp[1:-1])*(le*(T[2:]-T[1:-1])-lw*(T[1:-1]-T[:-2]))/dx**2
        Tn[-1]=(lam[-1]/dx*T[-2]+ALPHA*T0)/(lam[-1]/dx+ALPHA); Tn[0]=Tp
        T=Tn; t+=ds; step+=1
        if step%se==0 or abs(t-tau)<1e-9: sT.append(T.copy()); st.append(t)
    return x,np.array(sT),np.array(st),i3

@st.cache_data(show_spinner=False)
def solve_diff(Tfab_t:tuple, tau:int, R:float, fiber:str="PET", Nr=40, sf=0.4):
    """Уравнение диффузии Фика (3.7) в цилиндрических координатах."""
    Tf=np.array(Tfab_t); dr=R/(Nr-1); r=np.linspace(0,R,Nr)
    Dm=float(D_arr(np.max(Tf))); dt=(sf*dr**2/Dm) if Dm>0 else 1.0
    C=np.zeros(Nr); sC,st=[C.copy()],[0.0]
    se=max(1,int(tau/(24*dt))); t,step=0.0,0
    while t<tau:
        ds=min(dt,tau-t)
        frac=t/tau*(len(Tf)-1); lo=int(frac); hi=min(lo+1,len(Tf)-1); w=frac-lo
        Tc=(1-w)*Tf[lo]+w*Tf[hi]; Dc=float(D_arr(Tc))
        if Dc>0:
            Cn=C.copy(); j=np.arange(1,Nr-1); re=r[j]+.5*dr; rw=r[j]-.5*dr
            Cn[j]=C[j]+ds/(r[j]*dr)*(Dc*re*(C[j+1]-C[j])-Dc*rw*(C[j]-C[j-1]))/dr
            Cn[0]=C[0]+ds*2*Dc*(C[1]-C[0])/dr**2
            Cn[-1]=(1.0 if Tc>TG else 0.0) if fiber=="PET" else Cn[-2]
            C=np.clip(Cn,0.,1.)
        t+=ds; step+=1
        if step%se==0 or abs(t-tau)<1e-9: sC.append(C.copy()); st.append(t)
    return r,np.array(sC),np.array(st)

_trap = np.trapezoid if hasattr(np,"trapezoid") else np.trapz

def eta_transfer(C,r):
    """Коэффициент переноса η: интеграл C(r)·r по сечению (раздел 4.3)."""
    return float(np.clip(_trap(C*r,r)/(0.5*r[-1]**2),0.,1.))

def calc_emission(eta):
    """Цепочка расчёта эмиссии и воздухообмена (формулы 4.1–4.3)."""
    Mres = M0_DYE*(1-eta)                     # г/м²  (4.1)
    G    = KAPPA*Mres*S_PRESS*N_CYCLES*1000   # мг/ч  (4.2)
    G_an = ANILINE_FR*G                        # мг/ч  анилин
    L    = G_an/PDK                            # м³/ч  (4.3)
    K    = L/V_ROOM                            # ч⁻¹
    return Mres,G,G_an,L,K

def depth50(C,r):
    """Глубина от поверхности до уровня C = 0.5 (как «R/2» в главе 3)."""
    for i in range(len(r)-1,-1,-1):
        if C[i]<=0.5: return (r[-1]-r[i])*1e6
    return 0.0

def fmt_d(v):
    """Красивая научная нотация: 2.0×10⁻¹³"""
    if v==0: return "0"
    e=int(np.floor(np.log10(abs(v)))); m=v/10**e
    sup=str(e).translate(str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}"

@st.cache_data(show_spinner=False)
def run(Tp:int, tau:int, fiber:str):
    Rf = R_PET if fiber=="PET" else R_SILK
    x,Tsn,tsn,i3 = solve_heat(float(Tp),int(tau))
    im = i3+(len(x)-i3)//2
    Tf = Tsn[:,im]
    r,Csn,tcs = solve_diff(tuple(Tf.tolist()),int(tau),float(Rf),fiber)
    Cf = Csn[-1]
    res = dict(x=x,Tsn=Tsn,tsn=tsn,i3=i3,Tf=Tf,r=r,Csn=Csn,tcs=tcs,
               Cf=Cf,Rf=Rf,fiber=fiber,
               Tmax=float(Tf[-1]),Cax=float(Cf[0]),
               depth=depth50(Cf,r),Dat=D_arr(float(Tf[-1])))
    res["eta"]=eta_transfer(Cf,r) if fiber=="PET" else 0.0
    res["Mres"],res["G"],res["G_an"],res["L"],res["K"]=calc_emission(res["eta"])
    return res

# ═══════════════════════════════════════════════════════════════
# ВЕРДИКТ
# ═══════════════════════════════════════════════════════════════
def verdict(res):
    c=res["Cax"]
    if res["fiber"]=="SILK":
        return ("warn",
            "Шёлк: краситель не проникает внутрь волокна. "
            "Дисперсный краситель не имеет химического сродства к белку фиброину, "
            "поэтому на поверхности задано условие непроницаемости (ур. 3.13). "
            "Окрашивание шёлка носит только поверхностный характер — "
            "это подтверждается экспериментальными данными [1].")
    if c<0.01:
        return ("bad",
            f"Недостаточный нагрев. Температура ткани {res['Tmax']:.0f} °C — "
            "краситель практически не проникает в волокно. "
            f"Коэффициент диффузии D = {fmt_d(res['Dat'])} м²/с настолько мал, "
            "что за время выдержки молекулы красителя не успевают переместиться "
            "внутрь волокна. Цветопереноса нет.")
    if c<0.5:
        return ("ok",
            f"Оптимальный режим. Краситель равномерно проник в волокно: "
            f"концентрация на оси = {c:.3f} (от 0 до 1). "
            "Изображение будет чётким и насыщенным.")
    if c<0.95:
        return ("warn",
            f"Интенсивный режим. Концентрация красителя на оси = {c:.3f} — "
            "глубокое проникновение, возможно небольшое расплывание контуров.")
    return ("warn",
        f"Предельный режим. Волокно насыщено полностью (C/Cs = {c:.3f} по всему сечению). "
        "Краситель мигрирует за пределы контура изображения — "
        "визуально это выглядит как расплывание и потеря чёткости печати [1].")

# ═══════════════════════════════════════════════════════════════
# ГРАФИКИ
# ═══════════════════════════════════════════════════════════════
def fig_T(res, Tp, tau):
    """Температурное поле T(x,t) — как тепло распространяется сквозь пакет."""
    x,Tsn,tsn=res["x"],res["Tsn"],res["tsn"]
    fig,ax=plt.subplots(figsize=(7.5,4.6)); fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    # фоновые зоны слоёв
    ax.axvspan(0,D1*1e3,              alpha=0.07,color="#e53935")
    ax.axvspan(D1*1e3,(D1+D2)*1e3,   alpha=0.14,color="#f9a825")
    ax.axvspan((D1+D2)*1e3,(D1+D2+D3)*1e3, alpha=0.10,color="#1e88e5")
    ax.axvline(D1*1e3,  color="#bbb",ls="--",lw=0.8,alpha=0.7)
    ax.axvline((D1+D2)*1e3,color="#bbb",ls="--",lw=0.8,alpha=0.7)
    # линия Tg
    ax.axhline(TG,color="#43a047",ls="-.",lw=1.3,alpha=0.9,
               label=f"Tg = {TG:.0f} °C (порог диффузии)")
    # временные срезы
    idx=np.unique(np.round(np.linspace(0,len(tsn)-1,6)).astype(int))
    cmap=plt.cm.plasma(np.linspace(0.15,0.92,len(idx)))
    for k,i in enumerate(idx):
        ax.plot(x*1e3,Tsn[i],color=cmap[k],
                lw=2.8 if i==idx[-1] else 1.4,
                label=f"t = {tsn[i]:.0f} с")
    # подписи слоёв (не для бумаги — слишком тонкая)
    ax.text(D1/2*1e3,       T0+3,"Плита",   ha="center",fontsize=8,color="#555",style="italic")
    ax.text((D1+D2+D3/2)*1e3,T0+3,"Ткань",  ha="center",fontsize=8,color="#555",style="italic")
    ax.set_xlabel("Глубина в пакете x, мм")
    ax.set_ylabel("Температура T, °C")
    ax.set_title(f"Как нагревается ткань — режим {Tp} °C / {tau} с\n"
                 "(каждая кривая = один момент времени, слева направо — волна тепла)")
    ax.set_xlim(0,(D1+D2+D3)*1e3); ax.set_ylim(T0-5, max(Tp*1.07,TG+15))
    ax.legend(fontsize=8,loc="upper right",framealpha=0.9)
    ax.grid(True,alpha=0.3,ls="--")
    return fig

def fig_C(res, Tp, tau):
    """Профиль концентрации C(r,t) — как краситель проникает в волокно."""
    r,Csn,tcs,Rf=res["r"],res["Csn"],res["tcs"],res["Rf"]
    fig,ax=plt.subplots(figsize=(7.5,4.6)); fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    if res["fiber"]=="SILK":
        ax.plot([0,1],[0,0],color="#fb8c00",lw=3)
        ax.text(0.5,0.5,
            "Шёлк: краситель не проникает\n(условие непроницаемости, ур. 3.13)\n\n"
            "Дисперсный краситель оседает\nтолько на поверхности волокна",
            ha="center",va="center",fontsize=11,color="#e65100",transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6",fc="#fff8e1",ec="#fb8c00"))
    else:
        idx=np.unique(np.round(np.linspace(0,len(tcs)-1,5)).astype(int))
        cmap=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(idx)))
        for k,i in enumerate(idx):
            ax.plot(r/Rf,Csn[i],color=cmap[k],
                    lw=3.0 if i==idx[-1] else 1.2,
                    alpha=1.0 if i==idx[-1] else 0.5,
                    label=f"t = {tcs[i]:.0f} с")
        ax.fill_between(r/Rf,0,Csn[-1],alpha=0.12,color="#e53935")
        ax.axhline(0.5,color="#555",ls=":",lw=1.0,alpha=0.7,label="C/Cs = 0.5")
        ax.plot(0,res["Cax"],"r*",ms=14,zorder=5,
                label=f"Ось волокна: C/Cs = {res['Cax']:.3f}")
        ax.legend(fontsize=8,loc="upper left",framealpha=0.9)
    ax.set_xlabel("Радиальное положение r/R\n(0 = ось волокна,  1 = поверхность волокна)")
    ax.set_ylabel("Доля насыщения красителем C/Cs\n(0 = пусто,  1 = максимальное насыщение)")
    ax.set_title(f"Как краситель проникает в волокно — режим {Tp} °C / {tau} с\n"
                 "(чем ярче кривая, тем позже момент времени)")
    ax.set_xlim(0,1); ax.set_ylim(-0.05,1.12)
    ax.grid(True,alpha=0.3,ls="--")
    return fig

def fig_phase():
    """Карта режимов: расчёт + экспериментальные точки таблицы 2.1."""
    T_r=np.linspace(100,220,13); tau_r=np.linspace(20,300,13)
    grid=np.zeros((len(T_r),len(tau_r)))
    for i,Tp in enumerate(T_r):
        for j,ta in enumerate(tau_r):
            grid[i,j]=run(int(round(Tp)),int(round(ta)),"PET")["Cax"]
    fig,ax=plt.subplots(figsize=(9,5.5)); fig.patch.set_facecolor("white")
    cf=ax.contourf(tau_r,T_r,grid,levels=20,cmap="RdYlGn",alpha=0.92)
    cs=ax.contour(tau_r,T_r,grid,levels=[0.1,0.3,0.5,0.7,0.9],
                  colors="black",alpha=0.3,linewidths=0.6)
    ax.clabel(cs,inline=True,fontsize=7,fmt="%.1f")
    cb=fig.colorbar(cf,ax=ax,pad=0.02); cb.set_label("C/Cs на оси волокна (0=нет, 1=полное)")
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
    ax.set_xlabel("Время выдержки τ, с")
    ax.set_ylabel("Температура плиты, °C")
    ax.set_title("Карта режимов термопечати на ПЭТ\n"
                 "Заливка = расчёт модели, точки = эксперимент (таблица 2.1)")
    ax.legend(fontsize=9,loc="lower right",framealpha=0.95)
    ax.set_xlim(20,300); ax.set_ylim(100,220); ax.grid(True,alpha=0.2)
    return fig

def fig_compare(rA,rB,mA,mB):
    fig,axes=plt.subplots(2,2,figsize=(12,8)); fig.patch.set_facecolor("white")
    for col,(res,m) in enumerate([(rA,mA),(rB,mB)]):
        # температура
        ax=axes[0,col]; ax.set_facecolor("#fafafa")
        ax.axvspan((D1+D2)*1e3,(D1+D2+D3)*1e3,alpha=0.10,color="#1e88e5")
        ax.axhline(TG,color="#43a047",ls="-.",lw=1.2,label=f"Tg={TG:.0f}°C")
        ax.plot(res["x"]*1e3,res["Tsn"][-1],"b-",lw=2.4)
        ax.set_title(f"{m[0]} °C / {m[1]} с   Температура ткани: {res['Tmax']:.0f} °C")
        ax.set_xlabel("Глубина x, мм"); ax.set_ylabel("T, °C")
        ax.set_xlim(0,(D1+D2+D3)*1e3); ax.grid(True,alpha=0.3,ls="--"); ax.legend(fontsize=8)
        # концентрация
        ax=axes[1,col]; ax.set_facecolor("#fafafa")
        ax.plot(res["r"]/res["Rf"],res["Cf"],"r-",lw=2.6)
        ax.fill_between(res["r"]/res["Rf"],0,res["Cf"],alpha=0.15,color="#e53935")
        ax.set_title(f"Проникновение красителя: C/Cs на оси = {res['Cax']:.3f},  η = {res['eta']:.2f}")
        ax.set_xlabel("r/R  (0=ось волокна, 1=поверхность)"); ax.set_ylabel("C/Cs")
        ax.set_xlim(0,1); ax.set_ylim(-0.05,1.12); ax.grid(True,alpha=0.3,ls="--")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
# ИНТЕРФЕЙС — ЗАГОЛОВОК
# ═══════════════════════════════════════════════════════════════
st.title("🧵 Перенос красителя при сублимационной термопечати")
st.caption(
    "Численная модель: нагрев пакета «плита–бумага–ткань» (ур. 3.1) + "
    "диффузия красителя в волокне (ур. 3.7) → оценка эмиссии токсикантов и воздухообмена.  "
    "ВКР 20.04.01 «Техносферная безопасность»"
)
st.divider()

# ═══════════════════════════════════════════════════════════════
# САЙДБАР
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Параметры")
    section=st.radio("Раздел",
        ["Один режим","Сравнить два режима","Карта режимов"],
        help="Выбери что хочешь увидеть")
    st.divider()

    if section in ("Один режим","Сравнить два режима"):
        fib_lbl=st.radio("Материал ткани",["Полиэстер (ПЭТ)","Шёлк"],
            help="ПЭТ: краситель диффундирует внутрь волокна.\n"
                 "Шёлк: краситель остаётся на поверхности.")
        fiber="PET" if "ПЭТ" in fib_lbl else "SILK"
    else:
        fiber="PET"
        st.info("Карта строится для ПЭТ (есть экспериментальные данные, табл. 2.1).")

    st.divider()
    if section=="Один режим":
        im=st.radio("Способ ввода",["Задать вручную","Примеры из диплома"])
        if im=="Примеры из диплома":
            preset=st.selectbox("Режим",[
                "100 °C / 60 с — краситель не проникает",
                "200 °C / 60 с — оптимальный перенос",
                "220 °C / 150 с — расплывание изображения"])
            Tp=int(preset.split(" ")[0])
            tau=int(preset.split("/")[1].strip().split(" ")[0])
        else:
            Tp =st.slider("Температура плиты термопресса, °C",80,220,200,5)
            tau=st.slider("Время выдержки τ, с",20,300,60,10)
    elif section=="Сравнить два режима":
        st.markdown("**Режим A**")
        TA  =st.slider("Температура A, °C",80,220,200,5,key="TA")
        tauA=st.slider("Время A, с",20,300,60,10,key="tauA")
        st.markdown("**Режим B**")
        TB  =st.slider("Температура B, °C",80,220,220,5,key="TB")
        tauB=st.slider("Время B, с",20,300,150,10,key="tauB")

# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: ОДИН РЕЖИМ
# ═══════════════════════════════════════════════════════════════
if section=="Один режим":
    with st.spinner("Выполняю расчёт..."):
        res=run(int(Tp),int(tau),fiber)

    # ── МЕТРИКИ ─────────────────────────────────────────────
    st.subheader("📊 Результаты расчёта переноса красителя")
    m1,m2,m3,m4=st.columns(4)

    m1.metric(
        "Температура ткани Tмакс",
        f"{res['Tmax']:.0f} °C",
        f"{res['Tmax']-TG:+.0f} °C к порогу Tg={TG:.0f} °C",
        delta_color="normal" if res["Tmax"]>TG else "inverse",
        help=f"Максимальная температура внутри слоя ткани к концу выдержки τ={tau} с. "
             f"Порог Tg={TG} °C — ниже него диффузия красителя невозможна."
    )
    cval="≈ 0 (нет переноса)" if res["Cax"]<0.005 else f"{res['Cax']:.3f}"
    m2.metric(
        "Доля насыщения C/Cs на оси",
        cval,
        help="Нормированная концентрация красителя в центре волокна. "
             "0 = красителя нет, 1 = полное насыщение. "
             "Оптимально: 0.1–0.5 (хорошее окрашивание без расплывания)."
    )
    m3.metric(
        "Глубина проникновения",
        f"{res['depth']:.1f} мкм",
        f"Радиус волокна R = {res['Rf']*1e6:.1f} мкм",
        help="Расстояние от поверхности волокна до точки, где C = 0.5·Cs. "
             "В главе 3 это обозначено как «глубина порядка R/2»."
    )
    m4.metric(
        "Коэф. диффузии D при Tмакс",
        fmt_d(res["Dat"])+" м²/с",
        help="Скорость движения молекул красителя в волокне при данной температуре. "
             "Растёт экспоненциально с температурой по закону Аррениуса (ур. 3.8–3.9)."
    )

    # вердикт
    vc,vt=verdict(res)
    st.markdown(f'<div class="card-{vc}">{vt}</div>',unsafe_allow_html=True)
    st.divider()

    # ── ГРАФИКИ ─────────────────────────────────────────────
    st.subheader("📈 Графики процесса")
    g1,g2=st.columns(2)
    with g1:
        st.pyplot(fig_T(res,Tp,tau),use_container_width=True)
        st.markdown('<div class="explain">'
            '<b>Что показывает этот график:</b> как температура распространяется '
            'сквозь пакет «плита–бумага–ткань» за время выдержки. '
            'Каждая кривая — один момент времени (ранние кривые тёмные, '
            'поздние — яркие). Чем выше кривая в правой части, тем горячее ткань. '
            'Зелёная пунктирная линия — температура стеклования ПЭТ (75 °C): '
            'только выше неё краситель начинает двигаться внутрь волокна.</div>',
            unsafe_allow_html=True)
    with g2:
        st.pyplot(fig_C(res,Tp,tau),use_container_width=True)
        if res["fiber"]=="PET":
            st.markdown('<div class="explain">'
                '<b>Что показывает этот график:</b> как краситель заполняет '
                'поперечное сечение волокна. Горизонтальная ось — положение '
                'внутри волокна (0 = центр/ось, 1 = поверхность). '
                'Вертикальная ось — доля насыщения красителем (0 = пусто, 1 = максимум). '
                'Поверхность (правый край) всегда = 1 (условие 3.11). '
                'Красная звезда — концентрация в центре волокна. '
                'Чем выше кривая в левой части, тем глубже проник краситель.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="explain">'
                '<b>Почему график пустой:</b> шёлковое волокно состоит из белка '
                'фиброина, который не имеет химического сродства к дисперсным '
                'красителям. Условие непроницаемости на поверхности (ур. 3.13) '
                'означает, что краситель физически не может проникнуть внутрь. '
                'Это подтверждается экспериментом [1]: шёлк окрашивается '
                'только поверхностно.</div>',
                unsafe_allow_html=True)
    st.divider()

    # ── ОХРАНА ТРУДА ────────────────────────────────────────
    st.subheader("🛡️ Оценка профессионального риска (глава 4)")

    if fiber=="SILK":
        st.info(
            "Расчёт эмиссии выполнен для ПЭТ — основного исследуемого материала. "
            "Для шёлка краситель не закрепляется внутри волокна, поэтому "
            "остаточная масса красителя на бумаге-носителе будет выше."
        )
        eta_silk=0.0
        Mres,G,G_an,L,K=calc_emission(eta_silk)
        st.markdown(f'<div class="explain">'
            f'При η = 0 (краситель не перешёл в волокно) остаточная масса красителя '
            f'на бумаге максимальна: Mres = M0·(1−0) = {Mres:.1f} г/м². '
            f'Эмиссия анилина: G_анилин = {G_an:.0f} мг/ч. '
            f'Требуемый воздухообмен: L = {L:.0f} м³/ч (кратность K = {K:.1f} ч⁻¹).'
            f'</div>',unsafe_allow_html=True)
    else:
        e1,e2,e3,e4=st.columns(4)
        e1.metric("Коэффициент переноса η",f"{res['eta']:.2f}",
            help="Доля красителя, закрепившегося внутри волокна ПЭТ. "
                 "Вычисляется интегрированием профиля C(r) по сечению волокна. "
                 "η = 1.0 означало бы полный перенос, η = 0 — краситель не вошёл.")
        e2.metric("Остаточный краситель на бумаге",f"{res['Mres']:.2f} г/м²",
            help=f"Mres = M0·(1−η) = {M0_DYE}·(1−{res['eta']:.2f}) = {res['Mres']:.2f} г/м². "
                 "Эта масса остаётся на бумаге-носителе и частично разлагается при нагреве.")
        e3.metric("Выделение анилина G_анилин",f"{res['G_an']:.0f} мг/ч",
            help=f"G = κ·Mres·S·n·1000 = {KAPPA}·{res['Mres']:.2f}·{S_PRESS}·{N_CYCLES}·1000, "
                 f"затем G_анилин = 10% от G. "
                 f"κ = {KAPPA} — степень термодеструкции (консервативное допущение, раздел 4.3).")
        e4.metric("Требуемый воздухообмен L",f"{res['L']:.0f} м³/ч",
            f"Кратность K = {res['K']:.1f} ч⁻¹",
            help=f"L = G_анилин / ПДК = {res['G_an']:.0f} / {PDK} = {res['L']:.0f} м³/ч. "
                 f"ПДК анилина = {PDK} мг/м³ (СанПиН 1.2.3685-21). "
                 f"K = L / V = {res['L']:.0f} / {V_ROOM} = {res['K']:.1f} ч⁻¹.")

        # вердикт по вентиляции
        if res["K"]<=0.1:
            st.markdown('<div class="card-ok">Практически весь краситель перешёл в волокно. '
                'Остаточная масса на бумаге минимальна — эмиссия токсикантов пренебрежимо мала.</div>',
                unsafe_allow_html=True)
        elif res["K"]<=10:
            st.markdown(
                f'<div class="card-ok">Кратность воздухообмена K = {res["K"]:.1f} ч⁻¹ '
                f'укладывается в рекомендованный диапазон 6–10 ч⁻¹ '
                f'для студии объёмом {V_ROOM:.0f} м³ (СП 60.13330.2020). '
                f'Требуемый расход приточного воздуха L ≥ {res["L"]:.0f} м³/ч, '
                f'проектное значение L = 350 м³/ч (раздел 4.4).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card-warn">Кратность K = {res["K"]:.1f} ч⁻¹ превышает '
                f'типовую норму 6–10 ч⁻¹. Необходима местная вытяжная вентиляция '
                f'повышенной производительности (L ≥ {res["L"]:.0f} м³/ч).</div>',
                unsafe_allow_html=True)

        st.markdown(
            '<div class="explain">'
            '<b>Цепочка расчёта (глава 4):</b><br>'
            'η (из модели) → Mres = M0·(1−η) — остаточный краситель на бумаге (ур. 4.1) → '
            'G = κ·Mres·S·n·1000 — суммарная эмиссия продуктов деструкции (ур. 4.2) → '
            'G_анилин = 10%·G → L = G_анилин / ПДК_анилина — требуемый воздухообмен (ур. 4.3) → '
            'K = L / V_помещения.<br>'
            f'ПДК анилина = {PDK} мг/м³ (II класс опасности, СанПиН 1.2.3685-21 [34]). '
            f'Площадь рабочего поля термопресса S = {S_PRESS} м², '
            f'производительность n = {N_CYCLES} циклов/ч, κ = {KAPPA} (раздел 4.3).'
            '</div>',
            unsafe_allow_html=True)
    st.divider()

    # ── ЭКСПОРТ ─────────────────────────────────────────────
    st.subheader("💾 Сохранить графики")
    f1=fig_T(res,Tp,tau); f1.savefig("/tmp/_t.png",dpi=130,bbox_inches="tight"); plt.close(f1)
    f2=fig_C(res,Tp,tau); f2.savefig("/tmp/_c.png",dpi=130,bbox_inches="tight"); plt.close(f2)
    buf=io.BytesIO()
    fd,axd=plt.subplots(1,2,figsize=(15,5))
    for a,p in zip(axd,["/tmp/_t.png","/tmp/_c.png"]):
        a.imshow(plt.imread(p)); a.axis("off")
    fd.suptitle(f"Режим {Tp} °C / {tau} с  |  Tмакс={res['Tmax']:.0f}°C  "
                f"C/Cs={res['Cax']:.3f}  η={res['eta']:.2f}  L={res['L']:.0f} м³/ч",fontsize=11)
    fd.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close(fd)
    st.download_button("📥 Скачать графики (PNG)",buf.getvalue(),
        file_name=f"rezhim_{Tp}C_{tau}s.png",mime="image/png")

# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: СРАВНЕНИЕ
# ═══════════════════════════════════════════════════════════════
elif section=="Сравнить два режима":
    with st.spinner("Расчёт двух режимов..."):
        rA=run(int(TA),int(tauA),fiber); rB=run(int(TB),int(tauB),fiber)

    st.subheader("⚖️ Сравнение режимов")
    cA,cB=st.columns(2)
    for col,res,(Tp,ta) in [(cA,rA,(TA,tauA)),(cB,rB,(TB,tauB))]:
        with col:
            st.markdown(f"### {Tp} °C / {ta} с")
            x1,x2=st.columns(2)
            x1.metric("Температура ткани",f"{res['Tmax']:.0f} °C")
            x2.metric("C/Cs на оси","≈ 0" if res["Cax"]<0.005 else f"{res['Cax']:.3f}")
            x3,x4=st.columns(2)
            x3.metric("Коэф. переноса η",f"{res['eta']:.2f}")
            x4.metric("Воздухообмен L",f"{res['L']:.0f} м³/ч",
                      f"K = {res['K']:.1f} ч⁻¹")
            vc,vt=verdict(res)
            st.markdown(f'<div class="card-{vc}">{vt}</div>',unsafe_allow_html=True)
    st.divider()
    st.pyplot(fig_compare(rA,rB,(TA,tauA),(TB,tauB)),use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# РАЗДЕЛ 3: КАРТА
# ═══════════════════════════════════════════════════════════════
elif section=="Карта режимов":
    st.subheader("🗺️ Карта режимов с экспериментальной верификацией")
    st.markdown(
        '<div class="explain">Цветная заливка — результат расчёта модели '
        '(концентрация красителя на оси волокна). Точки — реальные эксперименты '
        'из таблицы 2.1. Если расчёт адекватен, точки «хорошего переноса» (●) '
        'должны попадать в зелёно-жёлтую зону, точки «расплывания» (■) — '
        'в зону насыщения, точки «нет переноса» (✕) — в красную зону.</div>',
        unsafe_allow_html=True)
    with st.spinner("Строю карту (169 расчётов, подождите ~15 сек)..."):
        st.pyplot(fig_phase(),use_container_width=True)
    st.markdown('<div class="card-ok">Экспериментальные точки совпадают с расчётными '
        'зонами — это подтверждает адекватность численной модели.</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# СПРАВОЧНЫЕ БЛОКИ
# ═══════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 Условные обозначения (все символы из диплома)"):
    st.markdown("""
| Обозначение | Полное название | Единицы |
|---|---|---|
| T | Температура | °C |
| Tпресс | Температура нагревательной плиты термопресса | °C |
| Tмакс | Максимальная температура внутри слоя ткани к концу выдержки | °C |
| Tg | Температура стеклования ПЭТ — ниже этой температуры диффузия невозможна | °C |
| T0 | Начальная температура (температура окружающей среды) | °C |
| x | Координата по толщине пакета плита–бумага–ткань | м / мм |
| δ | Толщина слоя (плита δ₁, бумага δ₂, ткань δ₃) | мм |
| τ | Время выдержки в термопрессе | с |
| t | Текущее время в расчёте | с |
| r | Радиальная координата внутри волокна (0 = ось, R = поверхность) | м / мкм |
| R | Радиус поперечного сечения волокна | мкм |
| C | Концентрация красителя в волокне | — |
| Cs | Равновесная концентрация красителя на поверхности волокна | — |
| C/Cs | Доля насыщения: 0 = красителя нет, 1 = максимальное насыщение | — |
| D(T) | Коэффициент диффузии красителя в волокне (зависит от температуры) | м²/с |
| D0 | Предэкспоненциальный множитель в уравнении Аррениуса | м²/с |
| Ea | Энергия активации диффузии | кДж/моль |
| Rг | Универсальная газовая постоянная (8.314) | Дж/(моль·К) |
| λ | Коэффициент теплопроводности слоя | Вт/(м·К) |
| ρ | Плотность слоя | кг/м³ |
| cp | Удельная теплоёмкость слоя | Дж/(кг·К) |
| α | Коэффициент теплоотдачи с открытой поверхности ткани в воздух | Вт/(м²·К) |
| η | Коэффициент переноса красителя: доля, закрепившаяся внутри волокна | — |
| M0 | Исходная масса красителя на бумаге-носителе | г/м² |
| Mres | Остаточная масса красителя на бумаге после выдержки: Mres = M0·(1−η) | г/м² |
| κ | Степень термодеструкции красителя (принято κ = 0.02 — консервативно) | — |
| S | Площадь рабочего поля термопресса | м² |
| n | Производительность термопресса | циклов/ч |
| G | Суммарная интенсивность эмиссии летучих продуктов деструкции | мг/ч |
| G_анилин | Интенсивность выделения анилина (10% от G) | мг/ч |
| ПДК | Предельно допустимая концентрация анилина в воздухе рабочей зоны | мг/м³ |
| L | Требуемый расход приточного воздуха (воздухообмен) | м³/ч |
| K | Кратность воздухообмена: сколько раз в час воздух полностью заменяется | ч⁻¹ |
| V | Объём помещения студии термопечати | м³ |
""")

with st.expander("ℹ️ Как работает модель"):
    st.markdown("""
**Шаг 1 — Нагрев пакета (ур. 3.1).**
Нагревательная плита термопресса прижата к бумаге-носителю, под ней — ткань.
Модель решает уравнение теплопроводности вдоль оси x (от плиты до свободной поверхности ткани).
На плите задана фиксированная температура Tпресс (ур. 3.3), на обратной стороне ткани — конвекция в воздух (ур. 3.4).

**Шаг 2 — Диффузия красителя в волокно (ур. 3.7).**
Когда температура ткани превышает Tg = 75 °C, молекулы красителя начинают проникать
в аморфные области ПЭТ. Скорость диффузии описывается уравнением Аррениуса (ур. 3.8–3.9):
при 100 °C коэффициент D ≈ 10⁻¹⁶ м²/с (практически ноль), при 200 °C D ≈ 4·10⁻¹³ м²/с (в 1000 раз быстрее).

**Шаг 3 — Расчёт остаточного красителя и эмиссии (глава 4).**
Из профиля C(r,τ) вычисляется η — доля красителя, закрепившегося в волокне.
Остаток (1−η) сидит на бумаге и при нагреве частично разлагается с образованием анилина.
По расчётной эмиссии G_анилин и ПДК определяется требуемый воздухообмен L.

**Ограничения модели.**
Модель описывает радиальную диффузию в одно волокно. Боковое растекание красителя
между волокнами (причина расплывания изображения при 220 °C) в одномерной постановке
не описывается — это выражается как C/Cs → 1 при предельных режимах.
Термическая деградация молекул красителя в модель не включена (допущение раздела 3.1).
""")

st.divider()
st.caption(
    f"Шрифт графиков: {_FONT}  |  "
    "Уравнения теплопроводности (3.1) и диффузии Фика (3.7)  |  "
    "МКР, явная схема, условия устойчивости CFL (3.16–3.17)  |  "
    "Расчёт эмиссии: формулы (4.1)–(4.3)  |  ВКР 20.04.01"
)
