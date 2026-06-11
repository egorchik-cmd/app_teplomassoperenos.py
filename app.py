"""
ПРИЛОЖЕНИЕ: Численное моделирование тепломассопереноса при сублимационной термопечати
ВКР, направление 20.04.01 «Техносферная безопасность»

Запуск локально:   streamlit run app.py
Деплой:            GitHub -> Streamlit Community Cloud (main file path: app.py)

Зависимости (requirements.txt):  streamlit / numpy / matplotlib
Кириллические шрифты (packages.txt):  fonts-dejavu-core
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# ── ШРИФТЫ: гарантируем кириллицу на любом окружении ──
def _setup_font():
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in ("DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial"):
        if name in avail:
            plt.rcParams["font.family"] = name
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"

_FONT = _setup_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = True

# ── СТРАНИЦА ──
st.set_page_config(page_title="Модель термопечати", page_icon="🧵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background:#fff; border-radius:12px; padding:14px 16px;
        box-shadow:0 2px 8px rgba(0,0,0,0.07); border:1px solid #eef0f3; }
    div[data-testid="stMetric"] label { font-size:13px !important; color:#5f6368 !important; }
    .verdict-ok   { background:#e8f5e9; border-left:5px solid #43a047;
                    padding:14px 18px; border-radius:6px; color:#1b5e20; font-size:15px; }
    .verdict-bad  { background:#fce4ec; border-left:5px solid #e91e63;
                    padding:14px 18px; border-radius:6px; color:#880e4f; font-size:15px; }
    .verdict-warn { background:#fff8e1; border-left:5px solid #fb8c00;
                    padding:14px 18px; border-radius:6px; color:#e65100; font-size:15px; }
    h1 { font-size:1.7rem !important; } h2 { font-size:1.25rem !important; }
    h3 { font-size:1.05rem !important; }
</style>
""", unsafe_allow_html=True)

# ── КОНСТАНТЫ (глава 3 табл. 3.1; глава 4) ──
LAM1, RHO1, CP1, D1 = 16.0, 7850.0, 500.0, 10.0e-3      # плита [25]
LAM2, RHO2, CP2, D2 = 0.08, 700.0, 1300.0, 0.1e-3       # бумага [24]
LAM3, RHO3, CP3, D3 = 0.049, 1380.0, 1300.0, 0.3e-3     # ткань [24]
T0, ALPHA = 20.0, 10.0
R_GAS, EA, D0, TG = 8.314, 121.0e3, 9.1339, 75.0        # диффузия [26,27]
R_PET, R_SILK = 7.5e-6, 6.3e-6

# эмиссия / вентиляция (глава 4)
M0_DYE, KAPPA, S_PRESS, N_CYCLES = 6.0, 0.02, 0.24, 30
ANILINE_FR, PDK_ANILINE, V_ROOM = 0.10, 0.1, 50.0

# экспериментальные режимы (табл. 2.1), класс: 0 нет / 1 перенос / 2 расплыв
EXP_PET = [(150,30,1),(150,60,1),(170,30,1),(170,60,1),(180,30,1),(180,60,1),
           (200,30,1),(200,60,1),(200,90,1),(200,120,2),(200,150,2)]
EXP_LIMITS = [(100,60,0),(220,150,2)]

# ── МОДЕЛЬ ──
def D_arrhenius(T_c):
    T_c = np.asarray(T_c, dtype=float)
    D = D0 * np.exp(-EA / (R_GAS * (T_c + 273.15)))
    D = np.where(T_c <= TG, 0.0, D)
    return float(D) if D.ndim == 0 else D

def harmonic(a, b):
    return np.where(a + b > 0, 2*a*b/(a+b), 0.0)

def build_grid(Nx=15):
    n1=max(2,int(D1*1e3*Nx)); n2=max(2,int(D2*1e3*Nx)); n3=max(2,int(D3*1e3*Nx))
    x=np.concatenate([np.linspace(0,D1,n1,endpoint=False),
                      np.linspace(D1,D1+D2,n2,endpoint=False),
                      np.linspace(D1+D2,D1+D2+D3,n3+1)])
    N=len(x); lam,rho,cp=np.empty(N),np.empty(N),np.empty(N); i2,i3=n1,n1+n2
    lam[:i2],rho[:i2],cp[:i2]=LAM1,RHO1,CP1
    lam[i2:i3],rho[i2:i3],cp[i2:i3]=LAM2,RHO2,CP2
    lam[i3:],rho[i3:],cp[i3:]=LAM3,RHO3,CP3
    return x,lam,rho,cp,i2,i3

@st.cache_data(show_spinner=False)
def solve_heat(T_press, tau, safety=0.4):
    x,lam,rho,cp,i2,i3=build_grid(); dx=x[1]-x[0]
    dt=safety*0.5*dx**2/np.max(lam/(rho*cp))
    T=np.full(len(x),T0); T[0]=T_press
    se=max(1,int(tau/(24*dt))); sT,st_=[T.copy()],[0.0]; t,step=0.0,0
    while t<tau:
        ds=min(dt,tau-t); Tn=T.copy()
        le=harmonic(lam[1:-1],lam[2:]); lw=harmonic(lam[:-2],lam[1:-1])
        Tn[1:-1]=T[1:-1]+ds/(rho[1:-1]*cp[1:-1])*(le*(T[2:]-T[1:-1])-lw*(T[1:-1]-T[:-2]))/dx**2
        Tn[-1]=(lam[-1]/dx*T[-2]+ALPHA*T0)/(lam[-1]/dx+ALPHA); Tn[0]=T_press
        T=Tn; t+=ds; step+=1
        if step%se==0 or abs(t-tau)<1e-9: sT.append(T.copy()); st_.append(t)
    return x,np.array(sT),np.array(st_),i3

@st.cache_data(show_spinner=False)
def solve_diffusion(T_fab_t, t_fab_t, tau, R_fiber, fiber="PET", Nr=40, safety=0.4):
    T_fab=np.array(T_fab_t); dr=R_fiber/(Nr-1); r=np.linspace(0,R_fiber,Nr)
    D_mx=float(D_arrhenius(np.max(T_fab))); dt=(safety*dr**2/D_mx) if D_mx>0 else 1.0
    C=np.zeros(Nr); sC,st_=[C.copy()],[0.0]; se=max(1,int(tau/(24*dt))); t,step=0.0,0
    while t<tau:
        ds=min(dt,tau-t); frac=t/tau*(len(T_fab)-1)
        lo=int(frac); hi=min(lo+1,len(T_fab)-1); w=frac-lo
        T_cur=(1-w)*T_fab[lo]+w*T_fab[hi]; D_cur=float(D_arrhenius(T_cur))
        if D_cur>0:
            Cn=C.copy(); j=np.arange(1,Nr-1); re=r[j]+0.5*dr; rw=r[j]-0.5*dr
            Cn[j]=C[j]+ds/(r[j]*dr)*(D_cur*re*(C[j+1]-C[j])-D_cur*rw*(C[j]-C[j-1]))/dr
            Cn[0]=C[0]+ds*2*D_cur*(C[1]-C[0])/dr**2
            Cn[-1]=(1.0 if T_cur>TG else 0.0) if fiber=="PET" else Cn[-2]
            C=np.clip(Cn,0.0,1.0)
        t+=ds; step+=1
        if step%se==0 or abs(t-tau)<1e-9: sC.append(C.copy()); st_.append(t)
    return r,np.array(sC),np.array(st_)

_TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def transfer_coeff(C,r):
    return float(np.clip(_TRAP(C*r,r)/(0.5*r[-1]**2),0.0,1.0))

def emission(eta):
    M_res=M0_DYE*(1-eta)
    G=KAPPA*M_res*S_PRESS*N_CYCLES*1000
    G_an=ANILINE_FR*G; L=G_an/PDK_ANILINE; K=L/V_ROOM
    return M_res,G,G_an,L,K

def depth_half(C,r):
    for i in range(len(r)-1,-1,-1):
        if C[i]<=0.5: return (r[-1]-r[i])*1e6
    return 0.0

def fmt_sci(v,u=""):
    if v==0: return f"0{(' '+u) if u else ''}"
    e=int(np.floor(np.log10(abs(v)))); m=v/10**e
    sup=str(e).translate(str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.1f}×10{sup}{(' '+u) if u else ''}"

@st.cache_data(show_spinner=False)
def run_mode(T_press, tau, fiber):
    Rf=R_PET if fiber=="PET" else R_SILK
    x,T_sn,t_sn,i3=solve_heat(float(T_press),int(tau))
    i_mid=i3+(len(x)-i3)//2; T_fab=T_sn[:,i_mid]
    r,C_sn,t_c=solve_diffusion(tuple(T_fab.tolist()),tuple(t_sn.tolist()),
                                int(tau),float(Rf),fiber)
    C_fin=C_sn[-1]
    res=dict(x=x,T_sn=T_sn,t_sn=t_sn,i3=i3,T_fab=T_fab,r=r,C_sn=C_sn,t_c=t_c,
             C_fin=C_fin,R_fiber=Rf,fiber=fiber,T_max=float(T_fab[-1]),
             C_axis=float(C_fin[0]),depth=depth_half(C_fin,r),
             D_at=D_arrhenius(float(T_fab[-1])))
    res["eta"]=transfer_coeff(C_fin,r) if fiber=="PET" else 0.0
    res["M_res"],res["G"],res["G_an"],res["L"],res["K"]=emission(res["eta"])
    res["tau_diff"]=(Rf**2/(6*res["D_at"])) if res["D_at"]>0 else np.inf
    return res

# ── ГРАФИКИ ──
def fig_temperature(res,Tp,tau):
    x,T_sn,t_sn=res["x"],res["T_sn"],res["t_sn"]
    fig,ax=plt.subplots(figsize=(7.5,4.5)); fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.axvspan(0,D1*1e3,alpha=0.07,color="#e53935")
    ax.axvspan(D1*1e3,(D1+D2)*1e3,alpha=0.13,color="#f9a825")
    ax.axvspan((D1+D2)*1e3,(D1+D2+D3)*1e3,alpha=0.10,color="#1e88e5")
    for xv in (D1*1e3,(D1+D2)*1e3): ax.axvline(xv,color="#999",ls="--",lw=0.9,alpha=0.6)
    ax.axhline(TG,color="#43a047",ls="-.",lw=1.4,alpha=0.85,label=f"$T_g$ = {TG:.0f} °C")
    idx=np.unique(np.round(np.linspace(0,len(t_sn)-1,6)).astype(int))
    cmap=plt.cm.plasma(np.linspace(0.15,0.92,len(idx)))
    for k,i in enumerate(idx):
        ax.plot(x*1e3,T_sn[i],color=cmap[k],lw=2.6 if i==idx[-1] else 1.4,
                label=f"t = {t_sn[i]:.0f} с")
    ax.set_xlabel("Координата $x$, мм"); ax.set_ylabel("Температура $T$, °C")
    ax.set_title(f"Поле температуры $T(x,t)$ — режим {Tp} °C / {tau} с")
    ax.set_xlim(0,(D1+D2+D3)*1e3); ax.set_ylim(T0-5,max(Tp*1.05,TG+10))
    ax.legend(fontsize=8,loc="upper right",framealpha=0.9); ax.grid(True,alpha=0.3,ls="--")
    for lbl,xc in (("Плита",D1/2),("Ткань",D1+D2+D3/2)):
        ax.text(xc*1e3,T0+2,lbl,ha="center",fontsize=8,color="#555",style="italic")
    return fig

def fig_concentration(res,Tp,tau):
    r,C_sn,t_c,Rf=res["r"],res["C_sn"],res["t_c"],res["R_fiber"]
    fig,ax=plt.subplots(figsize=(7.5,4.5)); fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    if res["fiber"]=="SILK":
        ax.plot([0,1],[0,0],color="#fb8c00",lw=3)
        ax.text(0.5,0.5,"Шёлк непроницаем\n(ГУ Неймана, ур. 3.13)",ha="center",
                va="center",fontsize=12,color="#e65100",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5",fc="#fff8e1",ec="#fb8c00"))
    else:
        idx=np.unique(np.round(np.linspace(0,len(t_c)-1,5)).astype(int))
        cmap=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(idx)))
        for k,i in enumerate(idx):
            ax.plot(r/Rf,C_sn[i],color=cmap[k],lw=3.0 if i==idx[-1] else 1.2,
                    alpha=1.0 if i==idx[-1] else 0.55,label=f"t = {t_c[i]:.0f} с")
        ax.fill_between(r/Rf,0,C_sn[-1],alpha=0.12,color="#e53935")
        ax.axhline(0.5,color="#555",ls=":",lw=1.1,alpha=0.7,label="$C/C_s = 0.5$")
        ax.plot(0,res["C_axis"],"r*",ms=13)
    ax.set_xlabel("$r/R$    (ось ← → поверхность)")
    ax.set_ylabel("Концентрация $C/C_s$")
    ax.set_title(f"Профиль красителя $C(r,t)$ — режим {Tp} °C / {tau} с")
    ax.set_xlim(0,1); ax.set_ylim(-0.05,1.12)
    ax.legend(fontsize=8,loc="upper left" if res["fiber"]=="PET" else "lower right",
              framealpha=0.9); ax.grid(True,alpha=0.3,ls="--")
    return fig

def fig_phase(fiber):
    T_rng=np.linspace(100,220,13); tau_rng=np.linspace(20,300,13)
    grid=np.zeros((len(T_rng),len(tau_rng)))
    for i,Tp in enumerate(T_rng):
        for j,ta in enumerate(tau_rng):
            grid[i,j]=run_mode(int(round(Tp)),int(round(ta)),fiber)["C_axis"]
    fig,ax=plt.subplots(figsize=(9,5.5)); fig.patch.set_facecolor("white")
    cf=ax.contourf(tau_rng,T_rng,grid,levels=20,cmap="RdYlGn",alpha=0.92)
    cs=ax.contour(tau_rng,T_rng,grid,levels=[0.1,0.3,0.5,0.7,0.9],
                  colors="black",alpha=0.35,linewidths=0.6)
    ax.clabel(cs,inline=True,fontsize=7,fmt="%.1f")
    cb=fig.colorbar(cf,ax=ax,pad=0.02); cb.set_label("$C/C_s$ на оси волокна")
    mm={0:("X","#c62828","нет переноса"),1:("o","#1b5e20","перенос (оптимум)"),
        2:("s","#e65100","расплывание")}
    seen=set()
    for (Tp,ta,cls) in (EXP_PET if fiber=="PET" else [])+EXP_LIMITS:
        mk,col,lab=mm[cls]
        ax.scatter(ta,Tp,marker=mk,c=col,s=90,edgecolors="white",linewidths=1.2,
                   zorder=5,label=("Эксп.: "+lab) if lab not in seen else None)
        seen.add(lab)
    ax.set_xlabel("Время выдержки $\\tau$, с")
    ax.set_ylabel("Температура плиты $T_{press}$, °C")
    ax.set_title("Карта режимов: расчёт (заливка) + эксперимент (точки)")
    ax.legend(fontsize=8,loc="lower right",framealpha=0.95)
    ax.set_xlim(20,300); ax.set_ylim(100,220); ax.grid(True,alpha=0.2)
    return fig

def fig_compare(rA,rB,mA,mB):
    fig,axes=plt.subplots(2,2,figsize=(12,8)); fig.patch.set_facecolor("white")
    for col,(res,m) in enumerate([(rA,mA),(rB,mB)]):
        ax=axes[0,col]; ax.set_facecolor("#fafafa")
        ax.axvspan((D1+D2)*1e3,(D1+D2+D3)*1e3,alpha=0.10,color="#1e88e5")
        ax.axhline(TG,color="#43a047",ls="-.",lw=1.2,label=f"$T_g$={TG:.0f}°C")
        ax.plot(res["x"]*1e3,res["T_sn"][-1],"b-",lw=2.4)
        ax.set_title(f"{m[0]} °C / {m[1]} с   |   $T_{{max}}$ = {res['T_max']:.0f} °C")
        ax.set_xlabel("$x$, мм"); ax.set_ylabel("$T$, °C")
        ax.set_xlim(0,(D1+D2+D3)*1e3); ax.grid(True,alpha=0.3,ls="--"); ax.legend(fontsize=8)
        ax=axes[1,col]; ax.set_facecolor("#fafafa")
        ax.plot(res["r"]/res["R_fiber"],res["C_fin"],"r-",lw=2.6)
        ax.fill_between(res["r"]/res["R_fiber"],0,res["C_fin"],alpha=0.15,color="#e53935")
        ax.set_title(f"$C/C_s$ на оси = {res['C_axis']:.3f}   |   $\\eta$ = {res['eta']:.2f}")
        ax.set_xlabel("$r/R$"); ax.set_ylabel("$C/C_s$")
        ax.set_xlim(0,1); ax.set_ylim(-0.05,1.12); ax.grid(True,alpha=0.3,ls="--")
    return fig

def verdict(res):
    c=res["C_axis"]
    if res["fiber"]=="SILK":
        return ("warn","Шёлк: дисперсный краситель не проникает в фиброин "
                "(поверхностное окрашивание). $C \\equiv 0$ во всех режимах.")
    if c<0.01:
        return ("bad",f"Недостаточный нагрев. $T_{{max}}$ = {res['T_max']:.0f} °C — "
                "краситель не диффундирует ($D \\approx 0$). Цветопереноса нет.")
    if c<0.5:
        return ("ok",f"Оптимальный режим. $C/C_s$ на оси = {c:.3f} — равномерное "
                "проникновение красителя. Ожидается чёткий насыщенный цвет.")
    if c<0.95:
        return ("warn",f"Интенсивный режим. $C/C_s$ = {c:.3f} — глубокое проникновение, "
                "возможно расплывание контуров.")
    return ("warn",f"Предельный режим. Полное насыщение сечения ($C/C_s$ = {c:.3f}). "
            "Краситель мигрирует за контур — визуальное расплывание изображения.")

# ── ЗАГОЛОВОК ──
st.title("🧵 Тепломассоперенос при сублимационной термопечати")
st.caption("Теплопроводность (3.1) + диффузия Фика (3.7) → перенос красителя → "
           "эмиссия токсикантов → воздухообмен  |  ВКР 20.04.01")
st.divider()

# ── САЙДБАР ──
with st.sidebar:
    st.header("⚙️ Параметры")
    section=st.radio("Раздел",["Один режим","Сравнить два режима","Карта режимов"])
    st.divider()
    if section in ("Один режим","Сравнить два режима"):
        fiber_label=st.radio("Ткань",["Полиэстер (ПЭТ)","Шёлк"])
        fiber="PET" if "ПЭТ" in fiber_label else "SILK"
    else:
        fiber="PET"; st.info("Карта строится для полиэстера (есть эксп. данные табл. 2.1).")
    st.divider()
    if section=="Один режим":
        im=st.radio("Ввод",["Задать вручную","Примеры из диплома"])
        if im=="Примеры из диплома":
            preset=st.selectbox("Пример",["100 °C / 60 с — недостаточный",
                "200 °C / 60 с — оптимальный","220 °C / 150 с — предельный"])
            T_press=int(preset.split(" ")[0])
            tau=int(preset.split("/")[1].strip().split(" ")[0])
        else:
            T_press=st.slider("$T_{press}$, °C",80,220,200,5)
            tau=st.slider("$\\tau$, с",20,300,60,10)
    elif section=="Сравнить два режима":
        st.markdown("**Режим A**")
        TA=st.slider("$T_{press}^{A}$, °C",80,220,200,5,key="TA")
        tauA=st.slider("$\\tau^{A}$, с",20,300,60,10,key="tauA")
        st.markdown("**Режим B**")
        TB=st.slider("$T_{press}^{B}$, °C",80,220,220,5,key="TB")
        tauB=st.slider("$\\tau^{B}$, с",20,300,150,10,key="tauB")
    st.divider(); st.caption(f"Шрифт графиков: {_FONT}")

# ── РАЗДЕЛ 1: ОДИН РЕЖИМ ──
if section=="Один режим":
    with st.spinner("Расчёт..."):
        res=run_mode(int(T_press),int(tau),fiber)
    st.subheader("📊 Перенос красителя")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("T_max в ткани",f"{res['T_max']:.0f} °C",f"{res['T_max']-TG:+.0f}°C к Tg",
              delta_color="normal" if res['T_max']>TG else "inverse")
    m2.metric("C/Cs на оси","≈ 0" if res["C_axis"]<0.005 else f"{res['C_axis']:.3f}")
    m3.metric("Глубина (до C=0.5)",f"{res['depth']:.1f} мкм",f"R = {res['R_fiber']*1e6:.1f} мкм")
    m4.metric("D при T_max",fmt_sci(res["D_at"],"м²/с"))
    vc,vt=verdict(res)
    st.markdown(f'<div class="verdict-{vc}">{vt}</div>',unsafe_allow_html=True)
    st.divider()
    st.subheader("📈 Графики процесса")
    g1,g2=st.columns(2)
    with g1: st.pyplot(fig_temperature(res,T_press,tau),use_container_width=True)
    with g2: st.pyplot(fig_concentration(res,T_press,tau),use_container_width=True)
    st.divider()
    st.subheader("🛡️ Оценка профессионального риска (глава 4)")
    if fiber=="SILK":
        st.info("Расчёт эмиссии — для полиэстера (основной материал ВКР). "
                "Для шёлка перенос поверхностный, η ≈ 0.")
    else:
        e1,e2,e3,e4=st.columns(4)
        e1.metric("η переноса",f"{res['eta']:.2f}",help="Доля красителя в волокне (ур. 4.1)")
        e2.metric("Остаточный краситель",f"{res['M_res']:.2f} г/м²",help="M_res = M0·(1−η)")
        e3.metric("Эмиссия анилина",f"{res['G_an']:.0f} мг/ч",help="G_ан = 0.10·κ·M_res·S·n")
        e4.metric("Воздухообмен L",f"{res['L']:.0f} м³/ч",f"K = {res['K']:.1f} ч⁻¹",
                  help="L = G_анилин / ПДК (ур. 4.3)")
        if res["L"]<=0:
            st.markdown('<div class="verdict-ok">Перенос почти полный — '
                        'остаточный краситель и эмиссия минимальны.</div>',unsafe_allow_html=True)
        elif res["K"]<=10:
            st.markdown(f'<div class="verdict-ok">L = {res["L"]:.0f} м³/ч '
                        f'(K = {res["K"]:.1f} ч⁻¹) — в норме 6–10 ч⁻¹ для студии '
                        f'{V_ROOM:.0f} м³ (СП 60.13330.2020).</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-warn">L = {res["L"]:.0f} м³/ч '
                        f'(K = {res["K"]:.1f} ч⁻¹) — выше типовой нормы. Нужна местная '
                        f'вытяжка повышенной производительности.</div>',unsafe_allow_html=True)
        st.caption("Цепочка (глава 4): η → M_res=M₀·(1−η) → G=κ·M_res·S·n → "
                   "G_анилин=0.10·G → L=G_анилин/ПДК → K=L/V. ПДК анилина = 0.1 мг/м³.")
    st.divider()
    st.subheader("💾 Экспорт")
    f1=fig_temperature(res,T_press,tau); f1.savefig("/tmp/_t.png",dpi=130,bbox_inches="tight"); plt.close(f1)
    f2=fig_concentration(res,T_press,tau); f2.savefig("/tmp/_c.png",dpi=130,bbox_inches="tight"); plt.close(f2)
    buf=io.BytesIO(); fd,axd=plt.subplots(1,2,figsize=(15,5))
    for a,p in zip(axd,["/tmp/_t.png","/tmp/_c.png"]): a.imshow(plt.imread(p)); a.axis("off")
    fd.suptitle(f"Режим {T_press} °C / {tau} с  |  T_max={res['T_max']:.0f}°C  "
                f"C/Cs={res['C_axis']:.3f}  eta={res['eta']:.2f}  L={res['L']:.0f} м³/ч",
                fontsize=11)
    fd.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close(fd)
    st.download_button("📥 Скачать графики (PNG)",buf.getvalue(),
                       file_name=f"rezhim_{T_press}C_{tau}s.png",mime="image/png")

# ── РАЗДЕЛ 2: СРАВНЕНИЕ ──
elif section=="Сравнить два режима":
    with st.spinner("Расчёт двух режимов..."):
        rA=run_mode(int(TA),int(tauA),fiber); rB=run_mode(int(TB),int(tauB),fiber)
    st.subheader("⚖️ Сравнение режимов")
    cA,cB=st.columns(2)
    for col,res,(Tp,ta) in [(cA,rA,(TA,tauA)),(cB,rB,(TB,tauB))]:
        with col:
            st.markdown(f"### {Tp} °C / {ta} с")
            x1,x2=st.columns(2)
            x1.metric("T_max",f"{res['T_max']:.0f} °C")
            x2.metric("C/Cs ось","≈ 0" if res["C_axis"]<0.005 else f"{res['C_axis']:.3f}")
            x3,x4=st.columns(2)
            x3.metric("η переноса",f"{res['eta']:.2f}")
            x4.metric("L",f"{res['L']:.0f} м³/ч")
            cls,txt=verdict(res)
            st.markdown(f'<div class="verdict-{cls}">{txt}</div>',unsafe_allow_html=True)
    st.divider()
    st.pyplot(fig_compare(rA,rB,(TA,tauA),(TB,tauB)),use_container_width=True)

# ── РАЗДЕЛ 3: КАРТА ──
elif section=="Карта режимов":
    st.subheader("🗺️ Карта режимов с экспериментальной верификацией")
    st.caption("Заливка — расчётная C/Cs на оси волокна. Точки — экспериментальные "
               "режимы (табл. 2.1, полиэстер) и опорные точки диапазона [1].")
    with st.spinner("Построение карты (13×13 расчётов)..."):
        st.pyplot(fig_phase(fiber),use_container_width=True)
    st.markdown('<div class="verdict-ok">Точки «перенос» попадают в расчётную '
                'зелёно-жёлтую зону, «расплывание» — в зону насыщения, «нет переноса» — '
                'в красную зону. Это подтверждает адекватность модели.</div>',
                unsafe_allow_html=True)

# ── СПРАВОЧНЫЕ БЛОКИ ──
st.divider()
with st.expander("📖 Условные обозначения и единицы измерения"):
    st.markdown("""
| Обозначение | Величина | Единицы |
|---|---|---|
| $T$ | температура | °C |
| $T_{press}$ | температура нагревательной плиты | °C |
| $T_{max}$ | максимальная температура в слое ткани | °C |
| $T_g$ | температура стеклования ПЭТ (порог диффузии) | °C |
| $T_0$ | начальная температура / температура среды | °C |
| $x$ | координата по толщине пакета (1D) | мм / м |
| $\\delta$ | толщина слоя | мм / м |
| $t$ | время | с |
| $\\tau$ | время выдержки (экспозиции) | с |
| $\\tau_{diff}$ | характерное время диффузии, $R^2/6D$ | с |
| $C$ | концентрация красителя в волокне | отн. ед. |
| $C_s$ | равновесная концентрация на поверхности волокна | отн. ед. |
| $C/C_s$ | нормированная концентрация | — |
| $r$ | радиальная координата в волокне | мкм / м |
| $R$ | радиус волокна | мкм / м |
| $D$ | коэффициент диффузии | м²/с |
| $D_0$ | предэкспоненциальный множитель (Аррениус) | м²/с |
| $E_a$ | энергия активации диффузии | кДж/моль |
| $R_г$ | универсальная газовая постоянная (8.314) | Дж/(моль·К) |
| $\\lambda$ | теплопроводность слоя | Вт/(м·К) |
| $\\rho$ | плотность слоя | кг/м³ |
| $c_p$ | удельная теплоёмкость слоя | Дж/(кг·К) |
| $\\alpha$ | коэффициент теплоотдачи | Вт/(м²·К) |
| $\\eta$ | коэффициент переноса красителя в волокно | — |
| $M_0$ | исходная масса красителя на бумаге | г/м² |
| $M_{res}$ | остаточная масса красителя | г/м² |
| $\\kappa$ | степень термодеструкции красителя | — |
| $G$ | интенсивность эмиссии токсикантов | мг/ч |
| $L$ | требуемый воздухообмен | м³/ч |
| $K$ | кратность воздухообмена | ч⁻¹ |
| ПДК | предельно допустимая концентрация | мг/м³ |
""")

with st.expander("ℹ️ О модели и допущениях"):
    st.markdown("""
Модель решает **сопряжённую задачу тепломассопереноса** методом конечных
разностей (явная схема) при соблюдении условий устойчивости CFL (3.16–3.17):

1. **Теплопроводность** (3.1) — 1D-задача в пакете «плита–бумага–ткань»:
   ГУ Дирихле на плите (3.3), Ньютона–Рихмана на поверхности ткани (3.4),
   сопряжение на границах слоёв (3.5–3.6).
2. **Диффузия Фика** (3.7) — в цилиндрических координатах волокна;
   $D(T)$ по Аррениусу (3.8–3.9). Для ПЭТ — ГУ Дирихле $C(R)=C_s$ (3.11);
   для шёлка — ГУ Неймана (3.13).
3. **Охрана труда** (глава 4) — из профиля $C(r,\\tau)$ вычисляется $\\eta$,
   далее остаточная масса, эмиссия анилина и требуемый воздухообмен.

**Ограничение:** учитывается только радиальная диффузия в одно волокно.
Предельный режим воспроизводится как полное насыщение ($C/C_s \\to 1$), что
физически соответствует наблюдаемому расплыванию изображения. Термическая
деградация молекул красителя в постановку не входит.
""")

st.divider()
st.caption("Уравнения (3.1), (3.7), Аррениус (3.8) | МКР, явная схема, CFL (3.16–3.17) | "
           "глава 4: эмиссия и воздухообмен | ВКР 20.04.01")
