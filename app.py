"""
StatPharm Insight — app.py
==========================
Aplikasi web analisis statistik data penelitian farmasi.

Jalankan:
    streamlit run app.py

Dependensi:
    pip install streamlit pandas scipy matplotlib seaborn openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# ── Konfigurasi halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatPharm Insight",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Kartu metrik */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a3a5c; }
    .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

    /* Badge hasil uji */
    .badge-pass {
        display: inline-block; padding: 0.2rem 0.7rem;
        background: #d1fae5; color: #065f46;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    }
    .badge-fail {
        display: inline-block; padding: 0.2rem 0.7rem;
        background: #fee2e2; color: #991b1b;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    }

    /* Narasi box */
    .narasi-box {
        background: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #1e293b;
    }

    /* Step indicator */
    .step-label {
        background: #1a3a5c; color: white;
        border-radius: 50%; width: 26px; height: 26px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700; margin-right: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Divider */
    .section-divider {
        border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# FUNGSI UTILITAS
# ══════════════════════════════════════════════════════════════════════════

def uji_normalitas(grup_data: dict) -> dict:
    hasil = {}
    for nama, data in grup_data.items():
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            hasil[nama] = {"stat": round(stat, 4), "p": round(p, 4),
                           "normal": p > 0.05}
        else:
            hasil[nama] = {"stat": None, "p": None, "normal": None,
                           "error": "Sampel < 3"}
    return hasil


def uji_homogenitas(grup_data: dict) -> dict:
    arrays = [v for v in grup_data.values() if len(v) >= 2]
    if len(arrays) < 2:
        return {"stat": None, "p": None, "homogen": None, "error": "Kurang dari 2 grup"}
    stat, p = stats.levene(*arrays)
    return {"stat": round(stat, 4), "p": round(p, 4), "homogen": p > 0.05}


def uji_beda(grup_data: dict, semua_normal: bool, homogen: bool) -> dict:
    arrays = list(grup_data.values())
    n_grup = len(arrays)

    if n_grup < 2:
        return {"uji": None, "stat": None, "p": None,
                "error": "Minimal 2 grup diperlukan"}

    if n_grup == 2:
        if semua_normal and homogen:
            uji = "Independent t-test"
            stat, p = stats.ttest_ind(arrays[0], arrays[1])
        elif semua_normal and not homogen:
            uji = "Welch t-test"
            stat, p = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
        else:
            uji = "Mann-Whitney U"
            stat, p = stats.mannwhitneyu(arrays[0], arrays[1],
                                          alternative="two-sided")
    else:
        if semua_normal and homogen:
            uji = "One-Way ANOVA"
            stat, p = stats.f_oneway(*arrays)
        else:
            uji = "Kruskal-Wallis"
            stat, p = stats.kruskal(*arrays)

    return {"uji": uji, "stat": round(float(stat), 4),
            "p": round(float(p), 4), "signifikan": p < 0.05}


def buat_narasi(deskriptif: pd.DataFrame, normalitas: dict,
                homogenitas: dict, beda: dict, kolom_grup: str,
                kolom_nilai: str) -> str:

    n_grup = len(normalitas)
    semua_normal = all(v.get("normal", False) for v in normalitas.values()
                       if v.get("normal") is not None)
    homogen = homogenitas.get("homogen", False)

    narasi = f"**Hasil Analisis StatPharm Insight**\n\n"

    # Deskriptif
    narasi += f"Data terdiri dari **{n_grup} kelompok** berdasarkan variabel "
    narasi += f"**{kolom_grup}** dengan variabel nilai **{kolom_nilai}**. "

    means = deskriptif.loc["mean"]
    best_grup = means.idxmax()
    narasi += (f"Nilai rata-rata tertinggi ditemukan pada kelompok **{best_grup}** "
               f"(Mean = {means[best_grup]:.3f}).\n\n")

    # Normalitas
    narasi += "**Uji Normalitas (Shapiro-Wilk):** "
    if semua_normal:
        narasi += ("Seluruh kelompok menunjukkan distribusi data yang **normal** "
                   "(p > 0,05), sehingga asumsi normalitas terpenuhi.\n\n")
    else:
        tidak_normal = [k for k, v in normalitas.items()
                        if v.get("normal") is False]
        narasi += (f"Kelompok **{', '.join(tidak_normal)}** tidak memenuhi asumsi "
                   f"normalitas (p ≤ 0,05), sehingga uji non-parametrik digunakan.\n\n")

    # Homogenitas
    if homogenitas.get("p") is not None:
        narasi += "**Uji Homogenitas (Levene):** "
        if homogen:
            narasi += "Varians antar kelompok **homogen** (p > 0,05).\n\n"
        else:
            narasi += "Varians antar kelompok **tidak homogen** (p ≤ 0,05).\n\n"

    # Uji beda
    if beda.get("uji"):
        narasi += f"**Uji Beda ({beda['uji']}):** "
        stat_val = beda['stat']
        p_val    = beda['p']
        if beda.get("signifikan"):
            narasi += (f"Terdapat perbedaan yang **signifikan** antar kelompok "
                       f"(statistik = {stat_val}, p = {p_val} < 0,05). "
                       f"Hipotesis nol (H₀) ditolak.")
        else:
            narasi += (f"**Tidak terdapat** perbedaan yang signifikan antar kelompok "
                       f"(statistik = {stat_val}, p = {p_val} ≥ 0,05). "
                       f"Hipotesis nol (H₀) gagal ditolak.")

    return narasi


def buat_boxplot(df: pd.DataFrame, kolom_grup: str, kolom_nilai: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Blues_d", n_colors=df[kolom_grup].nunique())
    sns.boxplot(data=df, x=kolom_grup, y=kolom_nilai,
                palette=palette, width=0.5, ax=ax,
                linewidth=1.2, flierprops=dict(marker="o", ms=5, alpha=0.5))
    ax.set_title(f"Boxplot: {kolom_nilai} per {kolom_grup}",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel(kolom_grup, fontsize=10)
    ax.set_ylabel(kolom_nilai, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def buat_barplot(deskriptif: pd.DataFrame, kolom_nilai: str):
    means  = deskriptif.loc["mean"]
    stds   = deskriptif.loc["std"]
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("Blues_d", n_colors=len(means))
    bars = ax.bar(means.index, means.values, yerr=stds.values,
                  color=palette, edgecolor="white", linewidth=0.8,
                  capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#334155"})
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds.max() * 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="600")
    ax.set_title(f"Rata-rata ± SD: {kolom_nilai}",
                 fontweight="bold", fontsize=12, pad=12)
    ax.set_xlabel("Kelompok", fontsize=10)
    ax.set_ylabel(kolom_nilai, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚗️ StatPharm Insight")
    st.markdown("---")
    st.markdown("**Panduan Penggunaan**")
    st.markdown("""
    <span class='step-label'>1</span> Upload file CSV<br><br>
    <span class='step-label'>2</span> Pilih kolom kelompok<br><br>
    <span class='step-label'>3</span> Pilih kolom nilai<br><br>
    <span class='step-label'>4</span> Lihat hasil analisis
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Format CSV yang Didukung**")
    st.markdown("- Kolom kelompok: teks/kategori\n- Kolom nilai: angka numerik\n- Separator: koma (`,`)")
    st.markdown("---")
    alpha = st.slider("Tingkat Signifikansi (α)", 0.01, 0.10, 0.05, 0.01,
                      format="%.2f")
    st.caption(f"Nilai p < {alpha} dianggap signifikan")
    st.markdown("---")
    st.caption("StatPharm Insight v1.0\nDibuat untuk analisis data farmasi")


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='main-header'>
    <h1>⚗️ StatPharm Insight</h1>
    <p>Analisis statistik otomatis untuk data penelitian farmasi — upload, pilih kolom, baca hasilnya.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# UPLOAD FILE
# ══════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader("Upload file CSV", type=["csv"],
                             help="Pastikan file CSV memiliki header kolom di baris pertama.")

if uploaded is None:
    st.info("👆 Upload file CSV untuk memulai analisis.")

    # Contoh data dummy untuk demo
    with st.expander("📥 Tidak punya file CSV? Download contoh di sini"):
        demo_data = pd.DataFrame({
            "Formulasi": (["F1"] * 10 + ["F2"] * 10 + ["F3"] * 10),
            "Kadar_Zat_Aktif": (
                np.random.normal(95, 2, 10).tolist() +
                np.random.normal(98, 1.5, 10).tolist() +
                np.random.normal(92, 3, 10).tolist()
            )
        })
        demo_data["Kadar_Zat_Aktif"] = demo_data["Kadar_Zat_Aktif"].round(2)
        csv_bytes = demo_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Contoh",
                           data=csv_bytes,
                           file_name="contoh_data_farmasi.csv",
                           mime="text/csv")
        st.dataframe(demo_data.head(8), use_container_width=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# LOAD & PREVIEW DATA
# ══════════════════════════════════════════════════════════════════════════
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 📋 Preview Data")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown(f"<div class='metric-card'><div class='value'>{df.shape[0]}</div>"
                f"<div class='label'>Total Baris</div></div>", unsafe_allow_html=True)
with col_info2:
    st.markdown(f"<div class='metric-card'><div class='value'>{df.shape[1]}</div>"
                f"<div class='label'>Total Kolom</div></div>", unsafe_allow_html=True)
with col_info3:
    missing = df.isnull().sum().sum()
    st.markdown(f"<div class='metric-card'><div class='value'>{missing}</div>"
                f"<div class='label'>Missing Values</div></div>", unsafe_allow_html=True)

st.dataframe(df.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PILIH KOLOM
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### ⚙️ Pilih Kolom Analisis")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    kolom_grup = st.selectbox("Kolom Kelompok (kategori)",
                               options=df.columns.tolist(),
                               help="Kolom yang berisi nama kelompok/formulasi")
with col_sel2:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("Tidak ditemukan kolom numerik dalam file CSV.")
        st.stop()
    kolom_nilai = st.selectbox("Kolom Nilai (numerik)",
                                options=numeric_cols,
                                help="Kolom yang berisi nilai yang ingin dianalisis")

if kolom_grup == kolom_nilai:
    st.warning("Kolom kelompok dan kolom nilai tidak boleh sama.")
    st.stop()

# Bersihkan missing
df_clean = df[[kolom_grup, kolom_nilai]].dropna()
df_clean[kolom_grup] = df_clean[kolom_grup].astype(str)
grup_names = sorted(df_clean[kolom_grup].unique())
grup_data  = {g: df_clean.loc[df_clean[kolom_grup] == g, kolom_nilai].values
              for g in grup_names}

if len(grup_names) < 2:
    st.error("Minimal 2 kelompok diperlukan untuk analisis beda.")
    st.stop()

if st.button("▶ Jalankan Analisis", type="primary", use_container_width=True):
    st.session_state["run_analysis"] = True

if not st.session_state.get("run_analysis"):
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# STATISTIK DESKRIPTIF
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 📊 Statistik Deskriptif")

deskriptif_raw = df_clean.groupby(kolom_grup)[kolom_nilai].agg(
    ["count", "mean", "std", "min",
     lambda x: x.quantile(0.25),
     "median",
     lambda x: x.quantile(0.75),
     "max"]
).round(4)
deskriptif_raw.columns = ["N", "Mean", "Std Dev", "Min", "Q1", "Median", "Q3", "Max"]

st.dataframe(deskriptif_raw.style.format("{:.4f}", subset=["Mean","Std Dev","Min","Q1","Median","Q3","Max"])
             .background_gradient(subset=["Mean"], cmap="Blues"),
             use_container_width=True)

# Untuk fungsi lain — format pivot
deskriptif_pivot = df_clean.groupby(kolom_grup)[kolom_nilai].agg(
    ["mean", "std", "min", "max"]).T


# ══════════════════════════════════════════════════════════════════════════
# VISUALISASI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 📈 Visualisasi")

tab_box, tab_bar = st.tabs(["📦 Boxplot", "📊 Rata-rata ± SD"])

with tab_box:
    fig_box = buat_boxplot(df_clean, kolom_grup, kolom_nilai)
    st.pyplot(fig_box, use_container_width=True)
    buf = io.BytesIO()
    fig_box.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("⬇️ Download Boxplot", data=buf.getvalue(),
                       file_name="boxplot.png", mime="image/png")

with tab_bar:
    fig_bar = buat_barplot(deskriptif_pivot, kolom_nilai)
    st.pyplot(fig_bar, use_container_width=True)
    buf2 = io.BytesIO()
    fig_bar.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
    st.download_button("⬇️ Download Barplot", data=buf2.getvalue(),
                       file_name="barplot.png", mime="image/png")


# ══════════════════════════════════════════════════════════════════════════
# UJI NORMALITAS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 🔬 Uji Normalitas — Shapiro-Wilk")
st.caption("H₀: Data terdistribusi normal | Ditolak jika p < α")

hasil_normal = uji_normalitas(grup_data)
semua_normal = all(v.get("normal", False) for v in hasil_normal.values()
                   if v.get("normal") is not None)

rows_normal = []
for grup, res in hasil_normal.items():
    if res.get("error"):
        rows_normal.append({"Kelompok": grup, "Statistik W": "-",
                             "p-value": "-", "Kesimpulan": res["error"]})
    else:
        badge = "Normal ✓" if res["normal"] else "Tidak Normal ✗"
        rows_normal.append({"Kelompok": grup,
                             "Statistik W": res["stat"],
                             "p-value": res["p"],
                             "Kesimpulan": badge})

df_normal = pd.DataFrame(rows_normal)

def color_kesimpulan(val):
    if "✓" in str(val):
        return "background-color: #d1fae5; color: #065f46; font-weight: 600"
    elif "✗" in str(val):
        return "background-color: #fee2e2; color: #991b1b; font-weight: 600"
    return ""

st.dataframe(df_normal.style.applymap(color_kesimpulan, subset=["Kesimpulan"]),
             use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# UJI HOMOGENITAS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 🔬 Uji Homogenitas — Levene")
st.caption("H₀: Varians antar kelompok homogen | Ditolak jika p < α")

hasil_homo = uji_homogenitas(grup_data)

if hasil_homo.get("error"):
    st.warning(hasil_homo["error"])
    homogen = False
else:
    homogen = hasil_homo["homogen"]
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Statistik Levene", hasil_homo["stat"])
    col_h2.metric("p-value", hasil_homo["p"])
    kesimpulan_homo = "Homogen ✓" if homogen else "Tidak Homogen ✗"
    col_h3.markdown(
        f"**Kesimpulan**<br>"
        f"<span class='{'badge-pass' if homogen else 'badge-fail'}'>"
        f"{kesimpulan_homo}</span>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# UJI BEDA
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### ⚖️ Uji Beda")

hasil_beda = uji_beda(grup_data, semua_normal, homogen)

if hasil_beda.get("error"):
    st.error(hasil_beda["error"])
else:
    signifikan = hasil_beda.get("signifikan", False)
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Uji yang Digunakan", hasil_beda["uji"])
    col_b2.metric("Statistik", hasil_beda["stat"])
    col_b3.metric("p-value", hasil_beda["p"])
    col_b4.markdown(
        f"**Kesimpulan**<br>"
        f"<span class='{'badge-pass' if signifikan else 'badge-fail'}'>"
        f"{'Signifikan ✓' if signifikan else 'Tidak Signifikan ✗'}</span>",
        unsafe_allow_html=True)

    st.markdown(
        f"**Pemilihan uji:** Data {'normal' if semua_normal else 'tidak normal'} + "
        f"varians {'homogen' if homogen else 'tidak homogen'} → **{hasil_beda['uji']}**"
    )


# ══════════════════════════════════════════════════════════════════════════
# NARASI OTOMATIS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### 📝 Narasi Hasil Otomatis")

if not hasil_beda.get("error"):
    narasi = buat_narasi(deskriptif_pivot, hasil_normal, hasil_homo,
                         hasil_beda, kolom_grup, kolom_nilai)
    st.markdown(f"<div class='narasi-box'>{narasi}</div>", unsafe_allow_html=True)

    st.download_button("⬇️ Download Narasi (.txt)",
                       data=narasi.replace("**", ""),
                       file_name="narasi_statpharm.txt",
                       mime="text/plain")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.caption("⚠️ StatPharm Insight adalah alat bantu analisis. "
           "Interpretasi akhir tetap memerlukan pertimbangan peneliti.")
