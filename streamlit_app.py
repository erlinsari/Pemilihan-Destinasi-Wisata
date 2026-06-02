"""
Streamlit App - SPK Pemilihan Destinasi Wisata Indonesia
Metode: SMART + SAW + TOPSIS
Dataset: Indonesia Tourism Destination (Kaggle)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import spearmanr
import io
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Destinasi Wisata Indonesia",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS CUSTOM (ANTI TEKS PUTIH / DARK MODE CONFLICT FIX)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Paksa latar belakang utama aplikasi berwarna terang */
    .stApp {
        background-color: #f8f9fa !important;
    }
    
    /* FIX MUTLAK: Paksa semua teks bawaan Streamlit di area utama berwarna gelap agar tidak gaib */
    .stApp .main .stMarkdown p, 
    .stApp .main .stMarkdown h1, 
    .stApp .main .stMarkdown h2, 
    .stApp .main .stMarkdown h3, 
    .stApp .main .stMarkdown h4, 
    .stApp .main .stMarkdown h5, 
    .stApp .main .stMarkdown h6, 
    .stApp .main .stMarkdown li, 
    .stApp .main .stMarkdown span,
    .stApp .main p,
    .stApp .main h1,
    .stApp .main h2,
    .stApp .main h3 {
        color: #1a1a2e !important;
    }
    
    /* Perbaikan teks komponen Metric bawaan */
    [data-testid="stMetricLabel"] div {
        color: #555555 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #1a1a2e !important;
    }
    
    /* Perbaikan teks komponen Tab bawaan */
    .stTabs [data-baseweb="tab"] p {
        color: #555555 !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #667eea !important;
        font-weight: bold;
    }
    
    /* Perbaikan teks info/warning alert box agar kontras */
    div[data-testid="stAlert"] p {
        color: #721c24 !important;
    }

    /* Style khusus judul & penjelasan */
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #1a1a2e !important;
        text-align: center; padding: 1rem 0 0.3rem;
    }
    .sub-title {
        font-size: 1rem; color: #555555 !important; text-align: center; margin-bottom: 1.5rem;
    }
    .method-box {
        background: #f8f9ff; border-left: 4px solid #667eea;
        padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0;
        color: #1a1a2e !important;
    }
    .method-box * {
        color: #1a1a2e !important;
    }
    
    /* Sidebar styling tetap gelap elegan */
    div[data-testid="stSidebar"] {background: #1a1a2e;}
    div[data-testid="stSidebar"] * {color: #e0e0e0 !important;}
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label {color: #adb5bd !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background: #f0f2f6; padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load dan preprocess dataset Indonesia Tourism Destination."""
    possible_paths = [
        '/kaggle/input/indonesia-tourism-destination/tourism_with_id.csv',
        'tourism_with_id.csv',
        'data/tourism_with_id.csv',
    ]
    rating_paths = [
        '/kaggle/input/indonesia-tourism-destination/tourism_rating.csv',
        'tourism_rating.csv',
        'data/tourism_rating.csv',
    ]

    df_main, df_rating = None, None
    for p in possible_paths:
        if os.path.exists(p):
            df_main = pd.read_csv(p)
            break
    for p in rating_paths:
        if os.path.exists(p):
            df_rating = pd.read_csv(p)
            break

    if df_main is None:
        # Generate realistic sample data
        np.random.seed(42)
        place_templates = {
            'Jakarta': ['Monas','Kota Tua','Ancol','TMII','Ragunan','Kepulauan Seribu','Museum Nasional','Museum Fatahillah'],
            'Bandung': ['Kawah Putih','Tangkuban Perahu','Trans Studio','Lembang','Dusun Bambu','Situ Patenggang','Farmhouse','Orchid Forest'],
            'Yogyakarta': ['Prambanan','Borobudur','Keraton Yogyakarta','Malioboro','Pantai Parangtritis','Goa Jomblang','Kalibiru','Merapi'],
            'Bali': ['Tanah Lot','Ubud Palace','Kuta Beach','GWK Bali','Seminyak','Uluwatu Temple','Tirta Gangga','Tegallalang'],
            'Surabaya': ['Kebun Binatang Surabaya','House of Sampoerna','Tugu Pahlawan','Pantai Kenjeran','Ciputra Waterpark','Suramadu Bridge','G-Walk','Monkasel']
        }
        records, ratings_list = [], []
        pid = 1
        cat_choices = ['Budaya','Alam','Taman Hiburan','Bahari']
        for city, places in place_templates.items():
            for place in places:
                price = np.random.choice([0,5000,10000,15000,20000,25000,30000,50000,75000,100000])
                rating = round(np.random.uniform(3.5, 5.0), 1)
                time_m = np.random.choice([60,90,120,150,180,240,300])
                n_rev = np.random.randint(80, 600)
                
                # Perbaikan: Distribusi kategori siklik agar filter kombinasi (misal Bali + Alam) tidak kosong bawaan
                assigned_cat = cat_choices[pid % 4]
                
                records.append({'Place_Id':pid,'Place_Name':place,'Category':assigned_cat,
                                'City':city,'Price':price,'Rating':rating,'Time_Minutes':time_m})
                for _ in range(n_rev):
                    ratings_list.append({'Place_Id':pid,'Place_Ratings':np.random.randint(1,6)})
                pid += 1
        df_main = pd.DataFrame(records)
        df_rating = pd.DataFrame(ratings_list)

    if df_rating is None:
        df_rating = pd.DataFrame({'Place_Id': df_main['Place_Id'], 'Place_Ratings': 4})

    # Hitung jumlah review per tempat
    review_count = df_rating.groupby('Place_Id').size().reset_index(name='Jumlah_Review')
    df = df_main.merge(review_count, on='Place_Id', how='left')
    df['Jumlah_Review'] = df['Jumlah_Review'].fillna(0).astype(int)
    df['Time_Minutes'] = pd.to_numeric(df['Time_Minutes'], errors='coerce').fillna(60)
    df = df[['Place_Id','Place_Name','Category','City','Price','Rating','Jumlah_Review','Time_Minutes']].dropna(subset=['Price','Rating'])
    return df


# ─────────────────────────────────────────────
#  SPK METHODS
# ─────────────────────────────────────────────
def normalize_saw(X, criteria, benefit):
    R = np.zeros_like(X, dtype=float)
    for j, c in enumerate(criteria):
        col = X[:, j]
        if benefit[c]:
            mx = col.max()
            R[:, j] = col / mx if mx != 0 else col
        else:
            mn = col.min()
            if mn == 0:
                nz = col[col > 0]
                pm = nz.min() if len(nz) > 0 else 1
                R[:, j] = np.where(col == 0, 1.0, pm / col)
            else:
                R[:, j] = mn / col
    return R


def saw_method(df, criteria, weights, benefit):
    X = df[criteria].values.astype(float)
    R = normalize_saw(X, criteria, benefit)
    w = np.array([weights[c] for c in criteria])
    scores = R.dot(w)
    res = df[['Place_Name','City','Category'] + criteria].copy()
    res['SAW_Score'] = np.round(scores, 4)
    res['SAW_Rank'] = res['SAW_Score'].rank(ascending=False, method='min').astype(int)
    return res.sort_values('SAW_Rank').reset_index(drop=True)


def topsis_method(df, criteria, weights, benefit):
    X = df[criteria].values.astype(float)
    denom = np.sqrt((X**2).sum(axis=0))
    denom = np.where(denom == 0, 1, denom)
    R = X / denom
    w = np.array([weights[c] for c in criteria])
    V = R * w
    A_pos = np.array([V[:,j].max() if benefit[c] else V[:,j].min() for j, c in enumerate(criteria)])
    A_neg = np.array([V[:,j].min() if benefit[c] else V[:,j].max() for j, c in enumerate(criteria)])
    D_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    dci = D_pos + D_neg
    dci = np.where(dci == 0, 1e-10, dci)
    C = D_neg / dci
    res = df[['Place_Name','City','Category'] + criteria].copy()
    res['D_pos'] = np.round(D_pos, 4)
    res['D_neg'] = np.round(D_neg, 4)
    res['TOPSIS_Score'] = np.round(C, 4)
    res['TOPSIS_Rank'] = res['TOPSIS_Score'].rank(ascending=False, method='min').astype(int)
    return res.sort_values('TOPSIS_Rank').reset_index(drop=True)


def smart_method(df, criteria, weights, benefit):
    X = df[criteria].values.astype(float)
    U = np.zeros_like(X, dtype=float)
    for j, c in enumerate(criteria):
        col = X[:, j]
        mn, mx = col.min(), col.max()
        rng = mx - mn
        if rng == 0:
            U[:, j] = 100.0
        elif benefit[c]:
            U[:, j] = (col - mn) / rng * 100
        else:
            U[:, j] = (mx - col) / rng * 100
    w = np.array([weights[c] for c in criteria])
    scores = U.dot(w)
    res = df[['Place_Name','City','Category'] + criteria].copy()
    res['SMART_Score'] = np.round(scores, 4)
    res['SMART_Rank'] = res['SMART_Score'].rank(ascending=False, method='min').astype(int)
    return res.sort_values('SMART_Rank').reset_index(drop=True)


def combine_results(df_base, df_saw, df_topsis, df_smart, criteria):
    df_c = df_base[['Place_Name','City','Category'] + criteria].copy()
    df_c = df_c.merge(df_saw[['Place_Name','SAW_Score','SAW_Rank']], on='Place_Name', how='left')
    df_c = df_c.merge(df_topsis[['Place_Name','TOPSIS_Score','TOPSIS_Rank']], on='Place_Name', how='left')
    df_c = df_c.merge(df_smart[['Place_Name','SMART_Score','SMART_Rank']], on='Place_Name', how='left')
    df_c['Avg_Rank'] = (df_c['SAW_Rank'] + df_c['TOPSIS_Rank'] + df_c['SMART_Rank']) / 3
    df_c['Final_Rank'] = df_c['Avg_Rank'].rank(method='min').astype(int)
    return df_c.sort_values('Final_Rank').reset_index(drop=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏝️ SPK Wisata")
    st.markdown("---")
    st.markdown("### 🔍 Filter Data")

    df_all = load_data()
    all_cities = ['Semua'] + sorted(df_all['City'].unique().tolist())
    selected_city = st.selectbox("Pilih Kota", all_cities)

    all_categories = ['Semua'] + sorted(df_all['Category'].unique().tolist())
    selected_cat = st.selectbox("Pilih Kategori", all_categories)

    top_n = st.slider("Jumlah Alternatif (Top N)", min_value=10, max_value=min(100, len(df_all)), value=40, step=5)

    st.markdown("---")
    st.markdown("### ⚖️ Bobot Kriteria")
    st.caption("Total bobot harus = 100%")

    w_price  = st.slider("Harga Tiket (Cost ↓)",       5, 60, 30, 5)
    w_rating = st.slider("Rating Pengunjung (Benefit ↑)", 5, 60, 35, 5)
    w_review = st.slider("Jumlah Review (Benefit ↑)",  5, 50, 20, 5)
    w_time   = st.slider("Waktu Kunjungan (Benefit ↑)", 5, 40, 15, 5)

    total_w = w_price + w_rating + w_review + w_time
    if total_w != 100:
        st.warning(f"⚠️ Total bobot = {total_w}% (harus 100%)")
    else:
        st.success("✅ Total bobot = 100%")

    st.markdown("---")
    st.caption("📌 Dataset: [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🏝️ WisataInsight Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Pemilihan Destinasi Wisata Indonesia • SMART + SAW + TOPSIS</div>', unsafe_allow_html=True)

if total_w != 100:
    st.error("⚠️ Sesuaikan bobot di sidebar hingga totalnya = 100% untuk menjalankan analisis.")
    st.stop()

# Prepare data & weights
CRITERIA = ['Price', 'Rating', 'Jumlah_Review', 'Time_Minutes']
CRITERIA_LABELS = {
    'Price': 'Harga Tiket (Rp)',
    'Rating': 'Rating',
    'Jumlah_Review': 'Jumlah Review',
    'Time_Minutes': 'Waktu (menit)'
}
BENEFIT = {'Price': False, 'Rating': True, 'Jumlah_Review': True, 'Time_Minutes': True}
WEIGHTS = {
    'Price': w_price/100,
    'Rating': w_rating/100,
    'Jumlah_Review': w_review/100,
    'Time_Minutes': w_time/100,
}

# Filter
df_work = df_all.copy()
if selected_city != 'Semua':
    df_work = df_work[df_work['City'] == selected_city]
if selected_cat != 'Semua':
    df_work = df_work[df_work['Category'] == selected_cat]

df_work = df_work.nlargest(top_n, 'Rating').reset_index(drop=True)

if len(df_work) < 3:
    st.warning("⚠️ Data hasil filter terlalu sedikit (kurang dari 3). Silakan ubah filter Kota atau Kategori di sidebar.")
    st.stop()

# Run methods
df_saw    = saw_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_topsis = topsis_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_smart  = smart_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_final  = combine_results(df_work, df_saw, df_topsis, df_smart, CRITERIA)


# ─────────────────────────────────────────────
#  INITIALIZE TABS
# ─────────────────────────────────────────────
tab_overview, tab_saw, tab_topsis, tab_smart, tab_compare, tab_data = st.tabs([
    "📊 Overview", "📐 SAW", "🎯 TOPSIS", "⭐ SMART", "🔄 Perbandingan", "📁 Data"
])


# ══════════════════════════════════════════════
#  TAB: OVERVIEW
# ══════════════════════════════════════════════
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Total Destinasi", len(df_work))
    with col2:
        st.metric("🏙️ Kota", df_work['City'].nunique())
    with col3:
        st.metric("⭐ Rating Tertinggi", f"{df_work['Rating'].max():.1f}")
    with col4:
        st.metric("🎫 Harga Min (Rp)", f"{df_work['Price'].min():,.0f}")

    st.markdown("---")

    # Top 5 cards
    st.markdown("### 🏆 Top 5 Destinasi Terbaik (Gabungan 3 Metode)")
    top5 = df_final.head(5)
    cols = st.columns(5)
    medal = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{'#FFD700' if i==0 else '#C0C0C0' if i==1 else '#CD7F32' if i==2 else '#667eea'} 0%,{'#FFA500' if i==0 else '#A9A9A9' if i==1 else '#8B4513' if i==2 else '#764ba2'} 100%);
                 padding:1rem;border-radius:12px;color:white !important;text-align:center;min-height:180px">
                <div style="font-size:1.8rem;color:white !important;">{medal[i]}</div>
                <div style="font-weight:800;font-size:0.9rem;margin:4px 0;color:white !important;">{row['Place_Name'][:20]}</div>
                <div style="font-size:0.75rem;opacity:0.85;color:white !important;">📍 {row['City']}</div>
                <div style="font-size:0.75rem;margin-top:6px;color:white !important;">TOPSIS: <b>{row['TOPSIS_Score']:.3f}</b></div>
                <div style="font-size:0.75rem;color:white !important;">⭐ {row['Rating']:.1f} | Rp{row['Price']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("### 📊 Distribusi per Kota")
        fig, ax = plt.subplots(figsize=(6, 4))
        city_cnt = df_work['City'].value_counts()
        bars = ax.bar(city_cnt.index, city_cnt.values,
                      color=sns.color_palette('Set2', len(city_cnt)))
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                    str(int(b.get_height())), ha='center', fontweight='bold')
        ax.set_title('Jumlah Destinasi per Kota')
        ax.set_ylabel('Jumlah')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### 🎫 Distribusi Harga Tiket")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_work['Price'], bins=15, color='#667eea', edgecolor='white', alpha=0.85)
        ax.set_title('Distribusi Harga Tiket')
        ax.set_xlabel('Harga (Rp)')
        ax.set_ylabel('Frekuensi')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Bobot
    st.markdown("### ⚖️ Bobot Kriteria (SMART Framework)")
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        bobot_df = pd.DataFrame({
            'Kriteria': [CRITERIA_LABELS[c] for c in CRITERIA],
            'Tipe': ['Cost ↓' if not BENEFIT[c] else 'Benefit ↑' for c in CRITERIA],
            'Bobot': [f"{WEIGHTS[c]:.0%}" for c in CRITERIA]
        })
        st.dataframe(bobot_df, hide_index=True, use_container_width=True)
    with col_w2:
        fig, ax = plt.subplots(figsize=(7, 3))
        labels = [CRITERIA_LABELS[c] for c in CRITERIA]
        vals   = [WEIGHTS[c] for c in CRITERIA]
        colors_b = ['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A']
        bars = ax.barh(labels, vals, color=colors_b, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f'{v:.0%}', va='center', fontweight='bold')
        ax.set_xlim(0, max(vals)*1.2)
        ax.set_title('Bobot Kriteria')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
#  TAB: SAW
# ══════════════════════════════════════════════
with tab_saw:
    st.markdown("## 📐 Metode SAW (Simple Additive Weighting)")
    st.markdown("""
    <div class="method-box">
    <b>Prinsip:</b> Normalisasi matriks keputusan → kalikan bobot → jumlahkan.<br>
    • <b>Benefit (↑)</b>: rᵢⱼ = xᵢⱼ / max(xⱼ)<br>
    • <b>Cost (↓)</b>: rᵢⱼ = min(xⱼ) / xᵢⱼ<br>
    • <b>Skor SAW</b>: Vᵢ = Σ(wⱼ × rᵢⱼ)
    </div>
    """, unsafe_allow_html=True)

    top_n_show = st.slider("Tampilkan Top N", 5, min(30, len(df_saw)), 10, key='saw_slider')

    display_saw = df_saw.head(top_n_show)[
        ['SAW_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','SAW_Score']
    ].rename(columns={'SAW_Rank':'Rank', 'SAW_Score':'Skor SAW'})
    st.dataframe(display_saw.style.background_gradient(subset=['Skor SAW'], cmap='Blues'),
                 hide_index=True, use_container_width=True)

    st.markdown("### 📊 Top 10 - Skor SAW")
    fig, ax = plt.subplots(figsize=(10, 5))
    top10_saw = df_saw.head(10)
    bars = ax.barh(top10_saw['Place_Name'].str[:22], top10_saw['SAW_Score'],
                   color='#2196F3', edgecolor='white', alpha=0.85)
    for bar, val in zip(bars, top10_saw['SAW_Score']):
        ax.text(bar.get_width()*0.5, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    ax.set_title('Top 10 Destinasi - Skor SAW', fontweight='bold')
    ax.set_xlabel('Skor SAW')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  TAB: TOPSIS
# ══════════════════════════════════════════════
with tab_topsis:
    st.markdown("## 🎯 Metode TOPSIS")
    st.markdown("""
    <div class="method-box">
    <b>Prinsip:</b> Pilih alternatif yang paling dekat ke solusi ideal positif dan terjauh dari ideal negatif.<br>
    • <b>Normalisasi</b>: rᵢⱼ = xᵢⱼ / √(Σxᵢⱼ²)<br>
    • <b>Ideal Positif (A⁺)</b>: max untuk benefit, min untuk cost<br>
    • <b>Skor TOPSIS</b>: Cᵢ = D⁻ / (D⁺ + D⁻)
    </div>
    """, unsafe_allow_html=True)

    top_n_t = st.slider("Tampilkan Top N", 5, min(30, len(df_topsis)), 10, key='topsis_slider')

    display_topsis = df_topsis.head(top_n_t)[
        ['TOPSIS_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','D_pos','D_neg','TOPSIS_Score']
    ].rename(columns={'TOPSIS_Rank':'Rank','TOPSIS_Score':'Skor TOPSIS','D_pos':'D⁺','D_neg':'D⁻'})
    st.dataframe(display_topsis.style.background_gradient(subset=['Skor TOPSIS'], cmap='Greens'),
                 hide_index=True, use_container_width=True)

    # Scatter D+ vs D-
    st.markdown("### 🗺️ Jarak ke Ideal Positif vs Negatif")
    fig, ax = plt.subplots(figsize=(9, 5))
    top20 = df_topsis.head(20)
    rest  = df_topsis.iloc[20:]
    ax.scatter(rest['D_pos'], rest['D_neg'], alpha=0.3, color='#ccc', label='Lainnya', s=40)
    sc = ax.scatter(top20['D_pos'], top20['D_neg'], c=top20['TOPSIS_Score'],
                    cmap='RdYlGn', s=100, zorder=5, edgecolors='white')
    for _, row in top20.head(5).iterrows():
        ax.annotate(row['Place_Name'][:16], (row['D_pos'], row['D_neg']),
                    textcoords='offset points', xytext=(5, 5), fontsize=7.5, color='#333')
    plt.colorbar(sc, ax=ax, label='Skor TOPSIS')
    ax.set_xlabel('Jarak ke Ideal Positif (D⁺) → lebih kecil lebih baik')
    ax.set_ylabel('Jarak ke Ideal Negatif (D⁻) → lebih besar lebih baik')
    ax.set_title('Scatter: D⁺ vs D⁻ per Destinasi', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  TAB: SMART
# ══════════════════════════════════════════════
with tab_smart:
    st.markdown("## ⭐ Metode SMART (Simple Multi-Attribute Rating Technique)")
    st.markdown("""
    <div class="method-box">
    <b>Prinsip:</b> Normalisasi nilai ke skala 0–100 (utility function), lalu weighted sum.<br>
    • <b>Benefit (↑)</b>: uᵢⱼ = (xᵢⱼ − min) / (max − min) × 100<br>
    • <b>Cost (↓)</b>: uᵢⱼ = (max − xᵢⱼ) / (max − min) × 100<br>
    • <b>Skor SMART</b>: Vᵢ = Σ(wⱼ × uᵢⱼ)
    </div>
    """, unsafe_allow_html=True)

    top_n_s = st.slider("Tampilkan Top N", 5, min(30, len(df_smart)), 10, key='smart_slider')

    display_smart = df_smart.head(top_n_s)[
        ['SMART_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','SMART_Score']
    ].rename(columns={'SMART_Rank':'Rank','SMART_Score':'Skor SMART'})
    st.dataframe(display_smart.style.background_gradient(subset=['Skor SMART'], cmap='Oranges'),
                 hide_index=True, use_container_width=True)

    st.markdown("### 📊 Top 10 - Skor SMART")
    fig, ax = plt.subplots(figsize=(10, 5))
    top10_sm = df_smart.head(10)
    bars = ax.barh(top10_sm['Place_Name'].str[:22], top10_sm['SMART_Score'],
                   color='#FF5722', edgecolor='white', alpha=0.85)
    for bar, val in zip(bars, top10_sm['SMART_Score']):
        ax.text(bar.get_width()*0.5, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    ax.set_title('Top 10 Destinasi - Skor SMART', fontweight='bold')
    ax.set_xlabel('Skor SMART (0-100)')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  TAB: PERBANDINGAN
# ══════════════════════════════════════════════
with tab_compare:
    st.markdown("## 🔄 Perbandingan Ketiga Metode")

    # Metrics
    c1, c2, c3 = st.columns(3)
    top_saw    = df_saw.iloc[0]['Place_Name']
    top_topsis = df_topsis.iloc[0]['Place_Name']
    top_smart  = df_smart.iloc[0]['Place_Name']
    c1.metric("🏆 Best SAW",    top_saw)
    c2.metric("🏆 Best TOPSIS", top_topsis)
    c3.metric("🏆 Best SMART",  top_smart)

    st.markdown("---")

    # Tabel perbandingan
    st.markdown("### 📋 Tabel Perbandingan Ranking Top 15")
    show_cols = ['Final_Rank','Place_Name','City','SAW_Rank','TOPSIS_Rank','SMART_Rank','Avg_Rank','SAW_Score','TOPSIS_Score','SMART_Score']
    df_show = df_final.head(15)[show_cols].copy()
    df_show.columns = ['Final Rank','Destinasi','Kota','Rank SAW','Rank TOPSIS','Rank SMART','Avg Rank','Skor SAW','Skor TOPSIS','Skor SMART']
    st.dataframe(df_show.style.background_gradient(subset=['Skor TOPSIS'], cmap='Blues'),
                 hide_index=True, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🌡️ Heatmap Ranking")
        top15 = df_final.head(15)[['Place_Name','SAW_Rank','TOPSIS_Rank','SMART_Rank']].copy()
        top15 = top15.set_index('Place_Name')
        top15.index = top15.index.str[:18]
        fig, ax = plt.subplots(figsize=(6, 7))
        sns.heatmap(top15, annot=True, fmt='.0f', cmap='YlOrRd_r',
                    linewidths=0.5, ax=ax,
                    cbar_kws={'label':'Ranking'})
        ax.set_title('Ranking per Metode (Top 15)', fontweight='bold')
        ax.set_xticklabels(['SAW','TOPSIS','SMART'])
        plt.yticks(fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### 📈 Korelasi Spearman")
        pairs = [('SAW_Rank','TOPSIS_Rank','SAW vs TOPSIS'),
                 ('SAW_Rank','SMART_Rank','SAW vs SMART'),
                 ('TOPSIS_Rank','SMART_Rank','TOPSIS vs SMART')]
        corr_data = []
        for a, b, label in pairs:
            r, p = spearmanr(df_final[a], df_final[b])
            corr_data.append({'Perbandingan': label, 'r Spearman': round(r,4), 'p-value': round(p,4),
                              'Status': '✅ Konsisten' if r > 0.8 else '⚠️ Berbeda'})
        st.dataframe(pd.DataFrame(corr_data), hide_index=True, use_container_width=True)

        st.markdown("#### Skor per Metode (Top 10)")
        fig, ax = plt.subplots(figsize=(7, 5))
        top10 = df_final.head(10)
        x = np.arange(len(top10))
        w = 0.25
        # Normalize SMART to 0-1 for comparison
        smart_norm = top10['SMART_Score'] / 100
        ax.bar(x - w, top10['SAW_Score'],    w, label='SAW',    color='#2196F3', alpha=0.85)
        ax.bar(x,     top10['TOPSIS_Score'], w, label='TOPSIS', color='#4CAF50', alpha=0.85)
        ax.bar(x + w, smart_norm,            w, label='SMART/100', color='#FF5722', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(top10['Place_Name'].str[:12], rotation=35, ha='right', fontsize=8)
        ax.set_title('Perbandingan Skor Ternormalisasi', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Export
    st.markdown("---")
    st.markdown("### 💾 Download Hasil")
    col_d1, col_d2, col_d3 = st.columns(3)

    def to_csv_bytes(df):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode('utf-8')

    with col_d1:
        st.download_button("📥 Download SAW (CSV)",    to_csv_bytes(df_saw),    "hasil_saw.csv",    "text/csv")
    with col_d2:
        st.download_button("📥 Download TOPSIS (CSV)", to_csv_bytes(df_topsis), "hasil_topsis.csv", "text/csv")
    with col_d3:
        st.download_button("📥 Download Perbandingan", to_csv_bytes(df_final),  "hasil_spk.csv",    "text/csv")


# ══════════════════════════════════════════════
#  TAB: DATA
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("## 📁 Dataset Mentah")
    st.info(f"Menampilkan {len(df_work)} destinasi setelah filter")

    search = st.text_input("🔎 Cari destinasi...", "")
    df_display = df_work.copy()
    if search:
        df_display = df_display[df_display['Place_Name'].str.contains(search, case=False)]

    st.dataframe(df_display[['Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes']],
                 hide_index=True, use_container_width=True, height=450)

    st.markdown("### 📊 Statistik Deskriptif")
    st.dataframe(df_work[CRITERIA].describe().round(2), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.82rem">
    SPK Pemilihan Destinasi Wisata Indonesia &bull; Metode: SMART + SAW + TOPSIS &bull;
    Dataset: <a href="https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination" target="_blank">Kaggle</a>
</div>
""", unsafe_allow_html=True)
