"""
Streamlit App - SPK Pemilihan Destinasi Wisata Indonesia
Metode: SMART + SAW + TOPSIS
Dataset: Indonesia Tourism Destination (Kaggle)
Optimized UI/UX: Pastel Blue Sidebar & High Contrast Visibility (No Powered Text)
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
#  CONFIG & THEME INITIALIZATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wonderful Indonesia - SPK Destinasi Wisata",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling untuk Matplotlib agar sesuai dengan tema website professional
plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': '#1e293b',
    'axes.labelcolor': '#475569',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'axes.edgecolor': '#e2e8f0',
    'grid.color': '#f1f5f9',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff'
})

# ─────────────────────────────────────────────
#  CSS CUSTOM (PASTEL BLUE SIDEBAR & HIGH CONTRAST)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Typography */
    html, body, [data-testid="stSidebar"] * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* FIX TEXT KONTRAS UTAMA (Mencegah teks hilang/putih di area putih) */
    [data-testid="stMainBlock"] h1, 
    [data-testid="stMainBlock"] h2, 
    [data-testid="stMainBlock"] h3, 
    [data-testid="stMainBlock"] h4,
    [data-testid="stMainBlock"] .stMarkdown p {
        color: #0f172a !important;
    }
    
    /* Header Section */
    .brand-badge {
        background: linear-gradient(135deg, #0d9488 0%, #0ea5e9 100%);
        color: white !important; padding: 6px 16px; border-radius: 50px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; display: inline-block; margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.6rem; font-weight: 800; color: #0f172a;
        text-align: center; letter-spacing: -0.5px; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 2.5rem; font-weight: 400;
    }
    
    /* SIDEBAR BARU: PASTEUR BLUE THEME WITH DARK TEXT */
    div[data-testid="stSidebar"] {
        background-color: #e0f2fe !important; /* Biru Pastel Terang */
        border-right: 1px solid #bae6fd;
    }
    /* Memaksa semua tulisan di dalam sidebar berwarna gelap agar kontras */
    div[data-testid="stSidebar"] h2, 
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h4 {
        color: #0369a1 !important; /* Biru deep untuk judul menu */
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] .stCaption {
        color: #1e293b !important; /* Abu gelap/hitam untuk teks biasa & label */
        font-weight: 600 !important;
    }
    /* Warna teks di dalam form drop-down/select-box sidebar */
    div[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #1e293b !important;
    }
    
    /* Travel Destination Cards (Top 5) */
    .tour-card {
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #f1f5f9;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .tour-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 10px 10px -5px rgba(15, 23, 42, 0.04);
        border-color: #e2e8f0;
    }
    .card-badge {
        position: absolute; top: 12px; left: 12px;
        padding: 4px 12px; border-radius: 50px; font-weight: 700; font-size: 0.75rem;
    }
    .badge-1 { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
    .badge-2 { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .badge-3 { background: #ffedd5; color: #ea580c; border: 1px solid #fed7aa; }
    .badge-general { background: #f0f9ff; color: #0284c7; border: 1px solid #bae6fd; }
    
    /* Method Info Box */
    .method-box {
        background: white; border-left: 5px solid #0d9488;
        padding: 1.2rem 1.5rem; border-radius: 12px; margin: 1rem 0 1.5rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.02);
    }
    
    /* Tabs Custom Interface */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background-color: #e2e8f0; padding: 6px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; background: transparent; padding: 8px 22px;
        color: #475569; font-weight: 600; font-size: 0.9rem; transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a; background: rgba(255,255,255,0.5);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white; color: #0d9488 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
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
        for city, places in place_templates.items():
            for place in places:
                price = np.random.choice([0,5000,10000,15000,20000,25000,30000,50000,75000,100000])
                rating = round(np.random.uniform(3.5, 5.0), 1)
                time_m = np.random.choice([60,90,120,150,180,240,300])
                n_rev = np.random.randint(80, 600)
                records.append({'Place_Id':pid,'Place_Name':place,'Category':np.random.choice(['Budaya','Alam','Taman Hiburan','Bahari']),
                                'City':city,'Price':price,'Rating':rating,'Time_Minutes':time_m})
                for _ in range(n_rev):
                    ratings_list.append({'Place_Id':pid,'Place_Ratings':np.random.randint(1,6)})
                pid += 1
        df_main = pd.DataFrame(records)
        df_rating = pd.DataFrame(ratings_list)

    if df_rating is None:
        df_rating = pd.DataFrame({'Place_Id': df_main['Place_Id'], 'Place_Ratings': 4})

    review_count = df_rating.groupby('Place_Id').size().reset_index(name='Jumlah_Review')
    df = df_main.merge(review_count, on='Place_Id', how='left')
    df['Jumlah_Review'] = df['Jumlah_Review'].fillna(0).astype(int)
    df['Time_Minutes'] = pd.to_numeric(df['Time_Minutes'], errors='coerce').fillna(60)
    df = df[['Place_Id','Place_Name','Category','City','Price','Rating','Jumlah_Review','Time_Minutes']].dropna(subset=['Price','Rating'])
    return df


# ─────────────────────────────────────────────
#  SPK CORE ALGORITHMS
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
#  SIDEBAR MANAGEMENT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 NAVIGATION PANEL")
    st.markdown("---")
    st.markdown("### 🔍 Filter Destinasi")

    df_all = load_data()
    all_cities = ['Semua'] + sorted(df_all['City'].unique().tolist())
    selected_city = st.selectbox("Wilayah Kota", all_cities)

    all_categories = ['Semua'] + sorted(df_all['Category'].unique().tolist())
    selected_cat = st.selectbox("Kategori Wisata", all_categories)

    top_n = st.slider("Limit Rekomendasi (Top N)", min_value=10, max_value=min(100, len(df_all)), value=40, step=5)

    st.markdown("---")
    st.markdown("### ⚖️ Bobot Prioritas Kriteria")
    st.caption("Sesuaikan preferensi Anda (Total harus 100%)")

    w_price  = st.slider("Budget Tiket Masuk (Cost)", 5, 60, 30, 5)
    w_rating = st.slider("Rating Destinasi (Benefit)", 5, 60, 35, 5)
    w_review = st.slider("Popularitas/Review (Benefit)", 5, 50, 20, 5)
    w_time   = st.slider("Durasi Kunjungan (Benefit)", 5, 40, 15, 5)

    total_w = w_price + w_rating + w_review + w_time
    if total_w != 100:
        st.sidebar.error(f"⚠️ Total Bobot: {total_w}% (Harus 100%)")
    else:
        st.sidebar.success("✅ Konfigurasi Bobot Valid (100%)")

    st.markdown("---")


# ─────────────────────────────────────────────
#  MAIN DASHBOARD WRAPPER
# ─────────────────────────────────────────────
st.markdown('<div style="text-align: center;"><span class="brand-badge">Wonderful Indonesia Engine</span></div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Sistem Pendukung Keputusan Wisata</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rekomendasi Destinasi Terbaik Menggunakan Multi-Criteria Decision Making (SMART, SAW, TOPSIS)</div>', unsafe_allow_html=True)

if total_w != 100:
    st.error("🚨 **Error Konfigurasi:** Mohon sesuaikan kembali bobot kriteria pada panel kiri agar akumulasi bernilai tepat 100% untuk memulai analisis.")
    st.stop()

# Build Configuration Environment
CRITERIA = ['Price', 'Rating', 'Jumlah_Review', 'Time_Minutes']
BENEFIT = {'Price': False, 'Rating': True, 'Jumlah_Review': True, 'Time_Minutes': True}
WEIGHTS = {'Price': w_price/100, 'Rating': w_rating/100, 'Jumlah_Review': w_review/100, 'Time_Minutes': w_time/100}

# Filter Pipeline Execution
df_work = df_all.copy()
if selected_city != 'Semua':
    df_work = df_work[df_work['City'] == selected_city]
if selected_cat != 'Semua':
    df_work = df_work[df_work['Category'] == selected_cat]

df_work = df_work.nlargest(top_n, 'Rating').reset_index(drop=True)

if len(df_work) < 3:
    st.warning("⚠️ Data hasil pencarian terlalu sedikit. Silakan ubah cakupan kombinasi filter pada sidebar.")
    st.stop()

# Execution Algorithm Machine
df_saw    = saw_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_topsis = topsis_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_smart  = smart_method(df_work, CRITERIA, WEIGHTS, BENEFIT)
df_final  = combine_results(df_work, df_saw, df_topsis, df_smart, CRITERIA)


# ─────────────────────────────────────────────
#  TABS INTERFACE VIEW SYSTEM
# ─────────────────────────────────────────────
tab_overview, tab_saw, tab_topsis, tab_smart, tab_compare, tab_data = st.tabs([
    "📊 Overview Hub", "📐 Analisis SAW", "🎯 Analisis TOPSIS", "⭐ Analisis SMART", "🔄 Komparasi Rank", "📁 Exploratory Data"
])

# ══════════════════════════════════════════════
#  TAB: OVERVIEW HUB
# ══════════════════════════════════════════════
with tab_overview:
    # Metric KPI Section
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Total Alternatif Terfilter", value=f"{len(df_work)} Lokasi")
    with m_col2:
        st.metric(label="Klaster Kota Tercakup", value=f"{df_work['City'].nunique()} Wilayah")
    with m_col3:
        st.metric(label="Skor Tertinggi Kualifikasi", value=f"{df_work['Rating'].max():.1f} / 5.0")
    with m_col4:
        st.metric(label="Opsi Tiket Termurah", value=f"Rp {df_work['Price'].min():,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Top 5 Destinasi Rekomendasi Utama (Konsensus Multi-Metode)")
    
    # Premium Travel Cards Creation
    top5 = df_final.head(5)
    cols = st.columns(5)
    medals = ["🥇 Rank 1", "🥈 Rank 2", "🥉 Rank 3", "🏅 Rank 4", "🏅 Rank 5"]
    badge_classes = ["badge-1", "badge-2", "badge-3", "badge-general", "badge-general"]
    
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="tour-card">
                <div>
                    <span class="card-badge {badge_classes[i]}">{medals[i]}</span>
                    <br><br>
                    <h4 style="margin: 12px 0 6px 0; color: #0f172a; font-size: 1.1rem; font-weight:700;">{row['Place_Name']}</h4>
                    <p style="color: #64748b; font-size: 0.8rem; margin: 0;">📍 {row['City']} &bull; <span style="font-style: italic;">{row['Category']}</span></p>
                </div>
                <div style="margin-top: 20px; border-top: 1px solid #f1f5f9; padding-top: 12px;">
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px;">Konsistensi Multi-Skor</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0d9488;">{row['TOPSIS_Score']:.3f} <span style="font-size:0.7rem; font-weight:400; color:#64748b;">(Top)</span></div>
                    <div style="font-size: 0.75rem; color: #475569; margin-top: 4px;">⭐ {row['Rating']:.1f} | Rp {row['Price']:,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Analytical Distribution Visualizations
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.markdown("#### 🗺️ Ketersediaan Destinasi Berdasarkan Lokasi")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        city_cnt = df_work['City'].value_counts()
        bars = ax.bar(city_cnt.index, city_cnt.values, color='#0ea5e9', width=0.5, edgecolor='#0284c7', alpha=0.9)
        ax.bar_label(bars, padding=3, weight='bold', size=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with graph_col2:
        st.markdown("#### 🎫 Sebaran Skema Harga Tiket Masuk")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(df_work['Price'], bins=12, color='#10b981', edgecolor='white', alpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Range Harga (IDR)')
        ax.set_ylabel('Frekuensi Kemunculan')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
#  TAB: ANALISIS SAW
# ══════════════════════════════════════════════
with tab_saw:
    st.markdown("### 📐 Pendekatan Simple Additive Weighting (SAW)")
    st.markdown("""
    <div class="method-box">
        Metode SAW sering juga dikenal sebagai istilah metode penjumlahan terbobot. 
        Proses dasarnya memerlukan langkah pencarian <b>normalisasi matriks keputusan (R)</b> yang sebanding dengan semua kriteria yang ada 
        tergantung tipe karakteristik (Benefit vs Cost).
    </div>
    """, unsafe_allow_html=True)

    saw_top_n = st.slider("Tampilkan Data Teratas", 5, min(30, len(df_saw)), 10, key='saw_slider')
    
    df_saw_display = df_saw.head(saw_top_n)[
        ['SAW_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','SAW_Score']
    ].rename(columns={'SAW_Rank':'Peringkat', 'Place_Name':'Nama Destinasi', 'City':'Kota', 'Category':'Kategori', 'SAW_Score':'Skor Akhir SAW'})
    
    st.dataframe(df_saw_display.style.background_gradient(subset=['Skor Akhir SAW'], cmap='BuGn'), hide_index=True, use_container_width=True)

    # Chart Performance
    st.markdown("<br><b>Peta Komparasi 10 Besar Hasil Model SAW</b>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    saw_g = df_saw.head(10)
    ax.barh(saw_g['Place_Name'], saw_g['SAW_Score'], color='#0d9488', alpha=0.85, height=0.6)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  TAB: ANALISIS TOPSIS
# ══════════════════════════════════════════════
with tab_topsis:
    st.markdown("### 🎯 Pendekatan Technique for Order Preference by Similarity to Ideal Solution")
    st.markdown("""
    <div class="method-box">
        TOPSIS memiliki prinsip dasar bahwa alternatif yang dipilih harus memiliki <b>jarak terdekat dari solusi ideal positif ($D^+$)</b> 
        dan memiliki <b>jarak terjauh dari solusi ideal negatif ($D^-$)</b> dari sudut pandang geometris ruang Euclidean.
    </div>
    """, unsafe_allow_html=True)

    topsis_top_n = st.slider("Tampilkan Data Teratas", 5, min(30, len(df_topsis)), 10, key='topsis_slider')
    
    df_topsis_display = df_topsis.head(topsis_top_n)[
        ['TOPSIS_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','D_pos','D_neg','TOPSIS_Score']
    ].rename(columns={'TOPSIS_Rank':'Peringkat', 'Place_Name':'Nama Destinasi', 'TOPSIS_Score':'Kedekatan Relatif (V)'})
    
    st.dataframe(df_topsis_display.style.background_gradient(subset=['Kedekatan Relatif (V)'], cmap='YlGnBu'), hide_index=True, use_container_width=True)

    # Scatter Chart
    st.markdown("<br><b>Peta Distribusi Geometris Solusi Ideal ($D^+$ vs $D^-$)</b>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(df_topsis['D_pos'], df_topsis['D_neg'], c=df_topsis['TOPSIS_Score'], cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Jarak Solusi Ideal Positif (D+) - Semakin Kecil Semakin Baik')
    ax.set_ylabel('Jarak Solusi Ideal Negatif (D-) - Semakin Besar Semakin Baik')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  TAB: ANALISIS SMART
# ══════════════════════════════════════════════
with tab_smart:
    st.markdown("### ⭐ Pendekatan Simple Multi-Attribute Rating Technique")
    st.markdown("""
    <div class="method-box">
        SMART merupakan metode pengambilan keputusan multi-kriteria yang berdasarkan pada teori bahwa setiap alternatif 
        terdiri dari beberapa kriteria yang memiliki nilai-nilai dan <b>setiap kriteria memiliki bobot</b> yang menggambarkan seberapa penting ia dibandingkan kriteria lain.
    </div>
    """, unsafe_allow_html=True)

    smart_top_n = st.slider("Tampilkan Data Teratas", 5, min(30, len(df_smart)), 10, key='smart_slider')
    
    df_smart_display = df_smart.head(smart_top_n)[
        ['SMART_Rank','Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes','SMART_Score']
    ].rename(columns={'SMART_Rank':'Peringkat', 'Place_Name':'Nama Destinasi', 'SMART_Score':'Skor Nilai Utility'})
    
    st.dataframe(df_smart_display.style.background_gradient(subset=['Skor Nilai Utility'], cmap='OrRd'), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB: KOMPARASI PERBANDINGAN RANKING
# ══════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔄 Validasi & Komparasi Antar Algoritma")
    
    c_col1, c_col2, c_col3 = st.columns(3)
    c_col1.metric("Juara 1 Model SAW", df_saw.iloc[0]['Place_Name'])
    c_col2.metric("Juara 1 Model TOPSIS", df_topsis.iloc[0]['Place_Name'])
    c_col3.metric("Juara 1 Model SMART", df_smart.iloc[0]['Place_Name'])
    
    st.markdown("<br><b>Tabel Komparasi Matrix Peringkat (Top 15)</b>", unsafe_allow_html=True)
    show_cols = ['Final_Rank','Place_Name','City','SAW_Rank','TOPSIS_Rank','SMART_Rank','Avg_Rank']
    df_show = df_final.head(15)[show_cols].copy()
    df_show.columns = ['Rank Konsensus', 'Nama Destinasi', 'Lokasi Kota', 'Rank SAW', 'Rank TOPSIS', 'Rank SMART', 'Rerata Nilai Rank']
    
    st.dataframe(df_show.style.set_properties(**{'text-align': 'center'}).background_gradient(subset=['Rerata Nilai Rank'], cmap='Blues_r'), hide_index=True, use_container_width=True)

    # Spearman Correlation Analyzer
    st.markdown("<br><b>Uji Signifikansi Korelasi Peringkat (Spearman Rank Correlation)</b>", unsafe_allow_html=True)
    pairs = [('SAW_Rank','TOPSIS_Rank','SAW vs TOPSIS'), ('SAW_Rank','SMART_Rank','SAW vs SMART'), ('TOPSIS_Rank','SMART_Rank','TOPSIS vs SMART')]
    corr_data = []
    for a, b, label in pairs:
        r, p = spearmanr(df_final[a], df_final[b])
        corr_data.append({'Kombinasi Pengujian': label, 'Koefisien Spearman (r)': round(r, 4), 'P-Value': round(p, 5), 'Status Konvergen': '✅ Konsisten & Stabil' if r > 0.8 else '⚠️ Terdapat Deviasi'})
    st.dataframe(pd.DataFrame(corr_data), hide_index=True, use_container_width=True)

    # Export Report Button
    st.markdown("---")
    st.markdown("### 💾 Unduh Dokumen Laporan Hasil Keputusan (Spreadsheet)")
    def to_csv_bytes(df):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode('utf-8')

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.download_button("📥 Ekspor Output Matrix SAW", to_csv_bytes(df_saw), "laporan_saw.csv", "text/csv", use_container_width=True)
    with d_col2:
        st.download_button("📥 Ekspor Output Matrix TOPSIS", to_csv_bytes(df_topsis), "laporan_topsis.csv", "text/csv", use_container_width=True)
    with d_col3:
        st.download_button("📥 Gabungan Konsensus Final", to_csv_bytes(df_final), "rekomendasi_final_spk.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════
#  TAB: DATA EXPLORATORY
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("### 📁 Manajemen Data Mentah & Deskriptif")
    
    search_query = st.text_input("🔎 Ketik kata kunci nama tempat wisata untuk mencari...", "")
    df_display = df_work.copy()
    if search_query:
        df_display = df_display[df_display['Place_Name'].str.contains(search_query, case=False)]

    st.dataframe(df_display[['Place_Name','City','Category','Price','Rating','Jumlah_Review','Time_Minutes']], hide_index=True, use_container_width=True, height=350)
    
    st.markdown("#### 📈 Rangkuman Statistik Deskriptif Atribut Kriteria")
    st.dataframe(df_work[CRITERIA].describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 10px 0;">
    &copy; 2026 Wonderful Indonesia SPK Platform Hub &bull; Framework SMART, SAW, & TOPSIS Integration
</div>
""", unsafe_allow_html=True)
