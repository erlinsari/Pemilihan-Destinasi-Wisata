# ─────────────────────────────────────────────
#  CSS CUSTOM (UPDATE TERBARU)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* General Styling */
    .stApp { background-color: #f8f9fa; }
    
    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Title/Header Styling */
    .main-title {
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #2d3436;
        margin-bottom: 0.5rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        font-weight: 600;
        color: #636e72;
    }
    .stTabs [aria-selected="true"] { color: #0984e3 !important; }

    /* Button Styling */
    div.stButton > button {
        border-radius: 8px;
        border: none;
        background-color: #0984e3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN UI (Perubahan Struktur)
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🏝️ WisataInsight Indonesia</div>', unsafe_allow_html=True)
st.subheader("Sistem Pendukung Keputusan Destinasi Wisata")
st.markdown("Analisis keputusan multi-kriteria untuk rekomendasi terbaik.")

# ... (kode backend Anda tetap di sini) ...

# ══════════════════════════════════════════════
#  TAB: OVERVIEW (UI Ditingkatkan)
# ══════════════════════════════════════════════
with tab_overview:
    # Menggunakan columns untuk statistik singkat
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Data", len(df_work))
    with c2: st.metric("Kota Tersedia", df_work['City'].nunique())
    with c3: st.metric("Rating Max", f"{df_work['Rating'].max():.1f}")
    with c4: st.metric("Harga Terendah", f"Rp {df_work['Price'].min():,.0f}")

    st.divider()
    
    st.markdown("### 🏆 Top 3 Rekomendasi Utama")
    top3 = df_final.head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.info(f"**{row['Place_Name']}**\n\nRank: {i+1}\nScore: {row['TOPSIS_Score']:.3f}")
