# File: enhanced_app.py (dengan halaman Beranda elegan dan visualisasi yang proporsional)
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import ast
import csv
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from keybert import KeyBERT
import matplotlib.pyplot as plt
import hashlib
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Sistem Rekomendasi SDM BRIN", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f9fbfc;
    }
    .block-container {
        padding-top: 2rem;
    }
    .tag {
        display: inline-block;
        background-color: #e1ebf7;
        color: #333;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 13px;
    }
    .feature-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 100%;
        text-align: center;
    }
    .feature-title {
        font-size: 20px;
        font-weight: bold;
        color: #1a73e8;
        margin-bottom: 15px;
    }
    .intro-box {
        background-color: #eaf2fa;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 18px;
        text-align: justify;
    }
    h1.main-title {
        text-align: left;
        font-size: 56px;
        font-weight: bold;
        color: #1a73e8;
        margin-bottom: 0;
    }
    h2.subtitle {
        text-align: left;
        font-size: 24px;
        color: #333;
        margin-top: 0px;
    }
    h3.section-title {
        text-align: center;
        font-size: 26px;
        color: #444;
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi state admin login
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "admin_username" not in st.session_state:
    st.session_state.admin_username = ""

# === Fungsi Hash & Verifikasi Login Admin ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin_login(username, password):
    try:
        df_cred = pd.read_csv("admin_credentials.csv")
        match = df_cred[
            (df_cred["username"].str.strip() == username.strip()) &
            (df_cred["password"].str.strip() == password.strip())
        ]
        return not match.empty
    except Exception as e:
        st.error(f"Gagal membaca kredensial admin: {e}")
        return False


# ------------------ LOAD & CACHE ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_tr.csv")
    df["title"] = df["title"].fillna("")
    df["topik penelitian author"] = df["topik penelitian author"].fillna("")
    df["unit kerja"] = df["unit kerja"].fillna("")
    df["bidang_keahlian_author"] = df["bidang_keahlian_author"].fillna("[]")
    df["bidang_keahlian_author"] = df["bidang_keahlian_author"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    embeddings = np.load("embeddings2.npy")
    return df, embeddings

@st.cache_data
def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_kw_model():
    return KeyBERT(model)

# ------------------ INIT ------------------
df, embeddings = load_data()
model = load_model()
kw_model = load_kw_model()
topik_dict = load_mapping("topik_penelitian_author.txt")
bidang_dict = load_mapping("topik_bidang_keahlian.txt")
foto_path = "foto.jpg"
sdm_img_path = "sdm.jpg"

all_unit = sorted(df["unit kerja"].dropna().unique())
all_bidang = sorted(df["bidang_keahlian_author"].explode().dropna().unique())

if "active_page" not in st.session_state:
    st.session_state.active_page = "beranda"

# === SIDEBAR NAVIGASI ===
st.sidebar.markdown("## 📂 Navigasi")
if st.sidebar.button("🏠 Beranda", use_container_width=True):
    st.session_state.active_page = "beranda"
if st.sidebar.button("📚 Eksplorasi SDM", use_container_width=True):
    st.session_state.active_page = "eksplorasi"
if st.sidebar.button("🔍 Cari Rekomendasi SDM", use_container_width=True):
    st.session_state.active_page = "rekomendasi"

# === Login admin ===
if not st.session_state.admin_logged_in:
    with st.sidebar.expander("🔐 Login Admin"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_admin_login(username, password):
                st.session_state.admin_logged_in = True
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password salah.")

# === Setelah Login: Menu Tambahan untuk Admin ===
if st.session_state.admin_logged_in:
    st.sidebar.markdown("---")
    if st.sidebar.button("➕ Tambah Data SDM", use_container_width=True):
        st.session_state.active_page = "tambah"
    if st.sidebar.button("📁 Kelola Log Kontak", use_container_width=True):
        st.session_state.active_page = "log_kontak"
    if st.sidebar.button("📜 History Kontak", use_container_width=True):
        st.session_state.active_page = "history"
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.success("Anda telah logout.")
        st.rerun()


# ------------------ PREPROCESS ------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    stop_words_set = set(stopwords.words("indonesian") + stopwords.words("english"))
    if lang == "id":
        stop_words_set = set(stopwords.words("indonesian"))
    elif lang == "en":
        stop_words_set = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words_set]
    return " ".join(words)

# ------------------ PAGE: Beranda ------------------

# Dummy base64 sticker image
sticker_sdm_base64 = "https://img.icons8.com/color/96/conference-call.png"  # URL eksternal

if st.session_state.get("active_page") == "beranda":

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@500;700&display=swap');

        h1.center-title {
            text-align: center;
            font-size: 64px;
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            color: #000;
            margin-bottom: 5px;
        }

        .subtitle-tagline {
            text-align: center;
            font-size: 22px;
            font-family: 'Montserrat', sans-serif;
            color: #1a73e8;
            margin-top: 5px;
            margin-bottom: 35px;
        }

        .typewriter-left {
            font-family: 'Montserrat', sans-serif;
            font-size: 20px;
            font-weight: bold;
            color: #1a73e8;
            white-space: nowrap;
            overflow: hidden;
            border-right: 3px solid #1a73e8;
            width: 100%;
            animation: typing 4s steps(40, end) infinite, blink-caret .75s step-end infinite;
            margin-bottom: 20px;
        }

        @keyframes typing {
            0% { width: 0 }
            40% { width: 100% }
            60% { width: 100% }
            100% { width: 0 }
        }

        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: #1a73e8; }
        }

        .section-subheader {
            font-size: 22px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
            color: #000000;
            margin-bottom: 10px;
        }

        .intro-box {
            background-color: #EEF4FB;
            padding: 20px;
            border-radius: 14px;
            font-size: 17px;
            text-align: justify;
            font-family: 'Montserrat', sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.3s ease-in-out;
            position: relative;
            margin-bottom: 30px;
        }

        .intro-box:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }

        .sticker-sdm {
            position: absolute;
            top: -40px;
            right: -10px;
            width: 80px;
        }

        .fitur-box {
            background-color: #EEF4FB;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            height: 100%;
            font-family: 'Montserrat', sans-serif;
            transition: all 0.3s ease-in-out;
            text-align: center;
        }

        .fitur-box:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            cursor: pointer;
        }

        .fitur-title {
            font-size: 18px;
            font-weight: bold;
            color: #1a73e8;
            margin-bottom: 10px;
        }

        h3.section-title-center {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Montserrat', sans-serif;
            color: #000;
            margin-top: 50px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header & Tagline
    st.markdown("<h1 class='center-title'>SINERGI BRIN</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-tagline'>Sistem Integrasi Riset dan Rekomendasi SDM Implementatif</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='typewriter-left'>Temukan peneliti yang relevan dengan topik riset Anda!</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subheader'>APA ITU SINERGI BRIN?</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='intro-box'>
                <img class='sticker-sdm' src="{sticker_sdm_base64}">
                SINERGI BRIN adalah platform yang dirancang untuk memetakan dan menemukan SDM (peneliti) berdasarkan topik riset. 
                Platform ini menghubungkan kebutuhan penelitian dengan keahlian SDM yang tepat, mendorong kolaborasi antar institusi serta implementasi hasil riset.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-subheader' style='text-align:center;'>FITUR LAYANAN</div>", unsafe_allow_html=True)
        fitur_col1, fitur_col2 = st.columns(2)
        with fitur_col1:
            st.markdown("""
            <div class='fitur-box'>
                <div class='fitur-title'>🔍 Cari Rekomendasi SDM</div>
                <p>Temukan peneliti yang relevan secara otomatis berdasarkan judul dan deskripsi penelitian Anda.</p>
            </div>
            """, unsafe_allow_html=True)
        with fitur_col2:
            st.markdown("""
            <div class='fitur-box'>
                <div class='fitur-title'>📁 Eksplorasi SDM</div>
                <p>Lihat dan jelajahi daftar SDM yang tersedia berdasarkan bidang keahlian tertentu.</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        with open(sdm_img_path, "rb") as f:
            img_bytes = f.read()
            b64_img = base64.b64encode(img_bytes).decode()
        st.markdown(f"<img src='data:image/jpg;base64,{b64_img}' width='100%' style='border-radius: 16px;'>", unsafe_allow_html=True)

    # ===== VISUALISASI BIDANG KEAHLIAN =====
    st.markdown("<h3 class='section-title-center'>📊 DISTRIBUSI BIDANG KEAHLIAN</h3>", unsafe_allow_html=True)

    bidang_sdm_raw = df.groupby("author")["bidang_keahlian_author"].first()
    bidang_sdm_list = []
    for bidang_list in bidang_sdm_raw:
        bidang_sdm_list += list(set(bidang_list))
    bidang_sdm_series = pd.Series(bidang_sdm_list)
    bidang_sdm_count = bidang_sdm_series.value_counts().head(10)

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=bidang_sdm_count.values,
                y=bidang_sdm_count.index,
                orientation='h',
                marker=dict(color='#4A90E2'),
                hovertemplate='Jumlah SDM: %{x}<extra></extra>'
            )
        ])
        fig_bar.update_layout(
            title=dict(text="Frekuensi SDM per Bidang Keahlian", x=0.5, font=dict(size=18)),
            height=420,
            margin=dict(l=40, r=40, t=60, b=40),
            dragmode=False,
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_plot2:
        fig_donut = go.Figure(data=[
            go.Pie(
                labels=bidang_sdm_count.index,
                values=bidang_sdm_count.values,
                hole=0.45,
                pull=[0.05] * len(bidang_sdm_count),
                marker=dict(
                    colors=["#174ea6", "#286bd0", "#3c86f4", "#72a7f8", "#a4c4fa", "#d0e1fc"],
                    line=dict(color='black', width=1)
                ),
                hoverinfo="label+percent+value",
                textinfo='label+percent'
            )
        ])
        fig_donut.update_layout(
            title=dict(text="Proporsi SDM per Bidang Keahlian", x=0.5, font=dict(size=18)),
            height=420,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True)

# ------------------ PAGE: Rekomendasi SDM ------------------

# === INIT PAGE STATE ===
if "active_page" not in st.session_state:
    st.session_state.active_page = "rekomendasi"

# === GLOBAL STYLE ===
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            line-height: 1.5;
            text-align: justify;
        }
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            text-align: center;
        }
        .card img {
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .card h4 {
            margin: 10px 0 5px 0;
            font-weight: 700;
            color: #111827;
        }
        .card p {
            margin: 4px 0;
            color: #4b5563;
            font-size: 13px;
            text-align: left;
        }
        .tag {
            display: inline-block;
            background-color: #e1ecf4;
            color: #1a73e8;
            border-radius: 10px;
            padding: 4px 10px;
            margin: 2px 3px;
            font-size: 11px;
        }
    </style>
""", unsafe_allow_html=True)

# === PAGE: REKOMENDASI SDM ===
if st.session_state.active_page == "rekomendasi":
    st.markdown("<h2 style='text-align:center;'>🔍 SISTEM REKOMENDASI SDM BRIN</h2>", unsafe_allow_html=True)
    st.caption("Temukan SDM peneliti BRIN yang paling sesuai dengan topik riset Anda.")

    st.markdown("""
    <style>
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            text-align: center;
        }
        .card img {
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .card h4 {
            margin: 10px 0 5px 0;
            font-weight: 700;
            color: #111827;
        }
        .card p {
            margin: 4px 0;
            color: #4b5563;
            font-size: 13px;
            text-align: left;
        }
        .tag {
            display: inline-block;
            background-color: #e1ecf4;
            color: #1a73e8;
            border-radius: 10px;
            padding: 4px 10px;
            margin: 2px 3px;
            font-size: 11px;
        }
    </style>
""", unsafe_allow_html=True)

    # === FILTER ===
    with st.expander("🎯 Filter berdasarkan"):
        col1, col2 = st.columns(2)
        with col1:
            selected_unit = st.multiselect("🏢 Unit Kerja", all_unit)
        with col2:
            selected_bidang = st.multiselect("🧠 Bidang Keahlian", all_bidang)

    df_filtered = df.copy()
    if selected_unit:
        df_filtered = df_filtered[df_filtered["unit kerja"].isin(selected_unit)]
    if selected_bidang:
        df_filtered = df_filtered[df_filtered["bidang_keahlian_author"].apply(
            lambda x: any(b in x for b in selected_bidang))]

    # === INIT SESSION STATE ===
    if "judul" not in st.session_state:
        st.session_state["judul"] = ""
    if "deskripsi" not in st.session_state:
        st.session_state["deskripsi"] = ""
    if "hasil_rekomendasi" not in st.session_state:
        st.session_state["hasil_rekomendasi"] = []
    if "keyword_list" not in st.session_state:
        st.session_state["keyword_list"] = []

    # === INPUT FORM ===
    judul = st.text_input("📌 Judul Penelitian", key="judul")
    deskripsi = st.text_area("📝 Deskripsi Penelitian", key="deskripsi", height=100)

    # === TOMBOL SEJAJAR ===
    tombol1, tombol2, _ = st.columns(3)

    with tombol1:
        cari = st.button("🔍 Cari Rekomendasi")

    with tombol2:
        reset = st.button("🔄 Reset")

    # === AKSI TOMBOL CARI ===
    if cari:
        if not judul.strip() or not deskripsi.strip():
            st.warning("Judul dan deskripsi tidak boleh kosong.")
        elif df_filtered.empty:
            st.warning("Tidak ada data sesuai filter.")
        else:
            with st.spinner("Mencari SDM yang relevan..."):
                combined_input = preprocess_text(judul + ". " + deskripsi)
                keywords = kw_model.extract_keywords(
                    combined_input,
                    keyphrase_ngram_range=(1, 3),
                    use_mmr=True,
                    diversity=0.7,
                    top_n=5
                )
                keyword_list = [kw.lower() for kw, _ in keywords if not kw.isnumeric()]
                st.session_state["keyword_list"] = keyword_list

                def match_kw(text):
                    return any(kw in str(text).lower() for kw in keyword_list)

                df_filtered['match_title'] = df_filtered['title'].apply(match_kw)
                df_filtered['match_topic'] = df_filtered['topik penelitian author'].apply(match_kw)

                df_target = df_filtered[df_filtered['match_title'] | df_filtered['match_topic']]
                if df_target.empty:
                    df_target = df_filtered.copy()

                query_embedding = model.encode(combined_input)
                target_embeddings = embeddings[df_target.index]
                scores = cosine_similarity([query_embedding], target_embeddings)[0]
                df_target["score"] = scores

                top_results = df_target.sort_values("score", ascending=False)
                top_results = top_results.groupby("author", as_index=False).first().sort_values("score", ascending=False).head(30)
                st.session_state["hasil_rekomendasi"] = top_results.to_dict(orient="records")

    # === AKSI RESET ===
    if reset:
        for key in ["hasil_rekomendasi", "keyword_list", "judul", "deskripsi"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    # === TAMPILKAN HASIL ATAU RANDOM ===
    if st.session_state.get("hasil_rekomendasi"):
        st.markdown("### 🔑 Keyword Utama: " + ", ".join(f"`{kw}`" for kw in st.session_state.get("keyword_list", [])))
        st.markdown("### 👨‍🔬 Daftar Rekomendasi SDM")
        hasil = st.session_state["hasil_rekomendasi"]

        # === TOMBOL DOWNLOAD (di bawah hasil rekomendasi)
        hasil_df = pd.DataFrame(hasil)
        st.download_button(
            label="⬇️ Download CSV",
            data=hasil_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_rekomendasi_sdm.csv",
            mime="text/csv"
        )
    else:
        hasil = df_filtered.sample(n=min(6, len(df_filtered))).to_dict(orient="records")

    # === TAMPILKAN KARTU SDM ===
    for row_group in range(0, len(hasil), 3):
        cols = st.columns(3)
        for i, row in enumerate(hasil[row_group:row_group+3]):
            with cols[i]:
                bidang_tags = "".join([f"<span class='tag'>{b}</span>" for b in row["bidang_keahlian_author"]])
                author = row["author"]
                topik = topik_dict.get(author, "-")
                bidang = bidang_dict.get(topik, row.get("bidang_keahlian_author", "-"))
                foto_path = "foto.jpg"

                with st.container():
                    st.markdown(f"""
                        <div class='card'>
                            <img src='data:image/jpg;base64,{base64.b64encode(open(foto_path, "rb").read()).decode()}' width='90'>
                            <h4>{author}</h4>
                            <div>{bidang_tags}</div>
                    """, unsafe_allow_html=True)

                    with st.expander("📄 Lihat Detail SDM"):
                        score_value = row.get("score", "-")
                        score_display = f"{score_value:.4f}" if isinstance(score_value, (float, int)) else score_value
                        st.markdown(f"""
                            <div class='detail-wrapper'>
                                <p><b>Judul:</b> {row["title"]}</p>
                                <p><b>Topik:</b> {row["topik penelitian author"]}</p>
                                <p><b>Unit:</b> {row["unit kerja"]}</p>
                                <p><b>Score:</b> {score_display}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Simpan status konfirmasi di session state
                    confirm_key = f"confirm_chat_{row_group+i}"
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if not st.session_state[confirm_key]:
                        if st.button(f"💬 Chat Admin", key=f"chat_{row_group+i}"):
                            st.session_state[confirm_key] = True
                    else:
                        st.warning(f"Apakah Anda ingin menghubungi admin terkait **{author}**?")
                        konfirmasi, batal = st.columns([1, 1])
                        with konfirmasi:
                            if st.button("✅ Ya, Hubungi Admin", key=f"confirm_yes_{row_group+i}"):
                                st.session_state.active_page = "chat"
                                st.session_state.selected_author = row["author"]
                                st.session_state.selected_unit = row["unit kerja"]
                                st.session_state[confirm_key] = False  # reset
                        with batal:
                            if st.button("❌ Batal", key=f"confirm_cancel_{row_group+i}"):
                                st.session_state[confirm_key] = False

# === PAGE: CHAT ADMIN ===
elif st.session_state.active_page == "chat":
    import os
    import datetime

    st.markdown("## 💬 Hubungi Admin Terkait SDM")

    author = st.session_state.get("selected_author", "SDM Terpilih")
    unit_sdm = st.session_state.get("selected_unit", "-")
    st.info(f"Anda sedang menghubungi **{author}** dari unit **{unit_sdm}**")

    with st.form("chat_form"):
        nama = st.text_input("👤 Nama Lengkap Anda")
        instansi = st.text_input("🏢 Asal Instansi")
        email = st.text_input("📧 Email")
        telepon = st.text_input("📱 Nomor Telepon")
        topik = st.text_input("🧪 Topik Riset")
        bidang_dibutuhkan = st.text_input("🧠 Bidang Keahlian yang Dibutuhkan")
        keperluan = st.text_area("📌 Keperluan/Kolaborasi")

        submitted = st.form_submit_button("📤 Kirim")

        if submitted:
            if not all([nama.strip(), email.strip(), topik.strip(), keperluan.strip(), instansi.strip()]):  # ✅ validasi
                st.warning("Mohon lengkapi semua kolom wajib.")
            else:
                kontak_data = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Nama": nama,
                    "Asal Instansi": instansi,
                    "Email": email,
                    "Telepon": telepon,
                    "Topik Riset": topik,
                    "Bidang Keahlian yang Dibutuhkan": bidang_dibutuhkan,
                    "Keperluan": keperluan,
                    "Nama SDM": author,
                    "Unit Kerja SDM": unit_sdm
                }

                kontak_df = pd.DataFrame([kontak_data])
                try:
                    kontak_df.to_csv("log_kontak_sdm.csv", mode="a", index=False, header=not os.path.exists("log_kontak_sdm.csv"))
                except Exception as e:
                    st.error(f"Gagal menyimpan data: {e}")
                else:
                    st.success("Pesan Anda berhasil dikirim. Tim kami akan segera menghubungi Anda.")
                    st.markdown(f"""
                        ---
                        ### 💬 Chat Otomatis:
                        👋 Halo {nama} dari instansi **{instansi}**, terima kasih telah menghubungi kami terkait kolaborasi dengan **{author}** dari unit **{unit_sdm}**.<br><br>
                        Kami sudah mencatat topik Anda: _{topik}_ dan keperluan: _{keperluan}_<br>
                        Bidang keahlian yang dibutuhkan: _{bidang_dibutuhkan}_<br><br>
                        Tim kami akan segera menindaklanjuti dan menghubungi anda kembali. 🤝
                    """, unsafe_allow_html=True)

    st.button("🔙 Kembali ke Rekomendasi", on_click=lambda: st.session_state.update({"active_page": "rekomendasi"}))

# ------------------ PAGE: Eksplorasi SDM ------------------
elif st.session_state.active_page == "eksplorasi":
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@500;700&display=swap');

        h2.page-title {
            text-align: center;
            font-size: 48px;
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            color: #1a73e8;
            margin-bottom: 5px;
        }

        .tag {
            display: inline-block;
            background-color: #e1ecf4;
            color: #0366d6;
            padding: 4px 10px;
            margin: 3px 4px 3px 0;
            border-radius: 16px;
            font-size: 13px;
            font-family: 'Montserrat', sans-serif;
        }

        .card {
            background-color: #F9FAFB;
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            font-family: 'Montserrat', sans-serif;
            transition: all 0.3s ease-in-out;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        }

        .author-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .author-header h4 {
            margin: 0;
            font-size: 22px;
            font-weight: bold;
            color: #000;
        }

        .author-photo {
            width: 180px;
            height: 180px;
            border-radius: 12px;
            object-fit: cover;
            border: 3px solid #1a73e8;
        }

        .judul-scroll {
            max-height: 300px;
            overflow-y: auto;
            padding-right: 10px;
        }

        .judul-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 10px;
            padding-top: 10px;
        }

        .judul-item {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            font-size: 14px;
            font-family: 'Montserrat', sans-serif;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }

        .judul-item:hover {
            background-color: #eaf2fb;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='page-title'>📁 Eksplorasi SDM BRIN</h2>", unsafe_allow_html=True)
    st.caption("Telusuri daftar SDM BRIN berdasarkan unit kerja atau bidang keahlian.")

    col1, col2 = st.columns(2)
    with col1:
        filter_unit = st.selectbox("🏢 Pilih Unit Kerja", [""] + all_unit)
    with col2:
        filter_bidang = st.selectbox("🧠 Pilih Bidang Keahlian", [""] + all_bidang)

    df_explore = df.copy()
    if filter_unit:
        df_explore = df_explore[df_explore["unit kerja"] == filter_unit]
    if filter_bidang:
        df_explore = df_explore[df_explore["bidang_keahlian_author"].apply(lambda x: filter_bidang in x)]

    # Load base64 image for profile (sample placeholder image encoded)
    with open("foto.jpg", "rb") as image_file:
        base64_foto = base64.b64encode(image_file.read()).decode()

    if df_explore.empty:
        st.warning("Tidak ditemukan SDM dengan filter yang dipilih.")
    else:
        grouped = list(df_explore.groupby("author"))[:10]  # Batasi 10 SDM pertama
        for author, group in grouped:
            bidang_tags = "".join(f"<span class='tag'>{b}</span>" for b in group["bidang_keahlian_author"].explode().unique())
            topik_ = group["topik penelitian author"].iloc[0] if "topik penelitian author" in group else "-"
            unit = group["unit kerja"].iloc[0]
            judul_list = group[["title", "authors", "description", "publication_date"]].dropna(subset=["title"])

            with st.container():
                st.markdown(f"""
                    <div class='card'>
                        <div class='author-header'>
                            <h4>{author}</h4>
                            <img src='data:image/png;base64,{base64_foto}' class='author-photo'>
                        </div>
                        <p><b>Unit Kerja:</b> {unit}</p>
                        <p><b>Topik Penelitian:</b> {topik_}</p>
                        <div><b>Bidang Keahlian:</b> {bidang_tags}</div>
                    </div>
                """, unsafe_allow_html=True)

                with st.expander("📜 Rekam Jejak Judul Penelitian"):
                    st.markdown("<div class='judul-scroll'>", unsafe_allow_html=True)

                    for idx, row in judul_list.iterrows():
                        title = row["title"]
                        authors = row["authors"] if pd.notna(row["authors"]) else "-"
                        description = row["description"] if pd.notna(row["description"]) else "-"
                        pub_date = row["publication_date"] if pd.notna(row["publication_date"]) else "-"

                        # Ganti expander dengan toggle untuk menghindari nested expander
                        if st.toggle(f"📄 {title}", key=f"toggle_{author}_{idx}"):
                            st.markdown(f"**Penulis:** {authors}")
                            st.markdown(f"**Deskripsi:** {description}")
                            st.markdown(f"**Tanggal Publikasi:** {pub_date}")

                    st.markdown("</div>", unsafe_allow_html=True)

 
# ------------------ PAGE: Tambah Data ------------------
elif st.session_state.active_page == "tambah":
    st.title("➕ Tambah Data SDM Baru")
    st.markdown("Isi form berikut untuk menambahkan data SDM ke dalam sistem rekomendasi.")

    with st.form("form_tambah"):
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Nama Peneliti")
            title = st.text_input("Judul Penelitian")
            authors = st.text_input("Daftar Penulis")
            pub_date = st.text_input("Tahun Publikasi")
        with col2:
            journal = st.text_input("Jurnal")
            unit = st.selectbox("Unit Kerja", all_unit)
            unit_manual = st.text_input("Atau Tambahkan Unit Baru")
            topik = st.text_input("Topik Penelitian")

        description = st.text_area("Deskripsi / Abstrak")
        bidang_pilihan = st.multiselect("Bidang Keahlian", all_bidang)
        bidang_manual = st.text_input("Tambahan Bidang Keahlian (pisahkan koma)")

        simpan = st.form_submit_button("✅ Simpan")

        if simpan:
            new_fields = bidang_pilihan + [b.strip() for b in bidang_manual.split(",") if b.strip()]
            full_text = title + " " + description
            cleaned_text = preprocess_text(full_text)
            new_embedding = model.encode(cleaned_text)

            new_row = pd.DataFrame([{
                "author": author,
                "title": title,
                "authors": authors,
                "publication_date": pub_date,
                "journal": journal,
                "description": description,
                "unit kerja": unit_manual if unit_manual else unit,
                "topik penelitian author": topik,
                "full_text": full_text,
                "bidang_keahlian_author": new_fields
            }])

            new_row.to_csv("data_tr.csv", mode='a', index=False, header=False)

            try:
                embeddings_path = "embeddings2.npy"
                if os.path.exists(embeddings_path):
                    existing_embeddings = np.load(embeddings_path)
                    updated_embeddings = np.vstack([existing_embeddings, new_embedding])
                else:
                    updated_embeddings = np.array([new_embedding])
                np.save(embeddings_path, updated_embeddings)
            except Exception as e:
                st.error(f"Gagal menyimpan embedding baru: {e}")

            st.success("✅ Data berhasil ditambahkan!")
            st.experimental_rerun()

# # ------------------ PAGE: Kelola Log Kontak (Admin Only) ------------------
elif st.session_state.active_page == "log_kontak":
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Montserrat:wght@500;700&display=swap');

        h2.page-title {
            text-align: center;
            font-size: 48px;
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            color: #000;
            margin-bottom: 30px;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 16px;
        }

        .contact-card {
            background-color: #fdfdfd;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            font-family: 'Montserrat', sans-serif;
            transition: 0.3s ease-in-out;
        }

        .contact-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }

        .contact-card b {
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='page-title'>🗂 Kelola Log Kontak SDM</h2>", unsafe_allow_html=True)
    st.caption("Data hasil chat/kontak dari pengguna terkait SDM BRIN yang ingin diajak kolaborasi.")

    try:
        log_df = pd.read_csv("log_kontak_sdm.csv", dtype={"Telepon": str, "Nomor Telepon": str})

        total_kontak = len(log_df)
        confirmed = 0
        rejected = 0
        if os.path.exists("history_kontak.csv"):
            history_df = pd.read_csv("history_kontak.csv")
            confirmed = (history_df["Status"] == "Dikonfirmasi").sum()
            rejected = (history_df["Status"] == "Ditolak").sum()

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("📊 Total Kontak", total_kontak)
        colm2.metric("✅ Dikonfirmasi", confirmed)
        colm3.metric("❌ Ditolak", rejected)

        # Search bar
        search_query = st.text_input("🔍 Cari berdasarkan Nama, Email, atau Instansi:").lower()
        if search_query:
            log_df = log_df[log_df.apply(lambda row: search_query in str(row['Nama']).lower() or search_query in str(row['Email']).lower() or search_query in str(row['Asal Instansi']).lower(), axis=1)]

        # Filter by Unit Kerja
        unit_options = ["Semua"] + sorted(log_df["Unit Kerja SDM"].dropna().unique().tolist())
        selected_unit = st.selectbox("🏢 Filter berdasarkan Unit Kerja", unit_options)
        if selected_unit != "Semua":
            log_df = log_df[log_df["Unit Kerja SDM"] == selected_unit]

        if log_df.empty:
            st.info("Tidak ada data yang sesuai filter.")
        else:
            st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
            for index, row in log_df.iterrows():
                st.markdown(f"""
                    <div class='contact-card'>
                        <b>Nama:</b> {row['Nama']}<br>
                        <b>Asal Instansi:</b> {row.get('Asal Instansi', '-') if pd.notna(row.get('Asal Instansi')) else '-'}<br>
                        <b>Email:</b> {row['Email']}<br>
                        <b>Telepon:</b> {row['Telepon']}<br>
                        <b>Topik Riset:</b> {row['Topik Riset']}<br>
                        <b>Bidang Keahlian:</b> {row['Bidang Keahlian yang Dibutuhkan']}<br>
                        <b>Keperluan:</b> {row['Keperluan']}<br>
                        <b>SDM:</b> {row['Nama SDM']} ({row['Unit Kerja SDM']})<br><br>
                    """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"✅ Konfirmasi - {index}"):
                        row_data = row.to_dict()
                        row_data['Status'] = 'Dikonfirmasi'
                        history_file = "history_kontak.csv"
                        if not os.path.exists(history_file):
                            pd.DataFrame(columns=list(row_data.keys())).to_csv(history_file, index=False)
                        with open(history_file, "a", newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=row_data.keys())
                            writer.writerow(row_data)
                        log_df.drop(index, inplace=True)
                        log_df.to_csv("log_kontak_sdm.csv", index=False)
                        st.experimental_rerun()
                with col2:
                    if st.button(f"❌ Tolak - {index}"):
                        row_data = row.to_dict()
                        row_data['Status'] = 'Ditolak'
                        history_file = "history_kontak.csv"
                        if not os.path.exists(history_file):
                            pd.DataFrame(columns=list(row_data.keys())).to_csv(history_file, index=False)
                        with open(history_file, "a", newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=row_data.keys())
                            writer.writerow(row_data)
                        log_df.drop(index, inplace=True)
                        log_df.to_csv("log_kontak_sdm.csv", index=False)
                        st.experimental_rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                label="⬇️ Download Log Kontak",
                data=log_df.to_csv(index=False).encode("utf-8"),
                file_name="log_kontak_sdm.csv",
                mime="text/csv"
            )

    except FileNotFoundError:
        st.warning("File log_kontak_sdm.csv belum tersedia.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca log kontak: {e}")

# ------------------ PAGE: History Kontak ------------------
elif st.session_state.active_page == "history":
    st.markdown("""
        <style>
        .center-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            font-family: 'Playfair Display', serif;
            color: #000;
            margin-bottom: 20px;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 16px;
        }
        .history-card {
            background-color: #ffffff;
            border-left: 6px solid #1a73e8;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
            font-family: 'Montserrat', sans-serif;
        }
        .status-konfirmasi {
            color: green;
            font-weight: bold;
        }
        .status-tolak {
            color: red;
            font-weight: bold;
        }
        .search-bar {
            padding: 8px 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
            width: 100%;
            margin-bottom: 20px;
            font-family: 'Montserrat', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='center-title'>📜 Riwayat Kontak SDM</h2>", unsafe_allow_html=True)
    st.caption("Menampilkan seluruh riwayat kontak yang sudah dikonfirmasi atau ditolak.")

    try:
        history_df = pd.read_csv("history_kontak.csv", dtype={"Telepon": str, "Nomor Telepon": str})
        if history_df.empty:
            st.info("Belum ada riwayat kontak.")
        else:
            # Tombol download dan statistik
            col_download, col_stat1, col_stat2 = st.columns([2, 1, 1])
            with col_download:
                st.download_button(
                    label="⬇️ Download History Kontak",
                    data=history_df.to_csv(index=False).encode("utf-8"),
                    file_name="history_kontak.csv",
                    mime="text/csv"
                )
            with col_stat1:
                st.metric("Dikonfirmasi", history_df[history_df["Status"] == "Dikonfirmasi"].shape[0])
            with col_stat2:
                st.metric("Ditolak", history_df[history_df["Status"] == "Ditolak"].shape[0])

            # === Search Filter ===
            search_query = st.text_input("🔍 Cari Nama / Email / Topik Riset", "").strip().lower()

            if search_query:
                history_df = history_df[
                    history_df["Nama"].str.lower().str.contains(search_query, na=False) |
                    history_df["Email"].str.lower().str.contains(search_query, na=False) |
                    history_df["Topik Riset"].str.lower().str.contains(search_query, na=False)
                ]

            if history_df.empty:
                st.warning("Tidak ditemukan kontak dengan kata kunci pencarian tersebut.")
            else:
                st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
                for _, row in history_df.iterrows():
                    status_html = (
                        f"<span class='status-konfirmasi'>✅ {row['Status']}</span>"
                        if row['Status'] == 'Dikonfirmasi'
                        else f"<span class='status-tolak'>❌ {row['Status']}</span>"
                    )
                    st.markdown(f"""
                        <div class='history-card'>
                            <b>Nama:</b> {row['Nama']}<br>
                            <b>Asal Instansi:</b> {row.get('Asal Instansi', '-')}<br>
                            <b>Email:</b> {row['Email']}<br>
                            <b>Telepon:</b> {row['Telepon']}<br>
                            <b>Topik Riset:</b> {row['Topik Riset']}<br>
                            <b>Bidang Keahlian:</b> {row['Bidang Keahlian yang Dibutuhkan']}<br>
                            <b>Keperluan:</b> {row['Keperluan']}<br>
                            <b>SDM:</b> {row['Nama SDM']} ({row['Unit Kerja SDM']})<br><br>
                            <b>Status:</b> {status_html}
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("File history_kontak.csv belum tersedia.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file history: {e}")
