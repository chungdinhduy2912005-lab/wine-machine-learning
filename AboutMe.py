import streamlit as st
import streamlit_shadcn_ui as ui

def AboutMe():
    st.markdown(
        """
        <style>
          .fade-in-up { animation: fadeInUp .55s ease-out both; }
          @keyframes fadeInUp {
            from { opacity: 0; transform: translate3d(0, 10px, 0); }
            to   { opacity: 1; transform: translate3d(0, 0, 0); }
          }

          .card {
            border: 1px solid rgba(0,0,0,.10);
            border-radius: 16px;
            padding: 18px 18px;
            background: rgba(255,255,255,.70);
          }
          @media (prefers-color-scheme: dark) {
            .card { background: rgba(20,20,20,.35); border-color: rgba(255,255,255,.10); }
          }

          .muted { opacity: .75; }
          .pill {
            display:inline-block;
            padding:.28rem .65rem;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,.12);
            margin: .18rem .22rem 0 0;
            font-size: .86rem;
            white-space: nowrap;
          }
          @media (prefers-color-scheme: dark) {
            .pill { border-color: rgba(255,255,255,.14); }
          }

          .section-title { font-weight: 650; font-size: 1.1rem; margin: 0 0 .35rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- HERO ---
    st.markdown(
        """
        <div class="card fade-in-up">
          <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
            <div style="font-size:1.35rem;">✨</div>
            <div style="font-size:1.75rem;font-weight:700;">Chung Đình Duy</div>
            <div class="muted">• MSSV: 52300192</div>
          </div>
          <div class="muted" style="margin-top:.4rem; font-size:1.02rem; line-height:1.5;">
            Dự án Giữa kỳ môn <b>Nhập môn Học máy</b>. Thực hiện phân loại rượu vang (Wine Classification) 
            và lựa chọn đặc trưng (Feature Selection) sử dụng tập dữ liệu UCI Wine Dataset.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --- METRICS ---
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.metric("Bài toán", "Phân loại / Hồi quy")
    with c2:
        st.metric("Ngôn ngữ", "Python 3.13")
    with c3:
        st.metric("Framework", "Streamlit")
    with c4:
        st.metric("Dataset", "UCI Wine")

    st.write("")

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">👋 Về dự án</div>
              <div class="muted" style="line-height:1.55;">
                Dự án này có trực quan hoá ma trận tương quan, so sánh MAE của Feature Selection và so sánh hiệu suất phân loại (LDA, Decision Tree, Random Forest).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Skills (pills)
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">🧠 Thuật toán & Kỹ thuật</div>
              <div>
                <span class="pill">Fisher LDA</span>
                <span class="pill">Logistic Regression</span>
                <span class="pill">Decision Tree</span>
                <span class="pill">Random Forest</span>
                <span class="pill">Correlation Analysis</span>
                <span class="pill">Standardization</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
