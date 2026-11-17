
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CẤU HÌNH APP
# ==========================================

st.set_page_config(
    page_title="Dự đoán năng suất lúa SVR",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 DỰ ĐOÁN NĂNG SUẤT LÚA THEO MÙA VỤ")
st.subheader("Ứng dụng trí tuệ nhân tạo - Mô hình SVR")

st.markdown("""
Nhập các thông số mùa vụ, điều kiện thời tiết và canh tác
để dự đoán năng suất lúa bằng thuật toán Support Vector Regression.
""")

# ==========================================
# 2. TẢI DỮ LIỆU HUẤN LUYỆN
# ==========================================

st.sidebar.header("📂 DỮ LIỆU HUẤN LUYỆN")

file = st.sidebar.file_uploader(
    "Tải dữ liệu CSV hoặc Excel",
    type=["csv", "xlsx"]
)

st.sidebar.caption(
    "Dữ liệu cần có các cột đầu vào và cột Nang_suat."
)

# ==========================================
# 3. NHẬP THÔNG TIN MÙA VỤ
# ==========================================

st.header("1. THÔNG TIN MÙA VỤ")

col1, col2, col3 = st.columns(3)

with col1:
    mua_vu = st.selectbox(
        "🌱 Mùa vụ",
        ["Đông Xuân", "Hè Thu", "Thu Đông"]
    )

with col2:
    nam = st.number_input(
        "📅 Năm sản xuất",
        min_value=2000,
        max_value=2100,
        value=2026
    )

with col3:
    giong_lua = st.selectbox(
        "🌾 Giống lúa",
        ["OM5451", "Đài Thơm 8", "IR50404",
         "ST25", "Jasmine 85"]
    )

# ==========================================
# 4. NHẬP ĐIỀU KIỆN THỜI TIẾT
# ==========================================

st.header("2. ĐIỀU KIỆN THỜI TIẾT")

col1, col2, col3, col4 = st.columns(4)

with col1:
    luong_mua = st.number_input(
        "🌧️ Lượng mưa (mm/vụ)",
        min_value=0.0,
        value=1200.0
    )

with col2:
    nhiet_do = st.number_input(
        "🌡️ Nhiệt độ trung bình (°C)",
        min_value=0.0,
        value=28.0
    )

with col3:
    do_am = st.number_input(
        "💧 Độ ẩm (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

with col4:
    gio_nang = st.number_input(
        "☀️ Giờ nắng TB/ngày",
        min_value=0.0,
        max_value=24.0,
        value=6.0
    )

# ==========================================
# 5. NHẬP ĐIỀU KIỆN CANH TÁC
# ==========================================

st.header("3. ĐIỀU KIỆN CANH TÁC")

col1, col2, col3, col4 = st.columns(4)

with col1:
    dien_tich = st.number_input(
        "📐 Diện tích (ha)",
        min_value=0.01,
        value=1.0
    )

with col2:
    phan_bon = st.number_input(
        "🌿 Lượng phân bón (kg/ha)",
        min_value=0.0,
        value=250.0
    )

with col3:
    nuoc_tuoi = st.number_input(
        "🚿 Lượng nước tưới (mm/vụ)",
        min_value=0.0,
        value=300.0
    )

with col4:
    thoi_gian = st.number_input(
        "⏳ Thời gian sinh trưởng (ngày)",
        min_value=1,
        value=110
    )

# ==========================================
# 6. LỰA CHỌN THUẬT TOÁN SVR
# ==========================================

st.header("4. CẤU HÌNH MÔ HÌNH SVR")

col1, col2, col3 = st.columns(3)

with col1:
    kernel = st.selectbox(
        "Chọn Kernel",
        ["rbf", "linear", "poly"]
    )

with col2:
    C = st.number_input(
        "Tham số C",
        min_value=0.1,
        value=10.0
    )

with col3:
    epsilon = st.number_input(
        "Tham số Epsilon",
        min_value=0.01,
        value=0.1
    )

# ==========================================
# 7. HÀM XỬ LÝ DỮ LIỆU
# ==========================================

features = [
    "Mua_vu",
    "Nam",
    "Giong_lua",
    "Luong_mua",
    "Nhiet_do",
    "Do_am",
    "Gio_nang",
    "Dien_tich",
    "Phan_bon",
    "Nuoc_tuoi",
    "Thoi_gian"
]

input_data = pd.DataFrame([{
    "Mua_vu": mua_vu,
    "Nam": nam,
    "Giong_lua": giong_lua,
    "Luong_mua": luong_mua,
    "Nhiet_do": nhiet_do,
    "Do_am": do_am,
    "Gio_nang": gio_nang,
    "Dien_tich": dien_tich,
    "Phan_bon": phan_bon,
    "Nuoc_tuoi": nuoc_tuoi,
    "Thoi_gian": thoi_gian
}])

# ==========================================
# 8. HUẤN LUYỆN MÔ HÌNH
# ==========================================

st.header("5. HUẤN LUYỆN MÔ HÌNH")

if file is not None:

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        df.columns = df.columns.astype(str).str.strip()

        st.success("Đã tải dữ liệu thành công!")

        st.write("Dữ liệu huấn luyện:")
        st.dataframe(df.head(10), use_container_width=True)

        required = features + ["Nang_suat"]

        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(
                "Thiếu các cột bắt buộc: " + ", ".join(missing)
            )
            st.stop()

        data = df[required].copy()

        # Xử lý biến số
        numeric_cols = [
            "Nam", "Luong_mua", "Nhiet_do", "Do_am",
            "Gio_nang", "Dien_tich", "Phan_bon",
            "Nuoc_tuoi", "Thoi_gian", "Nang_suat"
        ]

        for col in numeric_cols:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["Nang_suat"])

        if len(data) < 10:
            st.error(
                "Cần ít nhất 10 dòng dữ liệu hợp lệ để huấn luyện."
            )
            st.stop()

        X = data[features]
        y = data["Nang_suat"]

        numeric_features = [
            "Nam", "Luong_mua", "Nhiet_do", "Do_am",
            "Gio_nang", "Dien_tich", "Phan_bon",
            "Nuoc_tuoi", "Thoi_gian"
        ]

        categorical_features = [
            "Mua_vu", "Giong_lua"
        ]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # Chia dữ liệu 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("svr", SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon
            ))
        ])

        if st.button(
            "🚀 HUẤN LUYỆN MÔ HÌNH SVR",
            type="primary"
        ):

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.session_state["model"] = model
            st.session_state["mae"] = mean_absolute_error(
                y_test, y_pred
            )
            st.session_state["rmse"] = np.sqrt(
                mean_squared_error(y_test, y_pred)
            )
            st.session_state["r2"] = r2_score(
                y_test, y_pred
            )
            st.session_state["y_test"] = y_test.to_numpy()
            st.session_state["y_pred"] = y_pred

            st.success("Huấn luyện SVR thành công!")

    except Exception as e:
        st.error(f"Lỗi dữ liệu: {e}")

else:
    st.warning(
        "Hãy tải dữ liệu Excel/CSV để huấn luyện mô hình trước."
    )

# ==========================================
# 9. ĐÁNH GIÁ MÔ HÌNH
# ==========================================

if "model" in st.session_state:

    st.header("6. ĐÁNH GIÁ MÔ HÌNH")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{st.session_state['mae']:.4f}"
    )

    col2.metric(
        "RMSE",
        f"{st.session_state['rmse']:.4f}"
    )

    col3.metric(
        "R² SCORE",
        f"{st.session_state['r2']:.4f}"
    )

    # Biểu đồ
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        st.session_state["y_test"],
        st.session_state["y_pred"],
        alpha=0.7
    )

    min_v = min(
        st.session_state["y_test"].min(),
        st.session_state["y_pred"].min()
    )

    max_v = max(
        st.session_state["y_test"].max(),
        st.session_state["y_pred"].max()
    )

    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        "r--"
    )

    ax.set_title("So sánh năng suất thực tế và dự đoán")
    ax.set_xlabel("Năng suất thực tế (tấn/ha)")
    ax.set_ylabel("Năng suất dự đoán (tấn/ha)")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # ======================================
    # 10. DỰ ĐOÁN NĂNG SUẤT
    # ======================================

    st.header("7. KẾT QUẢ DỰ ĐOÁN NĂNG SUẤT LÚA")

    if st.button(
        "🌾 DỰ ĐOÁN NĂNG SUẤT",
        type="primary"
    ):

        prediction = st.session_state[
            "model"
        ].predict(input_data)[0]

        st.success("DỰ ĐOÁN THÀNH CÔNG!")

        col1, col2 = st.columns(2)

        col1.metric(
            "NĂNG SUẤT DỰ ĐOÁN",
            f"{prediction:.2f} tấn/ha"
        )

        san_luong = prediction * dien_tich

        col2.metric(
            "SẢN LƯỢNG DỰ KIẾN",
            f"{san_luong:.2f} tấn"
        )

        st.write("### Thông tin mùa vụ dự đoán")

        st.dataframe(input_data, use_container_width=True)

        st.info(
            "Kết quả dự đoán phụ thuộc vào chất lượng dữ liệu "
            "huấn luyện và điều kiện thực tế."
        )

else:
    st.info(
        "Sau khi huấn luyện mô hình thành công, "
        "bạn có thể sử dụng nút dự đoán."
    )

# ==========================================
# 11. CHÂN TRANG
# ==========================================

st.markdown("---")
st.caption(
    "ỨNG DỤNG DỰ ĐOÁN NĂNG SUẤT LÚA | "
    "SUPPORT VECTOR REGRESSION (SVR)"
)
