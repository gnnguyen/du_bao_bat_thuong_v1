import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime

# Import các hàm xử lý logic từ file bên ngoài
from du_bao_gia import predict_price_value, PRICE_MODEL_PATH
from du_bao_bat_thuong import detect_anomaly, save_abnormal_to_csv, OUTPUT_RESULT_FILE, save_normal_to_csv, \
    OUTPUT_NORMAL_FILE


# =============================================================================
# 1. CẤU HÌNH & LOAD TÀI NGUYÊN
# =============================================================================

# Load model AI (sử dụng cache để không phải load lại mỗi lần f5)
@st.cache_resource
def load_price_resources():
    try:
        with open(PRICE_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None


# =============================================================================
# 2. CÁC HÀM HỖ TRỢ XỬ LÝ DỮ LIỆU (HELPER FUNCTIONS)
# =============================================================================

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV.
    Nếu file không tồn tại hoặc lỗi, trả về DataFrame rỗng.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # Reset index để đảm bảo checkbox chọn dòng hoạt động đúng
        return df.reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi đọc file {file_path}: {e}")
        return pd.DataFrame()


def save_data(df, file_path):
    """
    Lưu DataFrame xuống file CSV (Ghi đè).
    """
    try:
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"Lỗi lưu file {file_path}: {e}")
        return False


def move_to_normal(df_abnormal, indices_to_move):
    """
    Chức năng DUYỆT TIN:
    Chuyển các dòng từ danh sách Bất thường (df_abnormal) -> sang danh sách Đã đăng (df_normal).
    """
    if not indices_to_move:
        return "Không có dòng nào được chọn.", 0

    # Lấy ra các dòng cần di chuyển
    rows_to_move = df_abnormal.loc[indices_to_move].copy()

    # Cập nhật trạng thái thành 'Bình thường' (đã duyệt)
    rows_to_move['Co_Bat_Thuong'] = 0
    rows_to_move['Ly_Do_Chi_Tiet'] = 'Đã được Admin duyệt'
    rows_to_move['Tiêu đề'] = rows_to_move['Tiêu đề'].str.replace('Cảnh báo GUI', 'Bài đăng đã duyệt', regex=False)

    # Dọn dẹp cột 'Chọn' (checkbox) trước khi gộp
    if 'Chọn' in rows_to_move.columns:
        rows_to_move = rows_to_move.drop(columns=['Chọn'])

    # Đọc danh sách bài đã đăng hiện tại
    df_normal = load_data(OUTPUT_NORMAL_FILE)
    if 'Chọn' in df_normal.columns:
        df_normal = df_normal.drop(columns=['Chọn'])

    # Gộp bài mới duyệt vào danh sách bài đã đăng
    df_normal = pd.concat([rows_to_move, df_normal], ignore_index=True)

    # Xóa bài đã duyệt khỏi danh sách bất thường
    df_abnormal_updated = df_abnormal.drop(indices_to_move)
    if 'Chọn' in df_abnormal_updated.columns:
        df_abnormal_updated = df_abnormal_updated.drop(columns=['Chọn'])

    # Lưu lại cả 2 file
    save_success_abnormal = save_data(df_abnormal_updated, OUTPUT_RESULT_FILE)
    save_success_normal = save_data(df_normal, OUTPUT_NORMAL_FILE)

    if save_success_abnormal and save_success_normal:
        return "Duyệt thành công!", len(rows_to_move)
    else:
        return "Lỗi khi lưu dữ liệu.", 0


def delete_rows(df, file_path, indices_to_delete):
    """
    Chức năng XÓA TIN:
    Xóa các dòng được chọn khỏi DataFrame và lưu lại file.
    """
    if not indices_to_delete:
        return "Không có dòng nào được chọn.", 0

    rows_deleted_count = len(indices_to_delete)

    # Xóa dòng theo index
    df_updated = df.drop(indices_to_delete)

    # Dọn dẹp cột 'Chọn'
    if 'Chọn' in df_updated.columns:
        df_updated = df_updated.drop(columns=['Chọn'])

    # Lưu file
    if save_data(df_updated, file_path):
        return "Xóa thành công!", rows_deleted_count
    else:
        return "Lỗi khi lưu dữ liệu.", 0


# =============================================================================
# 3. GIAO DIỆN CHÍNH (MAIN APP)
# =============================================================================

price_res = load_price_resources()

# Menu điều hướng bên trái
menu = ["Home", "Chợ xe máy cũ và Mục tiêu của dự án", "Đánh giá và lựa chọn mô hình thích hợp", "Dự đoán giá xe cũ",
        "Phát hiện bất thường", "Các Bài Đã Đăng", "Quản lý tin bất thường","Phân chia công việc trong nhóm nghiên cứu"]
choice = st.sidebar.selectbox('Menu', menu)

# -----------------------------------------------------------------------------
# TAB 1: TRANG CHỦ (HOME)
# -----------------------------------------------------------------------------
if choice == 'Home':
    st.markdown("<h1 style='text-align: center;color: black; font-size: 3em, '>ĐỒ ÁN TỐT NGHIỆP</h1>",
                unsafe_allow_html=True)
    st.image("mua_xe_may_cu.jpg", caption="MUỐN MUA LÀ CÓ NGAY!!!")
    st.markdown(
        "<h2 style='text-align: center;color:blue; font-size: 2em, '>NGUYỄN NGỌC GIAO - NGUYỄN THỊ TUYỂN</h2>",
        unsafe_allow_html=True)
    st.markdown("""
            ## 📘 Dự án: Hệ thống dự đoán giá xe cũ và phát hiện bất thường về giá.
            Ứng dụng giúp dự đoán giá xe và phát hiện bất thường giá xe dựa trên nội dung thông số kỹ thuật và thông tin xe
            """)
    col1, col2 = st.columns(2)
   
    with col2:
        st.markdown("#### 🤖 Cảnh báo bất thường")
        st.write("Cảnh báo bất thường dựa trên phân tích sai số giữa giá đề nghị và giá dự đoán.")
    
    with col1:
        st.markdown("#### 📊 Dự đoán giá trị xe")
        st.write("Ước lượng giá xe dựa vào mô hình học máy.")


# -----------------------------------------------------------------------------
# TAB 2: GIỚI THIỆU DỰ ÁN
# -----------------------------------------------------------------------------
elif choice=="Chợ xe máy cũ và Mục tiêu của dự án":
    st.subheader("Tóm tắt thông tin về Chợ xe máy cũ")
    st.markdown("""
            - Chợ Tốt là thị trường mua bán trực tuyến hàng đầu tại Việt Nam cung cấp đa dạng các hạng mục như mua bán nhà cửa, ô tô, xe máy phục vụ gia đình. 
            - Tuy nhiên việc quảng cáo các loại sản phẩm cũ không đúng với giá trị thực (quá cao hoặc quá thấp) do nhiều nguyên nhân sẽ ảnh hưởng đến thị trường và người dùng.
            """)
    
    st.subheader("Mục tiêu của dự án")
    st.markdown("""
            - Sử dụng các thuật toán machine learning xây dựng mô hình: 
                - Dự báo tương đối chính xác giá bán của các loại xe máy cũ căn cứ vào các thông số thực tế của xe phục vụ việc quảng cáo của người bán và việc tìm kiếm của người mua.
                - Phát hiện giá bán bất thường từ những thông số thực tế của xe máy rao bán.
            - Phát triển ứng dụng web để người sử dụng có thể truy xuất trực tuyến kết quả của các mô hình đã xây dựng.
            """)
    st.info("📁 Dataset từ trang chợ tốt gồm hơn 7000 xe từ 195 thương hiệu với nhiều phân khúc từ bình dân đến cao cấp.")
    # fig, ax = plt.subplots()
    # ax.hist(df["Giá"])
    # st.pyplot(fig)
    st.image("images/eda1.png")
    st.image("images/eda2.png")
    st.image("images/eda3.png")
    st.image("images/eda4.png")
    st.markdown("""
            - Phân bố giá xe có xu hướng lệch phải, nhiều xe giá thấp và ít xe giá cao, có giá trị outlier => bổ sung cột giá trị log của cột giá để giúp mô hình học tốt và ổn định hơn.
            - Phân bố số km đã đi có xu hướng lệch phải, nhiều xe có số km đã đi thấp và ít xe có số km đã đi cao.
            - Một số hãng xe có giá trị thương hiệu cao (như Harley Davidson, Triumph, BMW ), dung tích xe > 175cc, xuất xứ Đức, Mỹ ảnh hưởng đáng kể đến giá.

            """)  
# -----------------------------------------------------------------------------
# TAB 3: ĐÁNH GIÁ MÔ HÌNH
# -----------------------------------------------------------------------------
elif choice=="Đánh giá và lựa chọn mô hình thích hợp":
    st.subheader("Đánh giá và lựa chọn mô hình thích hợp cho bài toán dự đoán giá")
    st.image("danh_gia_mo_hinh.png")
    st.markdown("""
            - Mô hình XGBoost có kết quả r2 cao nhất so với các mô hình khác trên môi trường Scikit-learn.
            - Mô hình SVR, XGBoost cho giá trị MAE tốt nhất.
            - Vì vậy mô hình XGBoost sẽ được chọn để làm mô hình dự báo giá xe cũ.
            """)
    
    st.subheader("So sánh giá trị dự đoán và giá trị thực tế")
    st.image("so_sanh_gia_tri.png")
    st.markdown("""
            - Phần lớn các điểm số liệu nằm gần đường đỏ cho thấy mô hình dự đoán tương đối chính xác.
            - Tuy nhiên độ phân tán khá rộng, đặc biệt với các giá trị số liệu lớn. 
            """)

    st.subheader("Đánh giá và lựa chọn mô hình thích hợp cho bài toán cảnh báo bất thường")
    st.image("images/danh_gia_mo_hinh_anomaly.png")
    st.markdown("""
            - Mô hình có thể dự đoán giá xe máy cũ với các phương pháp biến động nhiều sai số trung bình khoảng 5–11% so với giá thực tế.
            - Các mô hình ISO Forest, IQR và Z-score (XGBoost) cho kết quả phát hiện bất thường khá gần nhau.
            - Do mô hình dự báo sử dụng XGBoost, nên Z-score (XGBoost) sẽ được sử dụng là phương pháp phát hiện bất thường do có độ tương thích cao với mô hình dự báo.
            """)

# -----------------------------------------------------------------------------
# TAB 4: CHỨC NĂNG DỰ ĐOÁN GIÁ
# -----------------------------------------------------------------------------
elif choice == 'Dự đoán giá xe cũ':
    st.header("🔮 Dự đoán giá xe cũ")
    if not price_res:
        st.error("⚠️ LỖI: Chưa tìm thấy file mô hình!")
        st.stop()

    # 1. Load dữ liệu mẫu để tạo danh sách gợi ý cho Dropdown
    try:
        df_sample_raw = pd.read_csv("subset_100motobikes.csv")
        THUONG_HIEU_LIST = sorted(df_sample_raw['Thương hiệu'].dropna().unique())
        DONG_XE_LIST = sorted(df_sample_raw['Dòng xe'].dropna().unique())
        LOAI_XE_LIST = sorted(df_sample_raw['Loại xe'].dropna().unique())
        DUNG_TICH_LIST = sorted(df_sample_raw['Dung tích xe'].dropna().unique())
        XUAT_XU_LIST = sorted(df_sample_raw['Xuất xứ'].dropna().unique())
        KHU_VUC_LIST = ['TP.HCM', 'Hà Nội', 'Đà Nẵng', 'Miền Nam (Lân cận)', 'Tỉnh thành khác']
    except:
        # Fallback nếu không có file dữ liệu
        THUONG_HIEU_LIST = ['Honda', 'Yamaha', 'Suzuki', 'Piaggio', 'SYM']
        DONG_XE_LIST = ['SH', 'Vision', 'Air Blade', 'Exciter', 'Wave']
        LOAI_XE_LIST = ['Tay ga', 'Xe số']
        DUNG_TICH_LIST = ['100 - 175 cc']
        XUAT_XU_LIST = ['Việt Nam']
        KHU_VUC_LIST = ['TP.HCM']

    # 2. Form nhập liệu
    st.write("### I. Thông tin xe")
    col1, col2 = st.columns(2)
    with col1:
        thuong_hieu = st.selectbox("Thương hiệu", THUONG_HIEU_LIST)
        dong_xe = st.selectbox("Dòng xe", DONG_XE_LIST)
        loai_xe = st.selectbox("Loại xe", LOAI_XE_LIST)
        tinh_trang = st.selectbox("Tình trạng", ['Đã sử dụng', 'Mới'])
        khu_vuc_ui = st.selectbox("Khu vực bán", KHU_VUC_LIST)
    with col2:
        dung_tich = st.selectbox("Dung tích", DUNG_TICH_LIST)
        xuat_xu = st.selectbox("Xuất xứ", XUAT_XU_LIST)
        nam = st.number_input("Năm đăng ký", 1990, 2025, 2020)
        km = st.number_input("Số Km đã đi", min_value=0, value=5000, step=1000)

    # 3. Chuẩn bị dữ liệu đầu vào cho Model
    input_dict = {
        'Thương hiệu': thuong_hieu, 'Dòng xe': dong_xe, 'Loại xe': loai_xe,
        'Dung tích xe': dung_tich, 'Xuất xứ': xuat_xu, 'nam': nam,
        'Số Km đã đi': km, 'Tình trạng': tinh_trang, 'Địa chỉ': khu_vuc_ui,
    }

    # 4. Dự đoán
    price = predict_price_value(input_dict, price_res)

    st.write("### II. Kết quả dự đoán")
    if st.button("💰 Dự đoán giá xe này"):
        st.success(f"Giá dự đoán tham khảo: **{price:,.2f} triệu VNĐ**")

# -----------------------------------------------------------------------------
# TAB 5: KIỂM TRA BẤT THƯỜNG & ĐĂNG TIN
# -----------------------------------------------------------------------------
elif choice == 'Phát hiện bất thường':
    st.header("🛡️ Kiểm tra & Đăng Tin")
    st.info(
        "Hệ thống sẽ kiểm tra giá. Nếu hợp lý, tin sẽ được đăng ngay. Nếu bất thường, cần sự xác nhận của bạn để gửi Admin.")

    if not price_res:
        st.error("⚠️ LỖI: Chưa tìm thấy file mô hình!")
        st.stop()

    # Load dữ liệu list (giống tab Dự đoán)
    try:
            df_sample_raw = pd.read_csv("subset_100motobikes.csv")
            THUONG_HIEU_LIST = sorted(df_sample_raw['Thương hiệu'].dropna().unique())
            DONG_XE_LIST = sorted(df_sample_raw['Dòng xe'].dropna().unique())
            LOAI_XE_LIST = sorted(df_sample_raw['Loại xe'].dropna().unique())
            DUNG_TICH_LIST = sorted(df_sample_raw['Dung tích xe'].dropna().unique())
            XUAT_XU_LIST = sorted(df_sample_raw['Xuất xứ'].dropna().unique())
            KHU_VUC_LIST = ['TP.HCM', 'Hà Nội', 'Đà Nẵng', 'Miền Nam (Lân cận)', 'Tỉnh thành khác']
    except:
        THUONG_HIEU_LIST = ['Honda', 'Yamaha']
        DONG_XE_LIST = ['SH', 'Vision']
        LOAI_XE_LIST = ['Tay ga']
        DUNG_TICH_LIST = ['100 - 175 cc']
        XUAT_XU_LIST = ['Việt Nam']
        KHU_VUC_LIST = ['TP.HCM']

    # --- NHẬP LIỆU ---
    st.write("### I. Nhập thông tin xe")
    col1, col2 = st.columns(2)
    with col1:
        thuong_hieu = st.selectbox("Thương hiệu", THUONG_HIEU_LIST, key='bt_th')
        dong_xe = st.selectbox("Dòng xe", DONG_XE_LIST, key='bt_dx')
        loai_xe = st.selectbox("Loại xe", LOAI_XE_LIST, key='bt_lx')
        tinh_trang = st.selectbox("Tình trạng", ['Đã sử dụng', 'Mới'], key='bt_tt')
        khu_vuc_ui = st.selectbox("Khu vực bán", KHU_VUC_LIST, key='bt_kv')
    with col2:
        dung_tich = st.selectbox("Dung tích", DUNG_TICH_LIST, key='bt_dt')
        xuat_xu = st.selectbox("Xuất xứ", XUAT_XU_LIST, key='bt_xx')
        nam = st.number_input("Năm đăng ký", 1990, 2025, 2020, key='bt_nam')
        km = st.number_input("Số Km đã đi", min_value=0, value=5000, step=1000, key='bt_km')

    input_dict = {
        'Thương hiệu': thuong_hieu, 'Dòng xe': dong_xe, 'Loại xe': loai_xe,
        'Dung tích xe': dung_tich, 'Xuất xứ': xuat_xu, 'nam': nam,
        'Số Km đã đi': km, 'Tình trạng': tinh_trang, 'Địa chỉ': khu_vuc_ui,
    }

    # Tính toán giá AI dự đoán
    ai_price = predict_price_value(input_dict, price_res)

    # --- KIỂM TRA & XỬ LÝ ---
    st.write("### II. Định giá bán")
    # st.caption(f"(AI định giá tham khảo: ~{ai_price:,.2f} triệu)")
    check_price = st.number_input("Nhập Giá bạn muốn bán (Triệu VNĐ)", step=1.0, format="%.2f")

    # Khởi tạo session state để lưu trạng thái xác nhận
    if 'confirm_abnormal' not in st.session_state:
        st.session_state.confirm_abnormal = False
    if 'abnormal_data' not in st.session_state:
        st.session_state.abnormal_data = None

    # Xử lý sự kiện bấm nút Kiểm tra
    if st.button("🚀 Kiểm tra & Đăng tin", type="primary"):
        if check_price <= 0:
            st.warning("Vui lòng nhập giá > 0")
            st.session_state.confirm_abnormal = False
        else:
            # Kiểm tra bất thường
            result = detect_anomaly(check_price, ai_price)

            # TRƯỜNG HỢP 1: GIÁ HỢP LÝ -> ĐĂNG NGAY
            if result['isAbnormal'] == 0:
                st.session_state.confirm_abnormal = False
                with st.spinner("Giá hợp lý. Đang đăng tin..."):
                    success, msg = save_normal_to_csv(input_dict, check_price, ai_price, result['reason'])
                    if success:
                        st.balloons()
                        st.success(f"✅ **ĐĂNG TIN THÀNH CÔNG!** {result['reason']}")
                        st.toast("Đã thêm vào danh sách bài đã đăng")
                    else:
                        st.error(f"Lỗi: {msg}")

            # TRƯỜNG HỢP 2: BẤT THƯỜNG -> KÍCH HOẠT CẢNH BÁO
            else:
                st.session_state.confirm_abnormal = True
                st.session_state.abnormal_data = {
                    'input': input_dict,
                    'check_price': check_price,
                    'ai_price': ai_price,
                    'reason': result['reason']
                }

    # Hiển thị UI xác nhận nếu phát hiện bất thường
    if st.session_state.confirm_abnormal and st.session_state.abnormal_data:
        st.divider()
        st.error(f"⚠️ **PHÁT HIỆN BẤT THƯỜNG:** {st.session_state.abnormal_data['reason']}")
        st.warning(
            "Tin này có mức giá chênh lệch lớn so với thị trường. Tin sẽ KHÔNG được đăng ngay mà phải chuyển qua Admin duyệt.")

        col_conf_1, col_conf_2 = st.columns([1, 1])
        # Nút xác nhận gửi Admin
        with col_conf_1:
            if st.button("⚠️ Xác nhận: Chuyển cho Admin"):
                data = st.session_state.abnormal_data
                success, msg = save_abnormal_to_csv(data['input'], data['check_price'], data['ai_price'],
                                                    data['reason'])
                if success:
                    st.info(f"📨 **Đã gửi yêu cầu.** {msg}")
                    st.session_state.confirm_abnormal = False  # Reset sau khi gửi
                    st.session_state.abnormal_data = None
                else:
                    st.error(msg)
        # Nút hủy
        with col_conf_2:
            if st.button("❌ Hủy bỏ"):
                st.session_state.confirm_abnormal = False
                st.session_state.abnormal_data = None
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 6: DANH SÁCH BÀI ĐÃ ĐĂNG (BÌNH THƯỜNG)
# -----------------------------------------------------------------------------
elif choice == 'Các Bài Đã Đăng':
    st.header("📝 Các Bài Đã Đăng")
    st.caption("Danh sách các tin đăng hợp lệ.")

    df_normal = load_data(OUTPUT_NORMAL_FILE)

    if df_normal.empty:
        st.info("Chưa có bài đăng nào.")
    else:
        # Thêm cột Checkbox 'Chọn' vào đầu DataFrame để thao tác
        if 'Chọn' not in df_normal.columns:
            df_normal.insert(0, "Chọn", False)
        else:
            df_normal['Chọn'] = False

        st.write(f"Tổng số bài: {len(df_normal)}")

        # Cấu hình hiển thị bảng
        column_config = {
            "Chọn": st.column_config.CheckboxColumn("Chọn", help="Tick để xóa", width="small"),
            "Gia_Thuc_Te_Trieu": st.column_config.NumberColumn("Giá Bán (Tr)", format="%.2f tr"),
            "Gia_AI_Du_Doan_Trieu": st.column_config.NumberColumn("AI Dự Đoán (Tr)", format="%.2f tr"),
            "Thời gian ghi nhận": st.column_config.DatetimeColumn("Thời gian", format="D/M/Y H:m"),
        }

        # Hiển thị bảng dữ liệu (cho phép chỉnh sửa cột checkbox)
        edited_df = st.data_editor(
            df_normal,
            column_config=column_config,
            disabled=[c for c in df_normal.columns if c != 'Chọn'],
            hide_index=True,
            use_container_width=True,
            key='editor_normal'
        )

        # Lấy danh sách các dòng được chọn
        selected_indices = edited_df[edited_df['Chọn'] == True].index.tolist()
        count_select = len(selected_indices)

        # Các nút chức năng Xóa
        st.divider()
        c1, c2, c3 = st.columns([2, 2, 6])

        with c1:
            if st.button(f"🗑️ Xóa ({count_select}) bài", type="primary", disabled=(count_select == 0)):
                msg, count = delete_rows(edited_df, OUTPUT_NORMAL_FILE, selected_indices)
                st.toast(f"{msg} Đã xóa {count} dòng.")
                st.rerun()

        with c2:
            if st.button("💥 Xóa TẤT CẢ"):
                if len(df_normal) > 0:
                    delete_rows(edited_df, OUTPUT_NORMAL_FILE, df_normal.index.tolist())
                    st.success("Đã xóa toàn bộ dữ liệu.")
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 7: QUẢN LÝ TIN BẤT THƯỜNG (ADMIN)
# -----------------------------------------------------------------------------
elif choice == 'Quản lý tin bất thường':
    st.header("🕵️ Duyệt Tin Bất Thường")
    st.caption("Admin xem xét các tin giá lệch cao/thấp để quyết định đăng hay xóa.")

    df_abnormal = load_data(OUTPUT_RESULT_FILE)

    if df_abnormal.empty:
        st.success("Sạch sẽ! Không có tin bất thường nào.")
    else:
        # Thêm cột Checkbox 'Chọn'
        if 'Chọn' not in df_abnormal.columns:
            df_abnormal.insert(0, "Chọn", False)
        else:
            df_abnormal['Chọn'] = False

        st.error(f"Cảnh báo: Có {len(df_abnormal)} tin cần duyệt.")

        column_config = {
            "Chọn": st.column_config.CheckboxColumn("Duyệt/Xóa", help="Tick để thực hiện thao tác", width="small"),
            "Gia_Thuc_Te_Trieu": st.column_config.NumberColumn("Giá Khách (Tr)", format="%.2f tr"),
            "Gia_AI_Du_Doan_Trieu": st.column_config.NumberColumn("AI (Tr)", format="%.2f tr"),
            "Ly_Do_Chi_Tiet": st.column_config.TextColumn("Lý do cảnh báo", width="medium"),
        }

        edited_df = st.data_editor(
            df_abnormal,
            column_config=column_config,
            disabled=[c for c in df_abnormal.columns if c != 'Chọn'],
            hide_index=True,
            use_container_width=True,
            key='editor_abnormal'
        )

        selected_indices = edited_df[edited_df['Chọn'] == True].index.tolist()
        count_select = len(selected_indices)

        # Thanh công cụ Admin (Duyệt/Xóa)
        st.divider()
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button(f"✅ Duyệt ({count_select})", type="primary", disabled=(count_select == 0),
                         help="Chuyển tin đã chọn sang mục Đã Đăng"):
                msg, count = move_to_normal(edited_df, selected_indices)
                st.success(f"{msg}")
                st.rerun()

        with c2:
            if st.button(f"🗑️ Xóa ({count_select})", disabled=(count_select == 0), help="Xóa vĩnh viễn tin đã chọn"):
                msg, count = delete_rows(edited_df, OUTPUT_RESULT_FILE, selected_indices)
                st.toast(f"Đã xóa {count} tin bất thường.")
                st.rerun()

        with c3:
            if st.button("✅ Duyệt TẤT CẢ", help="Chuyển TOÀN BỘ tin sang mục Đã Đăng"):
                if len(df_abnormal) > 0:
                    move_to_normal(edited_df, df_abnormal.index.tolist())
                    st.success("Đã duyệt tất cả!")
                    st.rerun()

        with c4:
            if st.button("💥 Xóa TẤT CẢ", help="Xóa sạch danh sách bất thường"):
                if len(df_abnormal) > 0:
                    delete_rows(edited_df, OUTPUT_RESULT_FILE, df_abnormal.index.tolist())
                    st.success("Đã xóa sạch danh sách bất thường.")
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 8: PHÂN CHIA CÔNG VIỆC TRONG NHÓM NGHIÊN CỨU
# -----------------------------------------------------------------------------
elif choice=="Phân chia công việc trong nhóm nghiên cứu":
    st.subheader("PHÂN CHIA CÔNG VIỆC TRONG NHÓM NGHIÊN CỨU")

    st.write('''### Nguyễn Ngọc Giao''')
    st.markdown("""
            - Tiền sử lý dữ liệu và xây dựng mô hình hồi quy trên môi trường Pyspark
            - Xây dựng mô hình phát hiện số liệu bất thường bằng Isolation Forest, khoảng giá trị Min/Max và tổng hợp kết quả
            - Xây dựng mô hình đề xuất các xe máy tương tự bằng Cosin similarity và Gensim
            - Xây dựng mô hình phân cụm Kmeans và Gausian Mixture Model trên môi trường Pyspark
            - Xây dựng GUI phần dự báo giá và phát hiện giá bất thường
            """)
    st.write('''### Nguyễn Thị Tuyển''')    
    st.markdown("""
            - Xây dựng mô hình dự báo trên môi trường Sklearn
            - Xây dựng mô hình phát hiện số liệu bất thường bằng IQR, KNN và Kmeans
            - Xây dựng mô hình đề xuất các xe máy tương tự bằng Cosin similarity và Gensim
            - Xây dựng mô hình phân cụm Kmeans, Gausian Mixture Model và Agglomerative Clustering trên môi trường Sklearn 
            - Xây dựng GUI phần Cosin similarity, Gensim và phân cụm 

            """)     










