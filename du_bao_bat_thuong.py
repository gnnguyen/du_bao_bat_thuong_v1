import pandas as pd
import numpy as np
import pickle
import re
import os
from datetime import datetime

# --- IMPORT ĐỒNG BỘ TỪ FILE DỰ BÁO GIÁ ---
from du_bao_gia import predict_price_value, PRICE_MODEL_PATH, extract_tech_features

# --- CẤU HÌNH FILE ---
# 1. File dữ liệu gốc (để đọc và xử lý hàng loạt)
INPUT_DATA_FILE = 'data_motobikes.xlsx'
# 2. File kết quả (Lưu cả kết quả chạy batch VÀ kết quả lưu từ GUI)
OUTPUT_RESULT_FILE = 'ket_qua_bat_thuong.csv'
# 3. HẰNG SỐ MỚI: File lưu các trường hợp BÌNH THƯỜNG (cho tab mới)
OUTPUT_NORMAL_FILE = 'ket_qua_binh_thuong.csv'

# Ngưỡng lệch 25%
THRESHOLD_PERCENT = 0.25


# =============================================================================
# HÀM HỖ TRỢ
# =============================================================================

def clean_price_to_million(price_str):
    """Chuyển đổi giá text sang số thực (Đơn vị: TRIỆU ĐỒNG)"""
    if pd.isna(price_str): return 0
    price_str = str(price_str).lower()
    try:
        if 'tr' in price_str:
            clean_val = re.sub(r'[^\d\.,]', '', price_str).replace(',', '.')
            return float(clean_val)
        clean_val = re.sub(r'[^\d]', '', price_str)
        if clean_val == '': return 0
        return float(clean_val) / 1_000_000
    except:
        return 0


# =============================================================================
# 1. HÀM DỰ ĐOÁN & KIỂM TRA
# =============================================================================

def detect_anomaly(user_price, predicted_price):
    """
    So sánh giá người dùng nhập và giá AI dự đoán.
    Trả về dictionary kết quả.
    """
    try:
        if predicted_price == 0:
            return {'isAbnormal': 1, 'reason': "Không thể định giá (Lỗi Model/Dữ liệu)"}

        if user_price <= 0:
            return {'isAbnormal': 1, 'reason': "Giá nhập vào không hợp lệ"}

        # Công thức độ lệch %
        diff_percent = (user_price - predicted_price) / predicted_price

        # Kiểm tra ngưỡng
        if diff_percent < -THRESHOLD_PERCENT:
            return {
                'isAbnormal': 1,
                'reason': f"Giá RẺ bất thường. AI dự đoán: {predicted_price:,.2f} tr. (Thấp hơn {abs(diff_percent):.0%})"
            }
        elif diff_percent > THRESHOLD_PERCENT:
            return {
                'isAbnormal': 1,
                'reason': f"Giá CAO bất thường. AI dự đoán: {predicted_price:,.2f} tr. (Cao hơn {diff_percent:.0%})"
            }

        return {'isAbnormal': 0, 'reason': f"Giá hợp lý (Chênh lệch {diff_percent:.0%})"}

    except Exception as e:
        return {'isAbnormal': 0, 'reason': f"Lỗi kiểm tra: {str(e)}"}


# =============================================================================
# 2. HÀM ĐỌC FILE ĐẦU VÀO VÀ DỰ ĐOÁN CẢ FILE
# =============================================================================

def process_batch_anomalies(input_path=INPUT_DATA_FILE, output_path=OUTPUT_RESULT_FILE):
    """
    Đọc file CSV gốc, dự đoán từng dòng và lưu ra file kết quả.
    Lưu ý: Hàm này GHI ĐÈ file output_path.
    """
    print(f"📂 Đang đọc dữ liệu từ: {input_path}...")

    # Load Model
    if not os.path.exists(PRICE_MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy model '{PRICE_MODEL_PATH}'")
        return
    try:
        with open(PRICE_MODEL_PATH, 'rb') as f:
            resources = pickle.load(f)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # Đọc File
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    print(f"✅ Đã tải {len(df)} dòng. Đang xử lý...")

    predictions = []
    reasons = []
    is_abnormal_list = []
    prices_million = []

    # Thêm cột thời gian (batch)
    df['Thời gian ghi nhận'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Loop xử lý từng dòng
    for index, row in df.iterrows():
        # Lấy giá thực tế
        actual_price_million = clean_price_to_million(row.get('Giá', 0))
        prices_million.append(actual_price_million)

        # Trích xuất feature text (Dùng hàm import từ du_bao_gia)
        tech_features = extract_tech_features(row)

        # Chuẩn bị input cho model (Đồng bộ cột với GUI)
        try:
            nam_dk = float(row.get('Năm đăng ký', 2019))
        except:
            nam_dk = 2019

        input_dict = {
            'Thương hiệu': row.get('Thương hiệu', 'Unknown'),
            'Dòng xe': row.get('Dòng xe', 'Unknown'),
            'Loại xe': row.get('Loại xe', 'Tay ga'),
            'Dung tích xe': row.get('Dung tích xe', '100 - 175 cc'),
            'Xuất xứ': row.get('Xuất xứ', 'Việt Nam'),
            'nam': nam_dk,
            'Số Km đã đi': float(row.get('Số Km đã đi', 5000) if pd.notnull(row.get('Số Km đã đi')) else 5000),
            'Tình trạng': row.get('Tình trạng', 'Đã sử dụng'),
            'Địa chỉ': row.get('Địa chỉ', '')#,
            # 'has_abs': tech_features['has_abs'],
            # 'has_smartkey': tech_features['has_smartkey'],
            # 'is_chinh_chu': tech_features['is_chinh_chu']
        }

        # Dự đoán
        try:
            pred_price = predict_price_value(input_dict, resources)
        except:
            pred_price = 0
        predictions.append(pred_price)

        # Kiểm tra bất thường
        res = detect_anomaly(actual_price_million, pred_price)
        is_abnormal_list.append(res['isAbnormal'])

        # Chỉ lưu lý do bất thường nếu có
        reason_text = res['reason'] if res['isAbnormal'] == 1 else ""
        reasons.append(reason_text)

    # Thêm cột kết quả vào DataFrame
    df['Gia_Thuc_Te_Trieu'] = prices_million
    df['Gia_AI_Du_Doan_Trieu'] = predictions
    df['Co_Bat_Thuong'] = is_abnormal_list
    df['Ly_Do_Chi_Tiet'] = reasons

    # Chỉ giữ lại các dòng bất thường
    df_abnormal_batch = df[df['Co_Bat_Thuong'] == 1].copy()

    # Lưu ra file (Ghi đè để tạo file chuẩn)
    if not df_abnormal_batch.empty:
        df_abnormal_batch.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ HOÀN TẤT BATCH! Đã lưu {len(df_abnormal_batch)} trường hợp bất thường tại: {output_path}")
    else:
        print(f"✅ HOÀN TẤT BATCH! Không tìm thấy bất thường nào.")
        # Tạo file rỗng với header nếu không có bất thường
        df_empty = pd.DataFrame(columns=df.columns)
        df_empty.to_csv(output_path, index=False, encoding='utf-8-sig')


# =============================================================================
# 3. HÀM LƯU TỪ GUI VÀO FILE KẾT QUẢ
# =============================================================================

# CẬP NHẬT: Thêm tham số predicted_price và reason
def save_abnormal_to_csv(input_dict, check_price, predicted_price, reason, file_path=OUTPUT_RESULT_FILE):
    """
    Nhận các thông tin từ GUI và lưu các trường hợp BẤT THƯỜNG vào file CSV.
    Lưu ý: Đảm bảo dòng mới nhất NẰM Ở ĐẦU.
    """
    try:
        # Chuẩn bị dòng dữ liệu mới (Mapping từ GUI input -> Cột CSV)
        new_row = {
            'Tiêu đề': f"Cảnh báo GUI: Giá {check_price:,.2f}tr cho {input_dict['Thương hiệu']} {input_dict['Dòng xe']}",
            'Giá': f"{check_price:,.2f} tr",  # Lưu giá người dùng nhập
            'Gia_Thuc_Te_Trieu': check_price,
            'Gia_AI_Du_Doan_Trieu': predicted_price,
            'Co_Bat_Thuong': 1,  # Đã gọi hàm save tức là có bất thường
            'Ly_Do_Chi_Tiet': reason,
            'Thương hiệu': input_dict['Thương hiệu'],
            'Dòng xe': input_dict['Dòng xe'],
            'Loại xe': input_dict['Loại xe'],
            'Năm đăng ký': input_dict['nam'],
            'Số Km đã đi': input_dict['Số Km đã đi'],
            'Tình trạng': input_dict['Tình trạng'],
            'Dung tích xe': input_dict['Dung tích xe'],
            'Xuất xứ': input_dict['Xuất xứ'],
            'Địa chỉ': input_dict['Địa chỉ'],
            # 'Mô tả chi tiết': f"Nguồn: GUI Input. ABS: {input_dict['has_abs']}, Smartkey: {input_dict['has_smartkey']}. Khu vực: {input_dict['Địa chỉ']}",
            'Thời gian ghi nhận': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        new_df = pd.DataFrame([new_row])

        # Logic Append: Đọc file cũ, nối dòng mới lên đầu, rồi ghi đè
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, encoding='utf-8-sig')
            # Nối new_df lên trên existing_df
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
            combined_df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        else:
            # File không tồn tại thì tạo mới
            new_df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')

        return True, f"Đã lưu vào {file_path}"

    except Exception as e:
        return False, f"Lỗi lưu file: {str(e)}"

# HÀM MỚI: LƯU DỮ LIỆU BÌNH THƯỜNG
def save_normal_to_csv(input_dict, check_price, predicted_price, reason, file_path=OUTPUT_NORMAL_FILE):
    """
    Nhận các thông tin từ GUI và lưu các trường hợp BÌNH THƯỜNG vào file CSV.
    """
    try:
        # Chuẩn bị dòng dữ liệu mới (Mapping từ GUI input -> Cột CSV)
        new_row = {
            'Tiêu đề': f"Bài đăng hợp lệ: Giá {check_price:,.2f}tr cho {input_dict['Thương hiệu']} {input_dict['Dòng xe']}",
            'Giá': f"{check_price:,.2f} tr",  # Lưu giá người dùng nhập
            'Gia_Thuc_Te_Trieu': check_price,
            'Gia_AI_Du_Doan_Trieu': predicted_price,
            'Co_Bat_Thuong': 0,  # Bình thường
            'Ly_Do_Chi_Tiet': reason,
            'Thương hiệu': input_dict['Thương hiệu'],
            'Dòng xe': input_dict['Dòng xe'],
            'Loại xe': input_dict['Loại xe'],
            'Năm đăng ký': input_dict['nam'],
            'Số Km đã đi': input_dict['Số Km đã đi'],
            'Tình trạng': input_dict['Tình trạng'],
            'Dung tích xe': input_dict['Dung tích xe'],
            'Xuất xứ': input_dict['Xuất xứ'],
            'Địa chỉ': input_dict['Địa chỉ'],
            'Thời gian ghi nhận': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        new_df = pd.DataFrame([new_row])

        # Logic Append: Đọc file cũ, nối dòng mới lên đầu, rồi ghi đè
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, encoding='utf-8-sig')
            # Nối new_df lên trên existing_df
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
            combined_df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        else:
            # File không tồn tại thì tạo mới
            new_df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8-sig')

        return True, f"Đã lưu vào {file_path}"

    except Exception as e:
        return False, f"Lỗi lưu file: {str(e)}"


if __name__ == "__main__":
    # Chạy thử batch process khi gọi file này
    process_batch_anomalies(input_path=INPUT_DATA_FILE)