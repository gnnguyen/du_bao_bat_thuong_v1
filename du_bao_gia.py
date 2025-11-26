import numpy as np
import pandas as pd
import pickle
import warnings
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

PRICE_MODEL_PATH = 'price_model.pkl'


def extract_location(address):
    """Phân nhóm vùng miền từ địa chỉ"""
    address = str(address).lower()
    if 'hồ chí minh' in address or 'hcm' in address:
        return 'TP.HCM'
    elif 'hà nội' in address:
        return 'Hà Nội'
    elif 'đà nẵng' in address:
        return 'Đà Nẵng'
    elif 'bình dương' in address or 'đồng nai' in address:
        return 'Miền Nam (Lân cận)'
    else:
        return 'Tỉnh thành khác'


def extract_tech_features(row):
    """Trích xuất tính năng từ Tiêu đề và Mô tả"""
    text = str(row.get('Tiêu đề', '')) + " " + str(row.get('Mô tả chi tiết', ''))
    text = text.lower()

    features = {
        'has_abs': 1 if 'abs' in text else 0,
        'has_smartkey': 1 if any(x in text for x in ['smartkey', 'smart key', 'khoá thông minh']) else 0,
        'is_chinh_chu': 1 if 'chính chủ' in text else 0,
        'is_zin': 1 if 'zin' in text or 'nguyên bản' in text else 0
    }
    return pd.Series(features)


def preprocess_price_data(df):
    """Tiền xử lý nâng cao"""
    # 1. Chọn các cột cần thiết
    cols = ['Giá', 'Thương hiệu', 'Dòng xe', 'Năm đăng ký', 'Số Km đã đi',
            'Loại xe', 'Dung tích xe', 'Xuất xứ', 'Địa chỉ', 'Tình trạng',
            'Tiêu đề', 'Mô tả chi tiết']

    df = df[[c for c in cols if c in df.columns]].copy()

    # 2. Xử lý Giá
    if 'Giá' in df.columns and df['Giá'].dtype == 'object':
        df['Giá'] = (df['Giá'].str.replace(" đ", "", regex=False)
                     .str.replace(".", "", regex=False)
                     .replace(",", ".", regex=False)
                     .astype(float)) / 1000000

    # 3. Xử lý Năm & Tuổi xe
    current_year = 2025
    if 'Năm đăng ký' in df.columns:
        df['nam'] = pd.to_numeric(df['Năm đăng ký'], errors='coerce')
        df['nam'] = df['nam'].fillna(df['nam'].median())
        # Tạo feature Tuổi xe
        df['tuoi_xe'] = current_year - df['nam']
        df['tuoi_xe'] = df['tuoi_xe'].apply(lambda x: max(0, x))

    # 4. Feature Engineering từ Text (ABS, Smartkey...)
    tech_feats = df.apply(extract_tech_features, axis=1)
    df = pd.concat([df, tech_feats], axis=1)

    # 5. Xử lý Địa điểm
    if 'Địa chỉ' in df.columns:
        df['khu_vuc'] = df['Địa chỉ'].apply(extract_location)
    else:
        df['khu_vuc'] = 'Tỉnh thành khác'

    # 6. Lọc rác & Outliers
    if len(df) > 1:
        # Lọc dung tích rác
        df = df[~df['Dung tích xe'].isin(['Không biết rõ', 'Đang cập nhật', 'Nhật Bản'])]
        df.dropna(subset=['Giá', 'nam'], inplace=True)

        # Log transform Price để giảm tác động của xe quá đắt
        df['log_gia'] = np.log1p(df['Giá'])

        # Lọc outlier 3-sigma
        mean_log = df['log_gia'].mean()
        std_log = df['log_gia'].std()
        df = df[(df["log_gia"] < mean_log + 3 * std_log) &
                (df["log_gia"] > mean_log - 3 * std_log)].copy()

    return df


def train_price_model(data_path):
    print("--- [PRICE] Đang huấn luyện mô hình dự báo giá (Nâng cao)... ---")
    try:
        df_raw = pd.read_csv(data_path)
    except:
        df_raw = pd.read_excel(data_path)

    df = preprocess_price_data(df_raw)

    # Label Encoding cho các cột phân loại
    categorical_cols = ['Thương hiệu', 'Dòng xe', 'Loại xe', 'Dung tích xe', 'Xuất xứ', 'khu_vuc', 'Tình trạng']
    encoders = {}

    for col in categorical_cols:
        # Fill NA trước khi encode
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Danh sách Features dùng để train
    features = [
        'Số Km đã đi', 'tuoi_xe',
        'has_abs', 'has_smartkey', 'is_chinh_chu',
        'Thương hiệu_encoded', 'Dòng xe_encoded',
        'Loại xe_encoded', 'Dung tích xe_encoded',
        'Xuất xứ_encoded', 'khu_vuc_encoded',
        'Tình trạng_encoded'
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df['log_gia']

    # Tăng nhẹ complexity của model
    model = XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=6, random_state=42)
    model.fit(X_scaled, y)

    resources = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'features_list': features
    }

    with open(PRICE_MODEL_PATH, 'wb') as f:
        pickle.dump(resources, f)
    print(f"--- [PRICE] Đã lưu model với {len(features)} đặc trưng ---")


def predict_price_value(input_dict, resources):
    """Hàm dự đoán giá gọi từ UI (Cập nhật nhận input mới)"""
    encoders = resources['encoders']
    current_year = 2025

    # 1. Tính toán các feature dẫn xuất từ input
    tuoi_xe = max(0, current_year - input_dict['nam'])

    # Map địa chỉ sang khu vực
    khu_vuc = extract_location(input_dict.get('Địa chỉ', ''))

    # 2. Xây dựng vector input

    encoded_input = []

    # --- Nhóm Số ---
    encoded_input.append(input_dict['Số Km đã đi'])
    encoded_input.append(tuoi_xe)

    # --- Nhóm Boolean (Lấy từ Checkbox UI) ---
    encoded_input.append(int(input_dict.get('has_abs', 0)))
    encoded_input.append(int(input_dict.get('has_smartkey', 0)))
    encoded_input.append(int(input_dict.get('is_chinh_chu', 0)))

    # --- Nhóm Phân loại ---
    def safe_encode(col_name, val):
        val = str(val)
        try:
            return encoders[col_name].transform([val])[0]
        except:
            return encoders[col_name].transform([encoders[col_name].classes_[0]])[0]

    encoded_input.append(safe_encode('Thương hiệu', input_dict['Thương hiệu']))
    encoded_input.append(safe_encode('Dòng xe', input_dict['Dòng xe']))
    encoded_input.append(safe_encode('Loại xe', input_dict['Loại xe']))
    encoded_input.append(safe_encode('Dung tích xe', input_dict['Dung tích xe']))
    encoded_input.append(safe_encode('Xuất xứ', input_dict['Xuất xứ']))
    encoded_input.append(safe_encode('khu_vuc', khu_vuc))
    encoded_input.append(safe_encode('Tình trạng', input_dict['Tình trạng']))

    # 3. Predict
    X_in = np.array(encoded_input).reshape(1, -1)

    if 'scaler' in resources:
        X_scaled = resources['scaler'].transform(X_in)
    else:
        X_scaled = X_in

    log_price = resources['model'].predict(X_scaled)[0]
    return np.expm1(log_price)


if __name__ == "__main__":
    path = 'data_motobikes.xlsx'  # File csv
    train_price_model(path)