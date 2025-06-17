# scripts/preprocess_lora_log.py

import pandas as pd
import os
import glob

def preprocess_all_logs(input_dir: str, output_filepath: str):
    """
    지정된 디렉토리의 모든 로그 파일을 읽어 통합하고,
    학습에 필요한 데이터만 추출 및 정제합니다.

    Args:
        input_dir (str): 원본 로그 파일들이 있는 디렉토리 경로.
        output_filepath (str): 정제된 데이터가 저장될 단일 CSV 파일 경로.
    """
    # 1. 지정된 디렉토리에서 모든 .csv 파일 목록 가져오기
    log_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not log_files:
        print(f"오류: '{input_dir}' 디렉토리에서 로그 파일(.csv)을 찾을 수 없습니다.")
        return

    print(f"총 {len(log_files)}개의 로그 파일을 발견했습니다. 통합을 시작합니다...")
    
    # 모든 로그 파일을 하나로 합치기
    df_list = []
    for file in log_files:
        try:
            df_list.append(pd.read_csv(file, low_memory=False))
        except Exception as e:
            print(f"경고: '{file}' 파일을 읽는 중 오류 발생. 건너뜁니다. ({e})")
    
    if not df_list:
        print("오류: 유효한 로그 파일을 읽지 못했습니다.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"모든 로그 파일 통합 완료. 총 로그 수: {len(full_df)}")

    # 2. 'DECODE_SUCCESS' 이벤트인 행만 필터링
    df_decoded = full_df[full_df['event_type'] == 'DECODE_SUCCESS'].copy()
    print(f"유효한 데이터 로그(DECODE_SUCCESS) 행: {len(df_decoded)}개")

    # 3. 필요한 13개 피처 컬럼 선택
    feature_columns = [
        #'decoded_ts',
        'decoded_accel_ax', 'decoded_accel_ay', 'decoded_accel_az',
        'decoded_gyro_gx', 'decoded_gyro_gy', 'decoded_gyro_gz',
        'decoded_angle_roll', 'decoded_angle_pitch', 'decoded_angle_yaw',
        'decoded_gps_lat', 'decoded_gps_lon', 'decoded_gps_altitude'
    ]
    
    # 선택한 컬럼이 모두 존재하는지 확인
    missing_cols = [col for col in feature_columns if col not in df_decoded.columns]
    if missing_cols:
        print(f"오류: 필수 컬럼이 로그 파일에 없습니다: {missing_cols}")
        return

    df_features = df_decoded[feature_columns].copy()

    # 4. 데이터 타입 변환 및 결측치(NaN) 처리
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    
    rows_before_na_drop = len(df_features)
    df_features.dropna(inplace=True)
    rows_after_na_drop = len(df_features)
    if rows_before_na_drop > rows_after_na_drop:
        print(f"{rows_before_na_drop - rows_after_na_drop}개의 행에서 결측치(NaN)가 발견되어 제거되었습니다.")

    # 5. 위도 또는 경도가 0인 데이터 제거
    rows_before_gps_drop = len(df_features)
    # 위도가 0이거나 경도가 0인 행을 식별
    gps_zero_mask = (df_features['decoded_gps_lat'] == 0) | (df_features['decoded_gps_lon'] == 0)
    df_features = df_features[~gps_zero_mask] # 마스크의 반대로 필터링
    rows_after_gps_drop = len(df_features)
    if rows_before_gps_drop > rows_after_gps_drop:
        print(f"{rows_before_gps_drop - rows_after_gps_drop}개의 행에서 GPS 좌표가 0으로 기록되어 제거되었습니다.")
        
    if df_features.empty:
        print("오류: 모든 데이터를 정제한 후 남은 데이터가 없습니다.")
        return
    
    # 6. 위도/경도 피처에서 정수부 제거 (소수부만 남김)
    print("위도/경도 피처에서 정수부를 제거하여 상대적 위치 변화에 집중합니다.")
    df_features['decoded_gps_lat'] = df_features['decoded_gps_lat'] - df_features['decoded_gps_lat'].astype(int)
    df_features['decoded_gps_lon'] = df_features['decoded_gps_lon'] - df_features['decoded_gps_lon'].astype(int)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    if df_features.empty:
        print("오류: 모든 데이터를 정제한 후 남은 데이터가 없습니다.")
        return

    # 7. 정제된 데이터 저장 (기존 6번 단계)
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_features.to_csv(output_filepath, index=False)
    
    print("-" * 30)
    print(f"✅ 최종 데이터 정제 완료!")
    print(f"   최종 학습 데이터 샘플 수: {len(df_features)}")
    print(f"   저장 경로: '{output_filepath}'")

if __name__ == '__main__':
    # 원본 로그 파일들이 있는 디렉토리
    RAW_LOGS_DIR = 'data/original/lora_logs' 
    # 정제된 데이터가 저장될 파일 경로
    CLEAN_DATA_FILE = 'data/original/clean_lora_data_combined.csv'
    
    preprocess_all_logs(RAW_LOGS_DIR, CLEAN_DATA_FILE)