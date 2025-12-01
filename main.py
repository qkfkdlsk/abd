import streamlit as st
import pandas as pd
import numpy as np
import io

# Streamlit 애플리케이션 제목 설정
st.title("📊 체력 측정 데이터 상관관계 분석 앱")
st.markdown("---")

# 데이터 파일 경로 (업로드된 파일 이름)
FILE_NAME = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"

@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 로드하고 필요한 전처리를 수행합니다.
    - 데이터프레임의 컬럼명을 정리하고, 숫자형 데이터로 변환할 수 없는 값은 NaN으로 처리합니다.
    """
    try:
        # 데이터 로드 시 인코딩 문제 방지를 위해 'cp949' 또는 'euc-kr' 사용 (한국어 환경)
        # 만약 로드에 실패하면 다른 인코딩으로 시도하거나 (utf-8 등), 사용자에게 인코딩 문제임을 알릴 수 있습니다.
        # Streamlit 환경에서는 사용자 업로드 파일이므로 직접 경로 지정 대신 업로드 기능을 사용해야 하지만,
        # 깃허브 코드 배포를 가정하여 파일 이름을 사용하고 로컬에 파일이 있다는 전제로 작성합니다.
        # 실제 Streamlit Cloud 배포 시에는 파일이 리포지토리에 포함되어야 합니다.
        
        # 파일 로드 (여기서는 로컬 파일 경로를 가정)
        df = pd.read_csv(file_path, encoding='euc-kr')

        # 컬럼명 정리 및 데이터 타입 변환
        # 모든 컬럼에서 숫자형이 아닌 데이터를 숫자로 변환하고, 변환 불가능한 값은 NaN으로 만듭니다.
        # 이 과정은 측정 데이터 외의 컬럼(ex. '센터명', '체력등급', '측정일')을 포함할 수 있으므로,
        # 분석에 사용할 '측정 항목' 컬럼만 선별적으로 숫자형으로 변환하는 것이 더 안전합니다.
        
        # 분석에 사용할 수치형 컬럼을 식별합니다. (예: 신장, 체중, 악력_좌, 윗몸말아올리기 등)
        # 모든 컬럼명 리스트
        all_columns = df.columns.tolist()
        
        # 제외할 비수치형/식별자 컬럼
        exclude_cols = ['센터명', '연령구분명', '체력등급', '측정일', '성별구분코드', '측정회차']
        
        # 분석 대상 수치형 컬럼 (간단하게 모든 컬럼에 대해 시도)
        numeric_cols = [col for col in all_columns if col not in exclude_cols]

        # 데이터 클리닝 및 숫자 변환
        for col in numeric_cols:
            # 쉼표(,) 제거 후 숫자 변환. 변환 불가 시 NaN 처리
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 숫자형 컬럼만 선택하여 반환
        return df[numeric_cols]

    except FileNotFoundError:
        st.error(f"⚠️ **파일을 찾을 수 없습니다:** `{file_path}`")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ **데이터 로드 중 오류 발생:** {e}")
        st.stop()
        
def get_correlation_analysis(df):
    """
    데이터프레임에서 상관관계 행렬을 계산하고, 가장 높은 양/음의 상관관계를 찾습니다.
    """
    # 상관관계 행렬 계산 (NaN이 있는 행은 자동으로 제외됨)
    corr_matrix = df.corr()
    
    # 자기 자신과의 상관관계(1) 및 중복 쌍을 제외하기 위해 상삼각 행렬만 사용
    corr_unstacked = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

    # 1. 가장 높은 양의 상관관계
    highest_positive_corr = corr_unstacked.iloc[[0]]
    
    # 2. 가장 높은 음의 상관관계
    lowest_negative_corr = corr_unstacked.iloc[[-1]]

    return highest_positive_corr, lowest_negative_corr

# 데이터 로드
data_df = load_data(FILE_NAME)

# 로드된 데이터의 기본 정보 표시
st.sidebar.header("📋 데이터 정보")
st.sidebar.dataframe(data_df.head(), use_container_width=True)
st.sidebar.text(f"행: {data_df.shape[0]}, 열: {data_df.shape[1]}")
st.sidebar.markdown(f"**분석에 사용된 컬럼:**\n{', '.join(data_df.columns)}")
st.markdown("---")


if data_df.empty or data_df.isnull().all().all():
    st.warning("⚠️ **분석 가능한 수치 데이터가 없거나 모두 결측치(NaN)입니다.** 데이터를 확인해 주세요.")
else:
    # 상관관계 분석 수행
    highest_pos, lowest_neg = get_correlation_analysis(data_df)
    
    st.header("🔍 데이터 간 상관관계 분석")

    # 상관관계 결과 표시 섹션
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("➕ 가장 높은 양의 상관관계")
        if st.button("양의 상관관계 결과 보기", key="pos_corr_btn"):
            if not highest_pos.empty:
                # 결과 포맷팅
                col1_name, col2_name = highest_pos.index[0]
                corr_value = highest_pos.values[0]
                
                st.success(f"### ✨ 결과")
                st.metric(label=f"**{col1_name}** 와 **{col2_name}**", value=f"{corr_value:.4f}")
                st.markdown(f"> **해석:** 두 변수는 강한 **정비례** 관계에 있습니다. 한 변수의 값이 증가하면 다른 변수의 값도 증가하는 경향이 매우 높습니다.")
            else:
                st.info("결과를 찾을 수 없습니다.")

    with col2:
        st.subheader("➖ 가장 높은 음의 상관관계")
        if st.button("음의 상관관계 결과 보기", key="neg_corr_btn"):
            if not lowest_neg.empty:
                # 결과 포맷팅
                col1_name, col2_name = lowest_neg.index[0]
                corr_value = lowest_neg.values[0]

                st.error(f"### ✨ 결과")
                st.metric(label=f"**{col1_name}** 와 **{col2_name}**", value=f"{corr_value:.4f}")
                st.markdown(f"> **해석:** 두 변수는 강한 **반비례** 관계에 있습니다. 한 변수의 값이 증가하면 다른 변수의 값은 감소하는 경향이 매우 높습니다.")
            else:
                st.info("결과를 찾을 수 없습니다.")
                
    st.markdown("---")
    
    # 선택적으로 전체 상관관계 행렬 표시
    st.header("🔢 전체 상관관계 행렬 (선택 사항)")
    if st.checkbox("전체 상관관계 행렬 데이터 보기"):
        st.dataframe(data_df.corr().style.background_gradient(cmap='coolwarm'), use_container_width=True)

    st.caption("결측치(NaN)가 있는 행은 상관관계 계산 시 자동으로 제외됩니다.")
