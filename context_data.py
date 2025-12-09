import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split
import re

# ===========================================================
# 1) Publisher Cleaning
# ===========================================================

def clean_publisher(pub: str) -> str:
    """출판사 문자열 정제"""
    if pd.isna(pub):
        return np.nan
    
    pub = str(pub).strip().lower()
    pub = re.sub(r'[^a-zA-Z0-9 &]', '', pub)  # 기본적인 허용 문자만 남기기
    pub = re.sub(r'\s+', ' ', pub)            # 중복 공백 제거

    return pub if pub != "" else np.nan

def encode_publisher_frequency_with_cutoff(df, min_count=5):
    """
    출판사 빈도 기반 인코딩 + 희귀 출판사 처리
    """
    # 1) 정제
    df['publisher'] = df['publisher'].apply(clean_publisher)

    # 2) 빈도 계산
    freq = df['publisher'].value_counts()

    # 3) frequency feature 생성 (선택)
    df['publisher_freq'] = df['publisher'].map(freq)

    # 4) 희귀 출판사 → "other_publisher"
    df['publisher_clean'] = df['publisher'].apply(
        lambda x: x if (not pd.isna(x) and freq[x] >= min_count) else "other_publisher"
    )

    return df


# ===========================================================
# 1) Author Cleaning
# ===========================================================
def clean_author(author: str) -> str:
    """작가 이름을 간단히 정제"""
    if pd.isna(author):
        return np.nan
    author = str(author).strip().lower()
    author = re.sub(r'[^a-zA-Z ]', '', author)
    return author if author != "" else np.nan


# ===========================================================
# 2) Author Frequency Encoding + Rare 처리
# ===========================================================
def encode_author_frequency_with_cutoff(df, min_count=5):
    """
    작가 빈도 기반 인코딩 + 희귀 작가 처리
    """
    df['book_author'] = df['book_author'].apply(clean_author)

    freq = df['book_author'].value_counts()

    df['author_freq'] = df['book_author'].map(freq)

    # rare author 처리
    df['author_clean'] = df['book_author'].apply(
        lambda x: x if (not pd.isna(x) and freq[x] >= min_count) else "other_author"
    )
    return df

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
            res.pop(i)

    return res

# ===========================================================
# 1. ISBN Cleaning
# ===========================================================
def clean_isbn(isbn: str) -> str:
    if pd.isna(isbn):
        return np.nan
    
    isbn = str(isbn).strip().replace(" ", "").replace("-", "")
    isbn = isbn.upper()
    isbn = re.sub(r'[^0-9X]', '', isbn)  # 숫자 + X만 남기기

    if len(isbn) not in [10, 13]:
        return np.nan

    return isbn

# ===========================================================
# 2. 동일 책 대표 ISBN 통일
# ===========================================================
def unify_isbn(books):
    isbn_mode_map = books.groupby(['book_title', 'book_author'])['isbn'] \
                         .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]) \
                         .reset_index()

    isbn_mode_map = isbn_mode_map.rename(columns={'isbn': 'rep_isbn'})
    books = books.merge(isbn_mode_map, on=['book_title', 'book_author'], how='left')

    books['isbn'] = books['rep_isbn']
    books = books.drop(columns=['rep_isbn'])
    return books

# ===========================================================
# 3. Rare ISBN 처리
# ===========================================================
def reduce_rare_isbn(books, min_count=3):
    isbn_counts = books['isbn'].value_counts()
    rare_isbns = isbn_counts[isbn_counts < min_count].index

    books['isbn'] = books['isbn'].apply(lambda x: 'rare_isbn' if x in rare_isbns else x)
    return books


def process_context_data(users, books):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()
    
    # -------------------------
    # ISBN 전처리 전체 파이프라인
    # -------------------------
    books_['isbn'] = books_['isbn'].apply(clean_isbn)
    books_ = unify_isbn(books_)
    books_ = reduce_rare_isbn(books_, min_count=3)

    # -------------------------
    # BOOKS 전처리
    # -------------------------
    
    # 1) Author frequency + rare 처리
    books_ = encode_author_frequency_with_cutoff(books_, min_count=20)
    
    # 2) Publisher frequency + rare 처리
    books_ = encode_publisher_frequency_with_cutoff(books_, min_count=20)
    
    # 3) category를 첫 번째 값으로 고정
    books_['category'] = books_['category'].apply(
        lambda x: str2list(x)[0] if not pd.isna(x) else np.nan
    )
    
    # 4) 언어 결측치 보완
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    
    # 5) 출판연도 binning
    books_['publication_range'] = books_['year_of_publication'].apply(
        lambda x: x // 10 * 10
    )  # 1990년대, 2000년대, 2010년대, ...

    # -------------------------
    # USERS 전처리
    # -------------------------
    users_['age'] = users_['age'].fillna(-1)
    
    # --- 도메인 기반 Age Binning ---
    bins = [0, 12, 17, 24, 34, 44, 54, 150]
    labels = [
        'child', 'teen', 'young_adult',
        'adult_25_34', 'adult_35_44', 'adult_45_54',
        'senior'
    ]
    
    users_['age_group'] = pd.cut(
        users_['age'],
        bins=bins,
        labels=labels,
        right=True
    )
    
    # 3) 음수(-1) 또는 binning 불가(NA)는 unknown 처리
    users_['age_group'] = users_['age_group'].astype(str)
    users_.loc[users_['age'] < 0, 'age_group'] = 'unknown'
    users_['age_group'] = users_['age_group'].fillna('unknown')
    
    # 기존 age_range 삭제 (사용 안 함)
    # users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)  # 10대, 20대, 30대, ...
    
    # -------------------------
    # LOCATION PROCESSING
    # -------------------------
    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    
    # LOCATION 결측치 보완
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
            
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
                
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state

               
    
    users_ = users_.drop(['location'], axis=1)

    return users_, books_

# ===========================================================
# CONTEXT DATA LOAD
# ===========================================================
def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'age_group', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'author_clean', 'publisher_clean', 'language', 'category', 'publication_range']
    
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'}) if args.model == 'NCF' \
                   else user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')
                  
    test_df = test_df.loc[test.index, sparse_cols]              
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        # 1) 우선 문자열로 변환 후 fillna
        all_df[col] = all_df[col].astype(str).fillna("unknown")

        # 2) category 변환 전에 unknown을 포함시키기 위해 한번 category로 변환
        all_df[col] = all_df[col].astype("category")

        # 3) 실제 카테고리 목록 얻기
        unique_labels = all_df[col].cat.categories
        
        # 4) mapping 저장
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        
        # 5) 다시 train/test에 code 적용
        train_df[col] = train_df[col].astype("category").cat.codes
        test_df[col] = test_df[col].astype("category").cat.codes
    
    field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']

    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }

    return data

# ===========================================================
# SPLIT & LOADER
# ===========================================================
def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def context_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
