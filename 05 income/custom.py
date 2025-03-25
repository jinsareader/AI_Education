import pandas
#원 핫 인코딩 함수
def one_hot_encoding(df : pandas.DataFrame, cols : list) :
    for c in cols :        
        onehot = pandas.get_dummies(df[c], c)
        df = df.join(onehot)
        df = df.drop(c, axis = 1)
    return df;

#이상치 제거 함수
def del_anomaly(df : pandas.DataFrame, cols : list, factor : float = 2) :
    for c in cols :
        min_lim = df[c].mean() - df[c].std() * factor
        max_lim = df[c].mean() + df[c].std() * factor
        df = df[(df[c] > min_lim) & (df[c] < max_lim)]
    return df;

#정규화 함수
def scaling(df : pandas.DataFrame, cols : list, origin_df : pandas.DataFrame = None) :
    if origin_df is None :
        for c in cols :
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    else :
        for c in cols :
            df[c] = (df[c] - origin_df[c].min()) / (origin_df[c].max() - origin_df[c].min())
    return df;