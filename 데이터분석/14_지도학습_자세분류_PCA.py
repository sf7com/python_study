import pandas as pd
#1) 데이터 로드
f_name_df = pd.read_csv('./데이터분석/data/features.txt',
    sep='\s+', names=['index', 'feature_name'])
f_name_df.head()
f_name_df.info()

#피처 이름의 중복값
f_name_df.duplicated(['feature_name']).value_counts()
# False    477
# True      84
f_name_df[f_name_df.duplicated(['feature_name'])]
#피처 이름 수정 : 인덱스_기존피처이름
f_name_df['feature_name'] = \
    f_name_df['index'].astype('str') + "_" + f_name_df['feature_name']
f_name_df['feature_name'].head()


#훈련/테스트 데이터 로드
X_train = pd.read_csv('./데이터분석/data/X_train.txt', sep='\s+',
        header=None, names=f_name_df['feature_name'])
X_train.head()
X_train.info()
X_train.isnull().sum()

X_test = pd.read_csv('./데이터분석/data/X_test.txt', sep='\s+',
        header=None, names=f_name_df['feature_name'])
X_test.head()
X_test.info()

Y_train = pd.read_csv('./데이터분석/data/y_train.txt', sep='\s+',
        header=None, names=['action'])
Y_train.head()
Y_train.info()

Y_test = pd.read_csv('./데이터분석/data/y_test.txt', sep='\s+',
        header=None, names=['action'])
Y_test.head()
Y_test.info()

#label 데이터 로드
label_df = pd.read_csv('./데이터분석/data/activity_labels.txt', sep='\s+',
        header=None, names=['index', 'label'])
label_df
#Y데이터들 숫자-> 문자열label으로 변환
label_df.set_index('index').T.to_dict('label')
#{1: ['WALKING'], 2: ['WALKING_UPSTAIRS'], 3: ['WALKING_DOWNSTAIRS'], 4: ['SITTING'], 5: ['STANDING'], 6: ['LAYING']}
label_dic = label_df.set_index('index').T.to_dict('label')
Y_train = Y_train.replace({'action':label_dic})
Y_test = Y_test.replace({'action':label_dic})
Y_train.head()
Y_test.head()
#------------------------------------------------------
#데이터 섞기
from sklearn.utils import shuffle
X_train_r = shuffle(X_train, random_state=13)
Y_train_r = shuffle(Y_train, random_state=13)

X_test_r = shuffle(X_test, random_state=43)
Y_test_r = shuffle(Y_test, random_state=43)

#데이터 차원축소
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)
pca.fit(X_train_r)
pca.components_.shape #(34, 561) 총 주성분 34개, 원본차원 561개
X_train_r_pca = pca.transform(X_train_r)
X_test_r_pca = pca.transform(X_test_r)
X_train_r_pca.shape #총 7352개 데이터, 차원 34
sum(pca.explained_variance_ratio_) #0.90093

#3) 분석모델 구축 및 훈련 - 의사결정나무
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=156)
model.fit(X_train_r_pca, Y_train_r)
model.score(X_train_r_pca, Y_train_r) #0.997 => 1.0
model.score(X_test_r_pca, Y_test_r) #0.8661 => 0.8069

model = DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01,
     random_state=156)
model.fit(X_train_r_pca, Y_train_r)
model.score(X_train_r_pca, Y_train_r) #0.85895 =>0.8264
model.score(X_test_r_pca, Y_test_r) #0.85153 =>0.8154

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_r_pca, Y_train_r)
model.score(X_train_r_pca, Y_train_r) #0.9502
model.score(X_test_r_pca, Y_test_r) #0.9121

