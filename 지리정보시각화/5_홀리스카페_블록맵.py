import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#그래프 폰트 변경
plt.rc('font', family='malgun gothic')
#블록맵 행정구역 경계선 x,y 데이터
BORDER_LINES = [
    [(3, 2), (5, 2), (5, 3), (9, 3), (9, 1)], # 인천
    [(2, 5), (3, 5), (3, 4), (8, 4), (8, 7), (7, 7), (7, 9), (4, 9), (4, 7), (1, 7)], # 서울
    [(1, 6), (1, 9), (3, 9), (3, 10), (8, 10), (8, 9),
     (9, 9), (9, 8), (10, 8), (10, 5), (9, 5), (9, 3)], # 경기도
    [(9, 12), (9, 10), (8, 10)], # 강원도
    [(10, 5), (11, 5), (11, 4), (12, 4), (12, 5), (13, 5),
     (13, 4), (14, 4), (14, 2)], # 충청남도
    [(11, 5), (12, 5), (12, 6), (15, 6), (15, 7), (13, 7),
     (13, 8), (11, 8), (11, 9), (10, 9), (10, 8)], # 충청북도
    [(14, 4), (15, 4), (15, 6)], # 대전시
    [(14, 7), (14, 9), (13, 9), (13, 11), (13, 13)], # 경상북도
    [(14, 8), (16, 8), (16, 10), (15, 10),
     (15, 11), (14, 11), (14, 12), (13, 12)], # 대구시
    [(15, 11), (16, 11), (16, 13)], # 울산시
    [(17, 1), (17, 3), (18, 3), (18, 6), (15, 6)], # 전라북도
    [(19, 2), (19, 4), (21, 4), (21, 3), (22, 3), (22, 2), (19, 2)], # 광주시
    [(18, 5), (20, 5), (20, 6)], # 전라남도
    [(16, 9), (18, 9), (18, 8), (19, 8), (19, 9), (20, 9), (20, 10)], # 부산시
]

#블록맵의 블록에 데이터 매핑 후 색을 표시하여 블록맵 그리는 함수
def draw_blockMap(blockedMap, targetData, filePath, color):
    whitelabelmin = (max(blockedMap[targetData]) - min(blockedMap[targetData])) * 0.25 + min(blockedMap[targetData])
    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(8, 13))
    temptitle = filePath.split('\\')[-1].split('.')[0]
    plt.title(temptitle)
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=color, edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'

        if row.isna()[0] :
            continue
        
        dispname = row['shortName']

        """
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. (중구, 서구)
        if row['광역시도'].endswith('시') and not row['광역시도'].startswith('세종'):
            dispname = '{}\n{}'.format(row['광역시도'][:2], row['행정구역'][:-1])
            if len(row['행정구역']) <= 2:
                dispname += row['행정구역'][-1]
        else:
            dispname = row['행정구역'][:-1]
        """       
        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 7.5, 1.5
        else:
            fontsize, linespacing = 11, 1.2

        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                      fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)
    
    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=4)

    plt.gca().invert_yaxis()
    #plt.gca().set_aspect(1)
    plt.axis('off')
    
    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()

    plt.savefig(filePath)
    plt.show()

#---------------------------------------------------------
# 홀리스 카페 블록맵 시각화
# (1) 데이터 불러오기
path = './지리정보시각화/data'
df = pd.read_csv(os.path.join(path,'홀리스카페_위도경도.csv')
                ,index_col=0)
df.head()
df.info()
#(2) 데이터 전처리
#(2-1) 주소 정보 : 시도, 군구 정보 분리
df['주소'] #서울특별시 서대문구 연세로 34 (창천동 31-12)  할리스

#x에는 주소 하나 들어간다
"서울특별시 중구 방배동".split() #공백을 기준으로 잘라서 List로반환
df['주소'].apply(lambda x:x.split()[:2]).to_list()

addr = pd.DataFrame(\
        df['주소'].apply(lambda x:x.split()[:2]).to_list(),
        columns=['시도', '군구'])
addr
addr['시도'].unique()
#시도 데이터 같은 자료 통일이 안되어있음
#ex) 서울특별시, 서울시, 서울 
#시도 값 -> 표준값으로 바꾸기(정규화)

#inplace 원본 데이터프레임에 바꾼값을 적용함
addr['시도'].replace('서울', '서울특별시', inplace=True)
addr['시도'].replace('서울시', '서울특별시', inplace=True)
addr['시도'].replace('대전', '대전광역시', inplace=True)
addr['시도'].replace('충북', '충청북도', inplace=True)
addr['시도'].replace('경북', '경상북도', inplace=True)
addr['시도'].replace('경기', '경기도', inplace=True)
addr['시도'].replace('울산', '울산광역시', inplace=True)
addr['시도'].replace('전북', '전라북도', inplace=True)
addr['시도'].replace('강원', '강원도', inplace=True)
addr['시도'].replace('대구시', '대구광역시', inplace=True)
addr['시도'].replace('대구', '대구광역시', inplace=True)
addr['시도'].replace('전남', '전라남도', inplace=True)
addr['시도'].replace('인천', '인천광역시', inplace=True)
addr['시도'].replace('세종', '세종특별자치시', inplace=True)
addr['시도'].replace('부산', '부산광역시', inplace=True)
addr['시도'].replace('경남', '경상남도', inplace=True)
addr['시도'].unique()

addr['군구'].unique()
# 금송로, 나성로, 특별자치시, 부산진구, 절재로
addr[addr['군구']=='금송로'] #31  세종특별자치시  금송로
addr.loc[31, '군구'] = '세종시'
addr[addr['군구']=='나성로']
addr.loc[159, '군구'] = '세종시'
addr[addr['군구']=='특별자치시']
addr.loc[315, '군구'] = '세종시'
addr[addr['군구']=='절재로']
addr.loc[394, '군구'] = '세종시'
addr['군구'].unique()
len(addr['시도'].unique())
#시도군구를 합친 데이터 만들기
addr['시도군구'] = addr.apply(lambda x:x['시도']+" "+x['군구'],
                    axis=1)
addr.head()
addr
#(2-2) 시도 군구별 매장 개수
addr['count'] = 0
addr_group = addr.groupby(['시도','군구','시도군구'], 
            as_index=False)['count'].count()
addr_group
#'시도군구' 칼럼을 인덱로 설정
addr_group.set_index('시도군구', inplace=True)
addr_group

#(2-3) 시도군구별 인구정보를 불러와서 주소정보와 합치기
pop = pd.read_excel(os.path.join(path, 
            '행정구역_시군구_별__성별_인구수_20211215091954.xlsx'))
#pop = pd.read_excel('./지리정보시각화/data/행정구역_시군구_별__성별_인구수_20211215091954.xlsx')
#pip install openpyxl
pop.head()
pop.info()
pop['군구'] #공백이 들어가있다.
pop['시도'] = pop['시도'].str.strip()
pop['군구'] = pop['군구'].str.strip()
pop.head()
pop['시도군구'] = pop.apply(lambda x:x['시도']+ " " + x['군구'],
                            axis=1)
pop =  pop[pop['군구'] != '소계']
pop = pop.set_index('시도군구')
pop.head()

#홀리스카페 주소 정보 + 인구 병합
#'시도군구' 정보는 인구 데이터가 많다 (혹은 같을 수 있다)
# 내부조인 방식으로 병합 : 홀리스카페 주소가 우리가 원하는 데이터+인구
#merge : 열방향으로 같은 인덱스끼리 데이터 결합
addr_pop = pd.merge(addr_group, pop, how='inner',
                left_index=True, right_index=True)
addr_pop
addr_pop = addr_pop[['시도_x','군구_x','총인구수','count']]
addr_pop
#칼럼이름 수정
addr_pop.columns = ['시도','군구','총인구수','count']
addr_pop.head()

#인구십만명당 매장의 개수의 값 구하기
addr_pop['ratio'] = addr_pop.apply(\
        lambda x:x['count']/x['총인구수']*100000, axis=1)
addr_pop.head()

#----------------------------------------------------
#3) 블록맵 지리 정보 불러와서 기존데이터와 결합
data_korea = pd.read_csv(os.path.join(path, 'data_draw_korea.csv'),
                    index_col=0)
data_korea.head()
data_korea['시도군구'] = data_korea.apply(\
            lambda x:x['광역시도']+ " " + x['행정구역'],axis=1)
data_korea.head()
#'시도군구'로 인덱스 설정
data_korea.set_index('시도군구', inplace=True)
data_korea.head()

#기존 매장정보 데이터와 블록맵 데이터 병합
#블록맵 데이터-> 전국 블록 위치 결정
#내부조인, 외부조인?
#data_korea 데이터가 사라지면? 
data_all = pd.merge(data_korea, addr_pop, how='outer',
        left_index=True, right_index=True)
data_all.head()
#강원도 고성군에는 홀리스 카페가 없어서, Nan으로 값이 들어가있다.
data_all[data_all['x'].isnull()]
#인천광역시 미추홀구는 블록맵 좌표가 없다.
#x,y(블록맵 위치좌표)가 없는 데이터 제거
data_all = data_all[data_all['x'].notnull()]
data_all.head()

draw_blockMap(data_all, 'count', 
            os.path.join(path, '행정구역별 홀리스카페 블록맵.png'),
            'Blues')
draw_blockMap(data_all, 'ratio', 
            os.path.join(path, 
            '행정구역별 인구수 백만명당 홀리스카페 블록맵.png'),
            'Reds')
data_all.head()
data_all['면적']
# 1040.07 *10^6 (m^2) 
#행정구역별 면적 10^7 (m^2) 당, 매장의 개수 시각화
data_all['면적당매장개수비율'] = data_all.apply(\
        lambda x:x['count']/x['면적']*10, axis=1)


draw_blockMap(data_all, '면적당매장개수비율', 
            os.path.join(path, 
            '행정구역별 면적당 홀리스카페 블록맵.png'),
            'Greens')
#11시 40분까지
data_all.sort_values(by=['면적당매장개수비율'], ascending=False)


#과제 
#전국 1등 로또 매장을 블록맵으로 시각화