import pandas as pd
import folium

hollys = pd.read_csv('./지리정보시각화/data/홀리스카페_위도경도.csv')

#(1) 지도생성
m = folium.Map(location=[hollys.loc[0,'위도'], hollys.loc[0,'경도']],
        zoom_start=7)

#(2) 마커클러스터 생성
from folium.plugins import MarkerCluster
m_cluster = MarkerCluster().add_to(m)

for idx, data in hollys.iterrows() : 
    if data['위도'] == 0 :
        continue
    else :
        folium.Marker(
            location=[data['위도'], data['경도']],
            popup=data['매장명'],
            icon=folium.Icon(icon='star', color='blue')
        ).add_to(m)
m.save('./지리정보시각화/data/hollys_map.html')

#17시에 다시 시작하겠습니다.

