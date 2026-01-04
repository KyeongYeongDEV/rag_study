FROM docker.elastic.co/elasticsearch/elasticsearch:8.12.0

# 한국어 형태소 분석기(Nori) 플러그인 설치
RUN elasticsearch-plugin install analysis-nori