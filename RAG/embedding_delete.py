from opensearchpy import OpenSearch
import os
from opensearchpy import OpenSearch

# OpenSearch 클라이언트 설정
opensearch_id = os.getenv('OPENSEARCH_ID')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
opensearch_url = os.getenv('OPENSEARCH_URL')

# 인증 정보 설정
opensearch_auth = (opensearch_id, opensearch_password)

# OpenSearch 클라이언트를 초기화
client = OpenSearch(hosts=[opensearch_url], http_auth=opensearch_auth)

index_name = 'law'
# 인덱스 삭제
client.indices.delete(index=index_name, ignore=[400, 404])