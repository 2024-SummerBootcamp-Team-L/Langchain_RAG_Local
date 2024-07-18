from opensearchpy import OpenSearch


client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "A769778aa!"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# 인덱스 이름 설정 (소문자로)
index_name = 'langchain_rag_test'


# 질문을 통해 OpenSearch에서 문서 검색
def search_documents(query):
    search_body = {
        "query": {
            "match": {
                "content": query
            }
        }
    }
    response = client.search(index=index_name, body={"query": {"match_all": {}}})
    hits = response['hits']['hits']
    return [hit['_source']['text'] for hit in hits]


# 메인 로직
if __name__ == "__main__":
    user_question = "꽁꽁고양이 밈 알려줘"

    # OpenSearch에서 관련 문서 검색
    search_results = search_documents(user_question)
    print(search_results)
    print("\n")

    # 검색된 문서를 하나의 컨텍스트로 합침
    context = " ".join(search_results)
    print(context)
