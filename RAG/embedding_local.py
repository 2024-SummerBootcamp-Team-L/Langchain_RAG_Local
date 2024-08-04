from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_community.document_loaders import DirectoryLoader
import torch
from transformers import AutoTokenizer, AutoModel

# .env 파일 로드
# OpenSearch 클러스터에 연결
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'A769778aa!'),  # 인증 정보 필요 시
    use_ssl=True,
    verify_certs=True,
    ssl_show_warn=False,
)

class MyEmbeddingModel:
    def __init__(self, model_name):
        # tokenizer와 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, doc):
        # 문서를 임베딩하는 로직 구현
        inputs = self.tokenizer(doc, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 문서의 임베딩을 얻기 위해 마지막 hidden state의 평균을 사용
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def embed_query(self, text):
        # 쿼리를 임베딩하는 로직 구현
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 쿼리의 임베딩을 얻기 위해 마지막 hidden state의 평균을 사용
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding


# 인덱스 구조 설정
index_body = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "nori_user_dict": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "user_dictionary": "user_dic.txt"
                }
            },
            "analyzer": {
                "korean_analyzer": {
                    "filter": [
                        "synonym", "lowercase"
                    ],
                    "tokenizer": "nori_user_dict"
                }
            },
            "filter": {
                "synonym": {
                    "type": "synonym_graph",
                    "synonyms_path": "synonyms.txt"
                }
            }
        }
    }
}

# metadata 추출 및 부여 sys
def create_metadata(docs):
    # add a custom metadata field, such as timestamp
    for idx, doc in enumerate(docs):
        doc.metadata["category"] = ""
        doc.metadata["path"] = "test/test.txt"


embed_model_name = "BM-K/KoSimCSE-roberta-multitask"


# metadata 만들기

path = "."
loader = DirectoryLoader(path, glob="*.txt", show_progress=True)
docs = loader.load()
create_metadata(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", "."],
    length_function=len,
)

documents = text_splitter.split_documents(docs)
print(documents)

index_name = 'langchain_rag_test'

# MyEmbeddingModel의 인스턴스를 생성
my_embedding = MyEmbeddingModel(embed_model_name)

vector_db = OpenSearchVectorSearch.from_documents(
    index_name=index_name,
    body=index_body,
    documents=documents,
    embedding=my_embedding,
    op_type="create",
    opensearch_url="https://localhost:9200",
    http_auth=("admin", "A769778aa!"),
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    bulk_size=1000000,
    timeout=360000
)


vector_db.add_documents(documents, bulk_size=1000000)