import networkx as nx
import matplotlib.pyplot as plt

# 지식 그래프 생성 
# Directed Graph 생성
G = nx.DiGraph()

# [주어 -> 관계 -> 목적어] 형태의 데이터 정의
# 실제 GraphRAG에서는 LLM이 텍스트를 읽고 이 관계를 자동으로 추출
triplets = [
    ("Steve Jobs", "founded", "Apple"),
    ("Steve Jobs", "founded", "Pixar"),
    ("Steve Jobs", "created", "iPhone"),
    ("Tim Cook", "CEO_of", "Apple"),
    ("Pixar", "acquired_by", "Disney"),
    ("Disney", "owns", "Mickey Mouse"),
    ("Elon Musk", "CEO_of", "Tesla"), # 관련 없는 데이터 추가
]

print("그래프에 노드와 엣지 추가")
for subject, predicate, obj in triplets:
    G.add_edge(subject, obj, relation=predicate)
    print(f"   + [Node] {subject} --({predicate})--> [Node] {obj}")

# 관계 추론 함수 (Multi-hop Reasoning)
def find_relation(start_node, end_node):
    print(f"질문: '{start_node}'와 '{end_node}'는 어떻게 연결되어 있나?")
    
    try:
        # 두 노드 사이의 최단 경로 찾기 
        path = nx.shortest_path(G, source=start_node, target=end_node)
        
        # 경로를 따라가며 관계 설명하기
        explanation = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            relation = G[u][v]['relation']
            explanation.append(f"{u} --({relation})--> {v}")
            
        print(f"✅ (총 {len(explanation)}단계 연결)")
        print(" -> ".join(explanation))
        
    except nx.NetworkXNoPath:
        print("❌ 두 대상 사이의 연결 고리를 찾을 수 없습니다.")
    except nx.NodeNotFound:
        print("❌ 그래프에 해당 데이터가 없습니다.")

# Case 1: 직접적인 관계 (벡터 검색도 잘함)
find_relation("Steve Jobs", "iPhone")

# Case 2: 다리 건너 관계 (GraphRAG의 핵심)
# 잡스 -> 픽사 -> 디즈니 -> 미키 마우스 예상
find_relation("Steve Jobs", "Mickey Mouse")

# Case 3: 관계 없음
find_relation("Steve Jobs", "Tesla")

# 그래프 시각화 - 팝업 창으로 그래프 모양을 보여줍니다.
print("그래프 시각화 창을 띄웁니다.")
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42) 
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, font_weight='bold', arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("My First Knowledge Graph")
plt.show()