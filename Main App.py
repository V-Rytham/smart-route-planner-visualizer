import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import heapq
import math

st.set_page_config(layout="wide")
st.title("Dijkstra & A* Algorithm - Random Graph Visualization with Disruptions and User Preferences")

radius = 15
img_size = 800
num_nodes = 12

# === INITIALIZATION ===
if 'graph_created' not in st.session_state:
    st.session_state.graph_created = True
    st.session_state.vertices = []
    st.session_state.edges = []
    st.session_state.selected_edge = None
    st.session_state.source = None
    st.session_state.destination = None
    st.session_state.shortest_path = None
    st.session_state.shortest_cost = None
    st.session_state.strategy = "Fastest"
    st.session_state.blocked_edges = set()
    st.session_state.preferred_edges = []
    st.session_state.avoided_edges = []

    for _ in range(num_nodes):
        x = random.randint(50, img_size - 50)
        y = random.randint(50, img_size - 50)
        st.session_state.vertices.append((x, y))

    connected = set()
    for i in range(num_nodes):
        j = random.choice([x for x in range(num_nodes) if x != i])
        if (i, j) not in connected and (j, i) not in connected:
            st.session_state.edges.append({
                'nodes': (i, j),
                'distance': random.randint(10, 100),
                'time': random.randint(5, 60),
                'traffic': random.randint(1, 100),
                'risk': random.randint(1, 10),
                'toll': random.randint(0, 10),
                'blocked': False,
                'reason': ""
            })
            connected.add((i, j))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.3:
                if (i, j) not in connected and (j, i) not in connected:
                    st.session_state.edges.append({
                        'nodes': (i, j),
                        'distance': random.randint(10, 100),
                        'time': random.randint(5, 60),
                        'traffic': random.randint(1, 100),
                        'risk': random.randint(1, 10),
                        'toll': random.randint(0, 10),
                        'blocked': False,
                        'reason': ""
                    })
                    connected.add((i, j))

strategy_weights = {
    "Fastest":    (1.0, 0.0, 0.5, 0.2),
    "Safest":     (0.5, 2.0, 0.3, 0.1),
    "Least Toll": (0.5, 0.5, 0.3, 3.0)
}

left, right = st.columns([3, 2])
image = Image.new('RGB', (img_size, img_size), color='white')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

def compute_edge_cost(edge, weights):
    α, β, γ, δ = weights
    if edge['blocked']:
        return float('inf')
    base_cost = α * edge['time'] + β * edge['risk'] + γ * edge['traffic'] + δ * edge['toll']

    label = f"{edge['nodes'][0]} - {edge['nodes'][1]}"
    if label in st.session_state.preferred_edges:
        base_cost *= 0.8
    if label in st.session_state.avoided_edges:
        base_cost *= 1.5
    return base_cost

def heuristic(u, v):
    x1, y1 = st.session_state.vertices[u]
    x2, y2 = st.session_state.vertices[v]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

for edge in st.session_state.edges:
    i, j = edge['nodes']
    x1, y1 = st.session_state.vertices[i]
    x2, y2 = st.session_state.vertices[j]

    cost = compute_edge_cost(edge, strategy_weights[st.session_state.strategy])

    if edge['blocked']:
        color = 'red'
        width = 4
    else:
        if cost == float('inf'):
            color = 'gray'
            width = 2
        elif cost < 50:
            color = 'green'
            width = 3
        elif cost < 100:
            color = 'orange'
            width = 3
        else:
            color = 'brown'
            width = 3

        label = f"{edge['nodes'][0]} - {edge['nodes'][1]}"
        if label in st.session_state.preferred_edges:
            color = 'blue'
        if label in st.session_state.avoided_edges:
            color = 'gray'

    draw.line((x1, y1, x2, y2), fill=color, width=width)
    mx, my = (x1 + x2) // 2 + 10, (y1 + y2) // 2 + 10
    label = f"T:{edge['time']} R:{edge['risk']} Tr:{edge['traffic']} Tl:{edge['toll']}"
    if edge['blocked']:
        label += f" ⚠️{edge['reason']}"
    draw.text((mx, my), label, fill='black', font=font)

for idx, (x, y) in enumerate(st.session_state.vertices):
    color = 'blue'
    if idx == st.session_state.source:
        color = 'green'
    elif idx == st.session_state.destination:
        color = 'red'
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline='black')
    draw.text((x - 5, y - 5), str(idx), fill='white', font=font)

if st.session_state.shortest_path:
    path = st.session_state.shortest_path
    for k in range(len(path) - 1):
        x1, y1 = st.session_state.vertices[path[k]]
        x2, y2 = st.session_state.vertices[path[k + 1]]
        draw.line((x1, y1, x2, y2), fill='cyan', width=6)

left.image(image, caption="Graph with vertices, edges, weights, disruptions, and preferences", use_column_width=True)

st.sidebar.markdown("## Select Strategy")
st.session_state.strategy = st.sidebar.selectbox("Strategy", list(strategy_weights.keys()))

st.sidebar.markdown("## Source and Destination")
cols = st.sidebar.columns(2)
st.session_state.source = cols[0].selectbox("Source", list(range(len(st.session_state.vertices))), key="src")
st.session_state.destination = cols[1].selectbox("Destination", list(range(len(st.session_state.vertices))), key="dst")

heuristic_enabled = st.sidebar.checkbox("Enable Heuristic (Use A*)")

st.sidebar.markdown("## Preferred / Avoid Edges")
all_labels = [f"{e['nodes'][0]} - {e['nodes'][1]}" for e in st.session_state.edges]
st.session_state.preferred_edges = st.sidebar.multiselect("Preferred Edges", all_labels, default=st.session_state.preferred_edges)
st.session_state.avoided_edges = st.sidebar.multiselect("Avoid Edges", all_labels, default=st.session_state.avoided_edges)

st.sidebar.markdown("## Edit Edges")
selected_label = st.sidebar.selectbox("Select Edge", all_labels)
selected_index = all_labels.index(selected_label)
edge = st.session_state.edges[selected_index]

distance = st.sidebar.slider("Distance", 1, 100, edge['distance'])
time = st.sidebar.slider("Time", 1, 100, edge['time'])
traffic = st.sidebar.slider("Traffic Intensity", 1, 100, edge['traffic'])
risk = st.sidebar.slider("Risk", 1, 10, edge['risk'])
toll = st.sidebar.slider("Toll", 0, 10, edge['toll'])
blocked = st.sidebar.checkbox("Blocked", value=edge['blocked'])
reason = edge['reason'] if edge['blocked'] else ""

if blocked and not edge['blocked']:
    reason = st.sidebar.text_input("Reason for blocking", value="Flooding")
elif not blocked:
    reason = ""

edge['distance'] = distance
edge['time'] = time
edge['traffic'] = traffic
edge['risk'] = risk
edge['toll'] = toll
edge['blocked'] = blocked
edge['reason'] = reason

if st.sidebar.button("Simulate Disaster (Random Blocking)"):
    for e in st.session_state.edges:
        e['blocked'] = False
        e['reason'] = ""
    st.session_state.blocked_edges.clear()
    candidates = [e for e in st.session_state.edges if not e['blocked']]
    to_block = random.sample(candidates, k=min(3, len(candidates)))
    reasons = ["Flooding", "Accident", "Roadwork", "Landslide"]
    for e in to_block:
        e['blocked'] = True
        e['reason'] = random.choice(reasons)
        st.session_state.blocked_edges.add(e['nodes'])
    st.experimental_rerun()

st.sidebar.markdown("## Run Algorithm")

def dijkstra(graph, src, dst):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        cost, u = heapq.heappop(heap)
        if u == dst:
            break
        if cost > dist[u]:
            continue
        for neighbor, weight in graph[u]:
            if dist[u] + weight < dist[neighbor]:
                dist[neighbor] = dist[u] + weight
                prev[neighbor] = u
                heapq.heappush(heap, (dist[neighbor], neighbor))

    path = []
    node = dst
    if dist[dst] == float('inf'):
        return None, float('inf')
    while node is not None:
        path.insert(0, node)
        node = prev[node]
    return path, dist[dst]

def a_star(graph, src, dst):
    open_set = [(0, src)]
    dist = {node: float('inf') for node in graph}
    dist[src] = 0
    prev = {node: None for node in graph}

    while open_set:
        cost, u = heapq.heappop(open_set)
        if u == dst:
            break
        if cost > dist[u]:
            continue
        for neighbor, weight in graph[u]:
            tentative = dist[u] + weight + heuristic(neighbor, dst)
            if tentative < dist[neighbor]:
                dist[neighbor] = tentative
                prev[neighbor] = u
                heapq.heappush(open_set, (tentative, neighbor))

    path = []
    node = dst
    if dist[dst] == float('inf'):
        return None, float('inf')
    while node is not None:
        path.insert(0, node)
        node = prev[node]

    cost_sum = 0
    for k in range(len(path) - 1):
        for e in st.session_state.edges:
            if set(e['nodes']) == set((path[k], path[k+1])):
                cost_sum += compute_edge_cost(e, strategy_weights[st.session_state.strategy])
                break
    return path, cost_sum

if st.sidebar.button("Run"):
    if st.session_state.source == st.session_state.destination:
        st.warning("Source and destination cannot be the same.")
    else:
        graph = {i: [] for i in range(num_nodes)}
        for e in st.session_state.edges:
            cost = compute_edge_cost(e, strategy_weights[st.session_state.strategy])
            if cost == float('inf'):
                continue
            i, j = e['nodes']
            graph[i].append((j, cost))
            graph[j].append((i, cost))

        if heuristic_enabled:
            path, cost = a_star(graph, st.session_state.source, st.session_state.destination)
        else:
            path, cost = dijkstra(graph, st.session_state.source, st.session_state.destination)

        if path is None:
            st.session_state.shortest_path = None
            st.session_state.shortest_cost = None
            st.error("No path found due to blocked edges or graph disconnection.")
        else:
            st.session_state.shortest_path = path
            st.session_state.shortest_cost = cost

if st.session_state.shortest_path:
    right.markdown(f"### Shortest Path (Cost: {st.session_state.shortest_cost:.2f})")
    right.write(" → ".join(map(str, st.session_state.shortest_path)))

right.markdown("---")
right.markdown("**Legend:**")
right.markdown("""
- **Blue nodes:** Normal nodes  
- **Green node:** Source  
- **Red node:** Destination  
- **Edges:**  
  - Green: Low cost  
  - Orange: Medium cost  
  - Brown: High cost  
  - Red (thick): Blocked edges (due to disasters)  
  - **Blue:** Preferred  
  - **Gray:** Avoided  
- Cyan thick path: Computed shortest path  
""")