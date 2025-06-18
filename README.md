# Smart Route Planner Visualizer 🚦

A dynamic visualization tool that uses **Dijkstra**, **A\***, and **Uniform Cost Search (UCS)** algorithms to simulate personalized pathfinding across a randomly generated graph — based on **user preferences** and **real-world constraints** like tolls, traffic, and risk.

## 💡 Problem Statement

Different users often prefer different types of routes even when starting and ending points are the same:
- Person A wants the **shortest path**
- Person B avoids **toll roads**
- Person C prefers **low-risk paths** (e.g., avoiding disaster zones)
- Person D just wants to reach the destination

This tool computes optimal paths tailored to each of these users — all on a randomly generated graph.

---

## 🔧 Tech Stack

- **Python** – Core logic and algorithm implementations  
- **Streamlit** – UI and backend interaction  
- **Algorithms Used:**  
  - Dijkstra  
  - A\* (with heuristics toggle)  
  - Uniform Cost Search (UCS)

---

## 🖼️ Features

- 🔁 Random Graph Generator  
- ✏️ Editable edge attributes (Distance, Time, Toll, Risk, Traffic)  
- ⚠️ Disaster Simulation – Randomly blocks edges  
- 🎯 User-defined Preferences – Avoid or prefer specific edge types  
- 🧠 Strategy Selection – Fastest, safest, cheapest, or default  
- 📊 Real-time visualization of computed paths

---

## 📷 Screenshots

![Screenshot](./7bc8f60f-9fe8-4f31-9c09-f0e3b0f836fa.png)

---

## 🏃 How to Run

```bash
git clone https://github.com/<your-username>/smart-route-planner-visualizer.git
cd smart-route-planner-visualizer
pip install -r requirements.txt
streamlit run app.py
