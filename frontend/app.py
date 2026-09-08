import os
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(
    page_title="The Ashen Era Archive | AI Assistant",
    page_icon="📜",
    layout="wide",
)

st.title("📜 The Ashen Era Archive")
st.caption("Intelligent Multi-Hop RAG, Visual Plate Analysis & Multi-Perspective Reasoning")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome, Archivist. Inquire about the lore, artifacts, factions, or visual plates of the Ashen Era.",
            "reasoning_trace": [],
            "citations": [],
            "images": [],
        }
    ]


def render_message_payload(msg):
    # 1. Reasoning trace expander
    if msg.get("reasoning_trace"):
        with st.expander("🔍 View AI Reasoning Trace (Track 1B)"):
            for step in msg["reasoning_trace"]:
                st.markdown(f"**Step {step['step_number']} (`{step['sub_query']}`):**\n{step['thought']}")

    # 2. Main answer text
    st.markdown(msg["content"])

    # 3. Relevant images / plates
    if msg.get("images"):
        st.markdown("##### 🖼️ Visual Archive Plates (Track 1A)")
        cols = st.columns(min(len(msg["images"]), 3))
        for idx, img_path in enumerate(msg["images"]):
            p = Path(img_path)
            if p.exists():
                with cols[idx % len(cols)]:
                    st.image(str(p), caption=f"Plate: {p.stem}", use_container_width=True)

    # 4. Citations
    if msg.get("citations"):
        st.divider()
        st.markdown("**Source Citations:**")
        for cite in msg["citations"]:
            doc_badge = "📜 Official Codex" if cite.get("type") == "document" else "✉️ Ephemera / Rumor"
            with st.container():
                st.markdown(f"**[{cite['citation_id']}]** *{cite['source']}* (Page {cite.get('page', 1)}) — `{doc_badge}`")
                st.caption(f"\"{cite['excerpt']}\"")


# Render past conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_payload(msg)
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the Ashen Era lore, heraldry, or historical records..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Query the Backend API
    with st.chat_message("assistant"):
        with st.spinner("Traversing archive codices, ephemera, and visual plates..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/ask",
                    json={"question": prompt, "max_hops": 2, "top_k_per_hop": 5},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    msg_obj = {
                        "role": "assistant",
                        "content": data.get("answer", ""),
                        "reasoning_trace": data.get("reasoning_trace", []),
                        "citations": data.get("citations", []),
                        "images": data.get("images", []),
                    }
                    render_message_payload(msg_obj)
                    st.session_state.messages.append(msg_obj)

                else:
                    st.error(f"Backend Error (Status {response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Failed to communicate with the backend server: {e}")