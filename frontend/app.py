import os
from pathlib import Path
import requests
import streamlit as st

st.set_page_config(
    page_title="The Orpheus| AI Assistant",
    page_icon="📜",
    layout="centered",
)

# --- Custom CSS for  Styling ---
st.markdown("""
<style>
/* Right-align user messages */
div[data-testid="stChatMessage"]:has(div.user-msg) {
    flex-direction: row-reverse;
}
div.user-msg {
    background-color: #2f2f2f;
    color: white;
    padding: 12px 18px;
    border-radius: 20px;
    max-width: fit-content;
    margin-left: auto;
    font-size: 1rem;
}

/* Restrict image size */
.stImage img {
    max-width: 250px !important;
    border-radius: 8px;
    cursor: zoom-in;
}

/* ChatGPT-style Citation Pills */
.cite-pill {
    display: inline-block;
    background-color: #333333;
    color: #cccccc;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 12px;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)



st.title("📜 Orpheus ")
st.caption("Intelligent Multi-Hop RAG, Visual Plate Analysis & Multi-Perspective Reasoning")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Ask Orpheus about the lore, artifacts, factions, or visual plates of the Ashen Era.",
            "reasoning_trace": [],
            "citations": [],
            "images": [],
        }
    ]


def render_message_payload(msg):
    # 1. Minimal reasoning trace expander
    if msg.get("reasoning_trace"):
        with st.expander("🔍 Searching for facts...",expanded=False):
            for step in msg["reasoning_trace"]:
                st.markdown(f"**Step {step['step_number']} (`{step['sub_query']}`):**\n{step['thought']}")

    # 2. Main answer text
    st.markdown(msg["content"])

    # 3. Relevant images / plates (Restricted to 250px by CSS, click to expand)
    if msg.get("images"):
        st.markdown("##### 🖼️ Visual Archive Plates")
        cols = st.columns(min(len(msg["images"]), 3))
        for idx, img_path in enumerate(msg["images"]):
            p = Path(img_path)
            if p.exists():
                with cols[idx % len(cols)]:
                    # Removed use_container_width=True to allow CSS sizing
                    st.image(str(p), caption=f"Plate: {p.stem}")

    # 4. Citations
    if msg.get("citations"):
        st.divider()
        for cite in msg["citations"]:
            doc_badge = "📜 Official Codex" if cite.get("type") == "document" else "✉️ Ephemera / Rumor"
            pill_html = f"""
            <div style="margin-bottom: 12px;">
                <strong>{cite.get('source', 'Unknown')}</strong> 
                <span class='cite-pill'>Page {cite.get('page', 1)}</span>
                <span class='cite-pill'>{doc_badge}</span><br/>
                <span style="font-size: 0.85rem; color: #a0a0a0;">"{cite.get('excerpt', '')}"</span>
            </div>
            """
            st.markdown(pill_html, unsafe_allow_html=True)

# Render past conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_payload(msg)
        else:
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask Orpheus about the Ashen Era "):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    # 2. Query the Backend API
    with st.chat_message("assistant"):
        msg_obj = {}

        #use st.status for the live "thinking" spinner
        with st.status("🔍 Searching for facts...",expanded=False) as status:
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
                    # Render the trace live inside the status box
                    for step in msg_obj["reasoning_trace"]:
                        st.markdown(f"**Step {step['step_number']} (`{step.get('sub_query', '')}`):**\n{step['thought']}")
                    
                    status.update(label="Interpretation Complete", state="complete", expanded=False)

                else:
                    status.update(label="Error", state="error", expanded=False)
                    st.error(f"Backend Error (Status {response.status_code}): {response.text}")

            except Exception as e:
                status.update(label="Connection Failed", state="error", expanded=False)
                st.error(f"Failed to communicate with the backend server: {e}")

        # Render the rest of the payload (Answer, Images, Citations)
        if msg_obj.get("content"):
            # Create a copy without reasoning_trace so it doesn't double-render the expander
            live_render_obj = msg_obj.copy()
            live_render_obj["reasoning_trace"] = []
            render_message_payload(live_render_obj)
            
            # Save the full object (with trace) to state for future rerenders
            st.session_state.messages.append(msg_obj)