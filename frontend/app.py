import os
from pathlib import Path
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="The Orpheus| AI Assistant",
    page_icon="📜",
    layout="centered",
)

# Bypasses Streamlit's CSS filter to force background and text motion
components.html(
    """
    <script>
    const style = parent.document.createElement('style');
    style.innerHTML = `
        @keyframes ambientLight {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes floatUp {
            0% { opacity: 0; transform: translateY(40px); }
            100% { opacity: 1; transform: translateY(0px); }
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #090412, #182042, #140d2b, #33102c) !important;
            background-size: 400% 400% !important;
            animation: ambientLight 12s ease-in-out infinite !important;
        }
        .hero-title {
            animation: floatUp 1.2s cubic-bezier(0.2, 0.8, 0.2, 1) forwards !important;
        }
        .hero-subtitle {
            opacity: 0; /* Starts hidden */
            animation: floatUp 1.2s cubic-bezier(0.2, 0.8, 0.2, 1) forwards !important;
            animation-delay: 0.3s !important;
        }
    `;
    parent.document.head.appendChild(style);
    </script>
    """,
    height=0,
    width=0
)


# --- Custom CSS for  Styling ---
st.markdown("""
<style>




[data-testid="stHeader"] {
    background: transparent !important;
    
}


/* 2. Fixed Top-Left Header  */
.top-left-header {
    position: fixed;
    top: 22px;
    left: 45px;
    z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
}
.header-tag {
    color: #a1a1aa;
    font-weight: 400;
    font-size: 1rem;
}

div[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
}

/*cht row spacing*/
div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: 24px !important;
}


/*right alined user bubble*/
div.user-msg {
    background-color:#2a2a2a !important;
    color: #f1f1f1 !important;
    padding: 12px 20px !important;
    border-radius: 24px !important ;
    max-width: fit-content;
    margin-left: auto;
    font-size: 1.05rem;
    line-height: 1.5;
}

/* 5. Assistant message bubble left aligned */
div.assistant-msg {
    background-color: transparent !important;
    padding: 10px 0px !important;
    color: #e3e3e3 !important;
    border: none !important; 
    max-width: 100%;
    font-size: 1.05rem;
    line-height: 1.6;
}

div[data-testid="stStatusWidget"],
div[data-testid="stStatusWidget"] summary,
div[data-testid="stStatusWidget"] div[data-testid="stExpander"]  {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}


/* Restrict image size */
.stImage img {
    max-width: 250px !important;
    border-radius: 8px;
    cursor: zoom-in;
}

/*  Citation Pills */
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

/* 9. Push content down slightly so it doesn't hit the new fixed header */
.block-container {
    padding-top: 4rem !important;
    background: transparent !important;
}

/* 10. Chat Input Box Adjustments (Shift Up & Make Larger) */
div[data-testid="stChatInput"] {
    padding-bottom: 45px !important; /* Pushes the input box upward */
}
div[data-testid="stChatInput"]
.stChatInputContainer {
    padding: 18px 24px !important; /* Makes the box taller */
    border-radius: 20px !important; /* Smooth rounding for the larger box */
}

div[data-testid="stChatInput"] textarea {
    font-size: 1.25rem !important; /* Makes the placeholder text and typed text larger */
}

div[data-testid="stChatInput"] button {
    transform: scale(1.2); /* Enlarges the send arrow to match the new box size */
    margin-right: 5px;
}

/* 11. MASSIVE Cinematic Welcome Typography */


.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 50vh;
    text-align: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
div.hero-title {
    font-size: 3rem !important; /* Extremely large */
    font-weight: 700 !important; /* Ultra Black weight */
    color: #ffffff !important;
    line-height: 1.05 !important; /* Tight vertical spacing */
    letter-spacing: -0.05em !important; /* Tight horizontal spacing */
    margin-bottom: 1rem !important;
}


</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class="top-left-header">
Orpheus <span class="header-tag"></span>
    </div>
    """,
    unsafe_allow_html=True
)


#st.caption("Intelligent Multi-Hop RAG, Visual Plate Analysis & Multi-Perspective Reasoning")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def render_message_payload(msg):
    # 1. Minimal reasoning trace expander
    if msg.get("reasoning_trace"):
        with st.expander("🔍 Searching for facts...",expanded=False):
            for step in msg["reasoning_trace"]:
                st.markdown(f"**Step {step['step_number']} (`{step['sub_query']}`):**\n{step['thought']}")

    # 2. Main answer text wrapped in a bubble CSS
    st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    # 3. Relevant images / plates (Restricted to 250px by CSS, click to expand)
    if msg.get("images"):
        st.markdown("</br>##### 🖼️ Visual Archive Plates")
        cols = st.columns(min(len(msg["images"]), 3))
        for idx, img_path in enumerate(msg["images"]):
            p = Path(img_path)
            if p.exists():
                with cols[idx % len(cols)]:
                    # Removed use_container_width=True to allow CSS sizing
                    st.image(str(p), caption=f"Plate: {p.stem}")

    # 4. Citations as sleek pills
    if msg.get("citations"):
        st.markdown("<br/>", unsafe_allow_html=True)
        for cite in msg["citations"]:
            doc_badge = "📜 Official Codex" if cite.get("type") == "document" else "✉️ Ephemera / Rumor"
            pill_html = f"""
            <div style="margin-bottom: 12px;">
                <strong>{cite.get('source', 'Unknown')}</strong> 
                <span class='cite-pill'>Page {cite.get('page', 1)}</span>
                <span class='cite-pill'>{doc_badge}</span><br/>
                <span style="font-size: 0.85rem; color: #a0a0a0;margin-left: 5px; ">"{cite.get('excerpt', '')}"</span>
            </div>
            """
            st.markdown(pill_html, unsafe_allow_html=True)

# Render past conversation history
#for msg in st.session_state.messages:
   # with st.chat_message(msg["role"]):
        #if msg["role"] == "assistant":
           # render_message_payload(msg)
        #else:
            #st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Ask Orpheus about the Ashen Era ")

# If chat is empty AND the user hasn't just hit enter, show the centered welcome screen
if not st.session_state.messages and not prompt:
    st.markdown(
        '''
       <div class="hero-container">
            <div class="hero-title">Ask Orpheus about the Ashen Era</div>
            
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    # Render past conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_message_payload(msg)
            else:
                st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)



    #process the new chat input
if prompt:
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


def get_image_display_target(img_ref: str) -> str:
    """Resolves an image reference to a valid local path or backend HTTP URL."""
    p = Path(img_ref)
    if p.exists():
        return str(p)
    local_in_images = Path("data/images") / p.name
    if local_in_images.exists():
        return str(local_in_images)
    return f"{API_BASE_URL.rstrip('/')}/images/{p.name}"


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
            target = get_image_display_target(img_path)
            plate_stem = Path(img_path).stem
            with cols[idx % len(cols)]:
                st.image(target, caption=f"Plate: {plate_stem}", use_container_width=True)

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
