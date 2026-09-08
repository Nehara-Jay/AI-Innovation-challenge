from openai import OpenAI
import streamlit as st
import requests


st.title("💬 Chatbot")

#initia;ize chat history state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

#render the past messages on tehs creen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#chat input bar at the bottom
if prompt := st.chat_input():

    # 1. Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Trigger the Backend API
    with st.chat_message("assistant"):
        with st.spinner("Searching the archives and evaluating clues..."):
            try:
                # HTTP POST request to your FastAPI server
                response = requests.post(
                    "http://localhost:8000/api/ask", 
                    json={"query": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 3. Render the Reasoning Trace (For the judges)
                    with st.expander("🔍 View AI Reasoning Trace (Track 1B)"):
                        for step in data["reasoning_trace"]:
                            st.markdown(f"**Step {step['step_number']} ({step['action']}):** {step['thought']}")
                    
                    # 4. Render the Final Answer
                    st.markdown(f"### Answer\n{data['answer']}")
                    
                    # 5. Render Citations
                    st.divider()
                    st.markdown("**Citations:**")
                    for cite in data["citations"]:
                        st.caption(f"📖 *{cite['document_name']} (Page {cite['page_number']})*")
                        st.text(f"\"{cite['snippet']}\"")
                    
                    # Save the AI's answer to state
                    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                
                else:
                    st.error(f"Backend Error: {response.status_code}")
            
            except Exception as e:
                st.error("Failed to connect to the backend server. Is FastAPI running?")

   