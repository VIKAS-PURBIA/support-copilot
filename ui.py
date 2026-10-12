"""Streamlit chat UI.  Run: streamlit run ui.py   (API must be running on port 8000)"""
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="Support Copilot", page_icon="💬")
st.title("💬 Support Copilot")

role = st.sidebar.radio("I am a…", ["customer", "agent"])
show_trace = st.sidebar.checkbox("Show agent trace")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
st.session_state.setdefault("messages", [])


def render(m):
    st.markdown(m["content"])
    if m.get("status") == "escalated":
        t = m["ticket"]
        st.warning(f"Escalated to a human · {t['ticket_id']} · {t['priority']} priority · {t['category']}")
    if m.get("citations"):
        with st.expander("Sources"):
            for c in m["citations"]:
                st.markdown(f"**[{c['n']}] {c['title']}** (`{c['source']}`)\n\n> {c['snippet']}")
    if show_trace and m.get("trace"):
        with st.expander("Agent trace"):
            st.json(m["trace"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m)

if prompt := st.chat_input("Ask a support question"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(f"{API}/ask", timeout=600,
                                  json={"message": prompt, "role": role, "history": history})
                r.raise_for_status()
                data = r.json()
            except Exception as e:  # noqa: BLE001
                data = {"status": "error", "answer": f"Could not reach the API: {e}", "citations": [], "trace": []}
        msg = {"role": "assistant", "content": data["answer"], "status": data["status"],
               "citations": data.get("citations", []), "ticket": data.get("ticket"), "trace": data.get("trace", [])}
        render(msg)
    st.session_state.messages.append(msg)
