import streamlit as st
from dotenv import load_dotenv

from src.agent_core import app as research_agent

load_dotenv()

st.set_page_config(
    page_title="Academic Research Co-Pilot",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Academic Research Co-Pilot")
st.info(
    "Enter a research topic below. The AI agent will search for relevant academic papers, "
    "read the most relevant one, and generate a concise report based on its findings."
)

query = st.text_input(
    "**Your Research Question:**",
    placeholder="e.g., What are the latest advancements in retrieval-augmented generation?"
)

if st.button("Generate Report", type="primary"):
    if query:
        # Use st.status to show the agent's progress
        with st.status("🤖 The agent is on the case...", expanded=True) as status:
            final_report = ""
            # A dictionary to map node names to user-friendly status messages
            status_map = {
                "search_papers": "Searching for relevant academic papers...",
                "select_best_paper": "Analyzing search results to select the best paper...",
                "process_paper": "Downloading, processing, and reading the selected paper...",
                "generate_report": "Synthesizing the findings into a final report...",
            }

            try:
                # Stream the agent's execution
                for chunk in research_agent.stream({"question": query}):
                    for node_name, output in chunk.items():
                        # Update the status message for the current step
                        status.update(label=status_map.get(node_name, "Working..."))
                        
                        # The final report is in the output of the 'generate_report' node
                        if node_name == "generate_report":
                            final_report = output.get("report", "No report was generated.")
                
                # Update the status to "complete" and collapse the box
                status.update(label="Report generated successfully!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="An error occurred.", state="error")

        # Display the final report outside the status box
        if final_report:
            st.markdown(final_report)

    else:
        st.warning("Please enter a research question first.")