import streamlit as st
from summarizer import Summarizer

# App Config
st.set_page_config(page_title="TextSummarizerPro", page_icon="📝", layout="wide")

st.title("📝 TextSummarizerPro")
st.subheader("AI-Powered Text Summarization using BART-Large (Hugging Face)")

# Initialize model
@st.cache_resource
def load_summarizer():
    return Summarizer(model_name="facebook/bart-large-cnn")

summarizer = load_summarizer()

# User Input
input_text = st.text_area("📄 Enter the text you want summarized:", height=250)

summary_length = st.radio(
    "Select Summary Length:",
    ("Short", "Medium", "Long")
)

length_map = {
    "Short": (60, 120),
    "Medium": (120, 200),
    "Long": (200, 300)
}

if st.button("✨ Summarize"):
    if input_text.strip():
        min_len, max_len = length_map[summary_length]

        with st.spinner("Generating summary…"):
            summary = summarizer.summarize(
                input_text,
                min_length=min_len,
                max_length=max_len
            )

        st.success("Summary generated!")
        st.write(summary)

        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please enter some text.")
