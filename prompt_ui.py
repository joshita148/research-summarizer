from dotenv import load_dotenv
import streamlit as st
import threading
from functions import (
    extract_arxiv_id, fetch_paper_cached, fetch_full_text,
    search_arxiv_by_title, get_summary, get_citations,
    get_comparison, get_timeline, get_podcast_script,
    generate_podcast_audio, export_as_markdown, export_as_pdf
)

load_dotenv()

# --- Session State ---
if "arxiv_input_val" not in st.session_state:
    st.session_state.arxiv_input_val = ""
if "current_paper_id" not in st.session_state:
    st.session_state.current_paper_id = ""
if "paper_history" not in st.session_state:
    st.session_state.paper_history = []
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "comparison_paper1" not in st.session_state:
    st.session_state.comparison_paper1 = None
if "comparison_paper2" not in st.session_state:
    st.session_state.comparison_paper2 = None
if "podcast_script" not in st.session_state:
    st.session_state.podcast_script = ""
if "podcast_paper_id" not in st.session_state:
    st.session_state.podcast_paper_id = ""

st.header("Research Tool")

mode = st.radio("Mode", ["Single Paper", "Compare Papers", "Podcast"], horizontal=True)

arxiv_input = st.text_input(
    "Enter ArXiv URL or Paper ID",
    placeholder="e.g. https://arxiv.org/abs/1706.03762 or 1706.03762",
    value=st.session_state.arxiv_input_val
)

if mode == "Compare Papers":
    arxiv_input2 = st.text_input("Enter Second ArXiv URL or Paper ID", placeholder="e.g. 1706.03762")

if mode != "Podcast":
    style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
else:
    style_input = "Beginner-Friendly"

if mode == "Single Paper":
    length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

def show_paper(paper_id: str):
    with st.spinner("Fetching paper..."):
        try:
            paper = fetch_paper_cached(paper_id)
        except Exception as e:
            st.error(f"ArXiv fetch failed: {e}")
            return

    st.markdown(f"### {paper['title']}")
    st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}")
    st.markdown(f"**Published:** {paper['published']}")
    st.markdown(f"**ArXiv ID:** [{paper_id}](https://arxiv.org/abs/{paper_id})")

    authors = paper['authors']
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) <= 5:
        author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        author_str = ", ".join(authors[:5]) + ", et al."

    year = paper['published'].split(" ")[-1]
    apa = f"{author_str}. ({year}). {paper['title']}. *arXiv*. https://doi.org/10.48550/arXiv.{paper_id}"
    st.markdown("**APA Citation**")
    st.code(apa, language=None)
    st.divider()

    with st.spinner("Fetching full paper text..."):
        try:
            full_text = fetch_full_text(paper_id)
        except Exception:
            full_text = paper['abstract']

    summary_result = [None]
    citations_result = [None]

    def run_summary():
        try:
            summary_result[0] = get_summary(paper['title'], full_text, style_input, length_input)
        except Exception:
            pass

    def run_citations():
        try:
            citations_result[0] = get_citations(paper['title'], full_text)
        except Exception:
            pass

    with st.spinner("Generating summary and extracting citations..."):
        t1 = threading.Thread(target=run_summary)
        t2 = threading.Thread(target=run_citations)
        t1.start(); t2.start()
        t1.join(); t2.join()

    if summary_result[0]:
        st.write(summary_result[0])
    st.divider()

    st.subheader("Key Citations")
    if citations_result[0]:
        for cite in citations_result[0]:
            if not cite.get("arxiv_id"):
                found_id = search_arxiv_by_title(cite["title"])
                if found_id:
                    cite["arxiv_id"] = found_id

        for i, cite in enumerate(citations_result[0]):
            col1, col2 = st.columns([3, 1])
            with col1:
                if cite.get("arxiv_id"):
                    st.markdown(f"**[{cite['title']}](https://arxiv.org/abs/{cite['arxiv_id']})**")
                else:
                    st.markdown(f"**{cite['title']}**")
                st.caption(cite.get("reason", ""))
            with col2:
                if cite.get("arxiv_id"):
                    if st.button("Dive In →", key=f"cite_{i}_{cite['arxiv_id']}"):
                        st.session_state.arxiv_input_val = cite["arxiv_id"]
                        st.rerun()
                else:
                    st.caption("Not on ArXiv")

    st.divider()
    st.subheader("Timeline - Where This Paper Stands")
    timeline_text = None
    with st.spinner("Building timeline..."):
        try:
            timeline_text = get_timeline(paper['title'], full_text, paper['published'], style_input)
            st.markdown(timeline_text)
        except Exception as err:
            st.error(f"Timeline error: {err}")

    if summary_result[0]:
        md = export_as_markdown(paper, summary_result[0], citations_result[0], timeline_text or "", paper_id)
        pdf = export_as_pdf(paper, summary_result[0], citations_result[0], timeline_text or "", paper_id)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download Markdown", data=md, file_name=f"{paper_id}_report.md", mime="text/markdown")
        with col2:
            st.download_button("⬇️ Download PDF", data=pdf, file_name=f"{paper_id}_report.pdf", mime="application/pdf")


# --- Mode Blocks ---
if mode == "Single Paper":
    if extract_arxiv_id(arxiv_input) != st.session_state.current_paper_id:
        st.session_state.current_paper_id = ""

    if st.button("Summarize"):
        paper_id = extract_arxiv_id(arxiv_input)
        st.session_state.current_paper_id = paper_id
        st.session_state.paper_history = []
        st.rerun()

    if st.session_state.current_paper_id:
        show_paper(st.session_state.current_paper_id)

elif mode == "Compare Papers":
    if st.button("Compare"):
        id1 = extract_arxiv_id(arxiv_input)
        id2 = extract_arxiv_id(arxiv_input2)

        with st.spinner("Fetching both papers..."):
            try:
                paper1 = fetch_paper_cached(id1)
                paper2 = fetch_paper_cached(id2)
            except Exception as e:
                st.error(f"Fetch failed: {e}")
                st.stop()

        with st.spinner("Comparing papers..."):
            comparison = get_comparison(
                paper1['title'], paper1['abstract'],
                paper2['title'], paper2['abstract'],
                style_input
            )

        st.session_state.comparison_result = comparison
        st.session_state.comparison_paper1 = paper1
        st.session_state.comparison_paper2 = paper2
        st.rerun()

    if st.session_state.comparison_result:
        paper1 = st.session_state.comparison_paper1
        paper2 = st.session_state.comparison_paper2

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {paper1['title']}")
            st.caption(f"{paper1['published']} · {', '.join(paper1['authors'][:3])}")
        with col2:
            st.markdown(f"### {paper2['title']}")
            st.caption(f"{paper2['published']} · {', '.join(paper2['authors'][:3])}")

        st.divider()
        st.markdown(st.session_state.comparison_result)
        st.divider()

        
elif mode == "Podcast":
    if st.button("Generate Podcast"):
        paper_id = extract_arxiv_id(arxiv_input)

        with st.spinner("Fetching paper..."):
            try:
                paper = fetch_paper_cached(paper_id)
            except Exception as e:
                st.error(f"Fetch failed: {e}")
                st.stop()

        st.markdown(f"### {paper['title']}")
        st.caption(f"{paper['published']} · {', '.join(paper['authors'][:3])}")
        st.divider()

        with st.spinner("Writing podcast script..."):
            script = get_podcast_script(paper['title'], paper['abstract'])

        with st.spinner("Generating audio... 2-3 minutes..."):
            try:
                audio_bytes = generate_podcast_audio(script)
                st.audio(audio_bytes, format="audio/wav")
                st.download_button("Download Podcast", data=audio_bytes, file_name="podcast.wav", mime="audio/wav")
            except Exception as e:
                st.error(f"Audio failed: {e}")