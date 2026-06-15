from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import arxiv
import json
import threading

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# --- Session State ---
if "current_paper_id" not in st.session_state:
    st.session_state.current_paper_id = ""
if "paper_history" not in st.session_state:
    st.session_state.paper_history = []

st.header("Research Tool")


arxiv_input = st.text_input("Enter ArXiv URL or Paper ID", placeholder="e.g. https://arxiv.org/abs/1706.03762 or 1706.03762")

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

def extract_arxiv_id(raw: str) -> str:
    """Pulls the paper ID out of a URL or returns as-is."""
    raw = raw.strip()
    if "arxiv.org/abs/" in raw:
        return raw.split("arxiv.org/abs/")[-1].split("v")[0].strip("/")
    if "arxiv.org/pdf/" in raw:
        return raw.split("arxiv.org/pdf/")[-1].replace(".pdf", "").split("v")[0].strip("/")
    return raw  # assume it's already a bare ID


def fetch_paper(paper_id: str):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    return next(client.results(search))

# template
summary_template = PromptTemplate(
    template="""
You are summarizing the research paper titled "{paper_title}" based on its abstract and metadata.

Abstract:
{abstract}

Please provide a summary with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present.
   - Explain mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available, respond with "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=["paper_title", "abstract", "style_input", "length_input"],
validate_template=True
)


citation_template = PromptTemplate(
    template="""
Based on the research paper "{paper_title}" with this abstract:

{abstract}

List the 5 most important papers that this work builds upon or cites.
For each, provide the paper title and its ArXiv ID if it exists on ArXiv.

Respond ONLY with a valid JSON array, no markdown, no backticks, no explanation:
[
  {{"title": "Paper Title Here", "arxiv_id": "1234.56789", "reason": "one line why it matters"}},
  {{"title": "Paper Title Here", "arxiv_id": null, "reason": "one line why it matters"}}
]

Use null for arxiv_id if the paper is not on ArXiv (e.g. older books, non-CS papers).
""",
    input_variables=["paper_title", "abstract"],
)


@st.cache_data(show_spinner=False)
def fetch_paper_cached(paper_id: str):
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search))
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "published": paper.published.strftime("%B %Y"),
        "abstract": paper.summary,
    }

@st.cache_data(show_spinner=False)
def search_arxiv_by_title(title: str):
    """Fallback: search ArXiv by title when Gemini doesn't return an ID."""
    client = arxiv.Client()
    search = arxiv.Search(query=title, max_results=1)
    results = list(client.results(search))
    if results:
        r = results[0]
        return r.entry_id.split("/abs/")[-1].split("v")[0]
    return None

@st.cache_data(show_spinner=False)
def get_summary(paper_title, abstract, style_input, length_input):
    chain = summary_template | model
    result = chain.invoke({
        "paper_title": paper_title,
        "abstract": abstract,
        "style_input": style_input,
        "length_input": length_input,
    })
    return result.content

@st.cache_data(show_spinner=False)
def get_citations(paper_title, abstract):
    cite_chain = citation_template | model
    result = cite_chain.invoke({
        "paper_title": paper_title,
        "abstract": abstract,
    })
    raw = result.content.strip().replace("```json", "").replace("```", "")
    return json.loads(raw)

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
    st.divider()

    # Run summary + citations concurrently
    summary_result = [None]
    citations_result = [None]
    errors = []

    def run_summary():
        try:
            summary_result[0] = get_summary(paper['title'], paper['abstract'], style_input, length_input)
        except Exception as e:
            errors.append(f"Summary error: {e}")

    def run_citations():
        try:
            citations_result[0] = get_citations(paper['title'], paper['abstract'])
        except Exception as e:
            errors.append(f"Citation error: {e}")

    with st.spinner("Generating summary and extracting citations..."):
        t1 = threading.Thread(target=run_summary)
        t2 = threading.Thread(target=run_citations)
        t1.start(); t2.start()
        t1.join(); t2.join()

    if summary_result[0]:
        st.write(summary_result[0])
    st.divider()

    st.subheader(" Key Citations ")

    if citations_result[0]:
        # Fallback: search ArXiv by title for nulls
        for cite in citations_result[0]:
            if not cite.get("arxiv_id"):
                found_id = search_arxiv_by_title(cite["title"])
                if found_id:
                    cite["arxiv_id"] = found_id

        for cite in citations_result[0]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{cite['title']}**")
                st.caption(cite.get("reason", ""))
            with col2:
                if cite.get("arxiv_id"):
                    if st.button("Fetch →", key=f"cite_{cite['arxiv_id']}"):
                        st.session_state.paper_history.append(paper_id)
                        st.session_state.current_paper_id = cite["arxiv_id"]
                        st.rerun()
                else:
                    st.caption("Not on ArXiv")
    
    if errors:
        st.warning("\n".join(errors))

# --- Breadcrumb trail ---
if st.session_state.paper_history:
    st.markdown("**Your trail:** " + " → ".join(st.session_state.paper_history))
    if st.button("⬅ Go Back"):
        prev = st.session_state.paper_history.pop()
        st.session_state.current_paper_id = prev
        st.rerun()
    st.divider()

# --- Main trigger ---
if st.button("Summarize"):
    paper_id = extract_arxiv_id(arxiv_input)
    st.session_state.current_paper_id = paper_id
    st.session_state.paper_history = []
    show_paper(paper_id)

elif st.session_state.current_paper_id:
    show_paper(st.session_state.current_paper_id)