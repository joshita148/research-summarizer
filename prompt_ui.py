from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import arxiv
import json
import threading
import os

load_dotenv()
#model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


# --- Session State ---
if "arxiv_input_val" not in st.session_state:
    st.session_state.arxiv_input_val = ""
if "current_paper_id" not in st.session_state:
    st.session_state.current_paper_id = ""
if "paper_history" not in st.session_state:
    st.session_state.paper_history = []
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "podcast_script" not in st.session_state:
    st.session_state.podcast_script = ""
if "podcast_paper_id" not in st.session_state:
    st.session_state.podcast_paper_id = ""

def get_models_in_order():
    models = []

    models.append(("Gemini 2.5 Flash", ChatGoogleGenerativeAI(model="gemini-2.5-flash")))
    models.append(("Groq LLaMA 3.3", ChatGroq(model="llama-3.3-70b-versatile")))
    models.append(("HuggingFace Mistral", HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation", max_new_tokens=1024,)))
    return models


st.header("Research Tool")

mode = st.radio("Mode", ["Single Paper", "Compare Papers", "Podcast"], horizontal=True)

arxiv_input = st.text_input("Enter ArXiv URL or Paper ID", placeholder="e.g. https://arxiv.org/abs/1706.03762 or 1706.03762", value=st.session_state.arxiv_input_val)


if mode == "Compare Papers":
    arxiv_input2 = st.text_input("Enter Second ArXiv URL or Paper ID", placeholder="e.g. 1706.03762")

if mode != "Podcast":
    style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
else:
    style_input = "Beginner-Friendly"  # default, not shown

if mode == "Single Paper":
    length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

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
input_variables=["paper_title", "abstract", "style_input", "length_input"])


citation_template = PromptTemplate(
    template="""
Extract the 5 most important references actually cited in this research paper.

Paper Title: "{paper_title}"
Paper Text: {abstract}

Look for a References or Bibliography section at the end of the text.
Extract ONLY citations that appear in the text — do not invent or suggest any.

Respond ONLY with a valid JSON array, no markdown, no backticks, no explanation:
[
  {{"title": "Exact Paper Title", "arxiv_id": "1234.56789", "reason": "one line why it matters to this paper"}},
  {{"title": "Exact Paper Title", "arxiv_id": null, "reason": "one line why it matters to this paper"}}
]

Use null for arxiv_id if not an ArXiv paper.
""",
    input_variables=["paper_title", "abstract"],
)


compare_template = PromptTemplate(
    template="""
You are comparing two research papers.

Paper 1: "{title1}"
Abstract 1: {abstract1}

Paper 2: "{title2}"
Abstract 2: {abstract2}

Provide a structured comparison with:
1. **Core Contributions** — what each paper uniquely contributes
2. **Where They Agree** — shared ideas, assumptions, or findings
3. **Where They Conflict** — contradictions or opposing approaches
4. **Which to Read First** — and why

Explanation Style: {style_input}
""",
    input_variables=["title1", "abstract1", "title2", "abstract2", "style_input"],
)


timeline_template = PromptTemplate(
    template="""
You are a research historian. Analyze this paper:

Title: "{paper_title}"
Published: {published}
Abstract: {abstract}

Create a timeline of how this paper fits into its field.

Rules:
- Only include REAL papers you are confident exist
- Use the exact publication year provided above for "This Paper"
- For "Before": 3 real foundational papers that led to this work, with correct years
- For "After": 3 real papers published after {published} that this work influenced. Only use "Too recent to assess influence yet" if the paper was published in the in the current year. For older papers, you must find real follow-up work — do NOT invent papers, but do NOT give up early either.
- Never guess or hallucinate paper titles or years

Format:
**Before**
- YEAR — Paper Name — why it matters

**This Paper**
- YEAR — {paper_title} — what it introduced

**After**
- YEAR — Paper Name — why it matters
(or "Too recent to assess influence yet.")

Style: {style_input}
""",
    input_variables=["paper_title", "abstract", "published", "style_input"]
)

podcast_template = PromptTemplate(
    template="""
You are writing a podcast script between two hosts: Joe and Jane.
They are discussing the research paper "{paper_title}" in a fun, engaging, accessible way.

Abstract:
{abstract}

Rules:
- Start with Joe introducing the paper and why it matters
- Alternate between Joe and Jane naturally
- Cover: what problem it solves, key ideas, interesting findings, real world impact
- End with Jane giving a key takeaway for listeners
- Format each line as: "Joe: ..." or "Jane: ..."
- Keep it conversational, not academic
- Total length: about 20-30 exchanges
""",
    input_variables=["paper_title", "abstract"]
)

@st.cache_data(show_spinner=False)
def get_podcast_script(paper_title: str, abstract: str) -> str:
    return invoke_with_fallback(podcast_template, {
        "paper_title": paper_title,
        "abstract": abstract,
    })


def generate_podcast_audio(script: str) -> bytes:
    import wave
    import io
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = f"TTS the following conversation between Joe and Jane:\n\n{script}"

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Joe",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Jane",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                            )
                        )
                    ]
                )
            )
        )
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def get_timeline(paper_title, abstract, published, style_input):
    return invoke_with_fallback(timeline_template, {
        "paper_title": paper_title,
        "abstract": abstract,
        "published": published,
        "style_input": style_input,
    })

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
def fetch_full_text(paper_id: str) -> str:
    import urllib.request
    import fitz  # pymupdf
    import tempfile
    import os

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        urllib.request.urlretrieve(pdf_url, f.name)
        tmp_path = f.name

    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    os.unlink(tmp_path)

    # Return first 8000 chars — enough for references + body
    return text[:8000]

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


MODELS = get_models_in_order()

def invoke_with_fallback(template, inputs: dict) -> str:
    """Try each model in order, fall back if invoke fails."""
    last_error = None
    
    for model_name, model in MODELS:
        try:
            chain = template | model
            result = chain.invoke(inputs)
            
            # Handle content parsing
            content = result.content
            if isinstance(content, list):
                content = " ".join(
                    block["text"] if isinstance(block, dict) else str(block)
                    for block in content
                    if not isinstance(block, dict) or block.get("type") == "text"
                )
            
            # Show which model actually responded
            st.caption(f"Response from: {model_name}")
            return content
            
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(f" All models failed. Last error: {last_error}")

@st.cache_data(show_spinner=False)
def get_summary(paper_title, abstract, style_input, length_input):
    return invoke_with_fallback(summary_template, {
        "paper_title": paper_title,
        "abstract": abstract,
        "style_input": style_input,
        "length_input": length_input,
    })

@st.cache_data(show_spinner=False)
def get_citations(paper_title, abstract):
    raw = invoke_with_fallback(citation_template, {
        "paper_title": paper_title,
        "abstract": abstract,
    })
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def export_as_markdown(paper, summary, citations, timeline, paper_id):
    authors = ', '.join(paper['authors'][:5])
    
    citations_md = ""
    if citations:
        for cite in citations:
            link = f"https://arxiv.org/abs/{cite['arxiv_id']}" if cite.get('arxiv_id') else "Not on ArXiv"
            citations_md += f"- **{cite['title']}** — {cite.get('reason', '')} [{link}]\n"

    md = f"""# {paper['title']}

**Authors:** {authors}  
**Published:** {paper['published']}  
**ArXiv:** https://arxiv.org/abs/{paper_id}

---

## Summary
{summary}

---

## Key Citations
{citations_md}

---

## Timeline
{timeline}
"""
    return md


def export_as_pdf(paper, summary, citations, timeline, paper_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=15, fontName="Helvetica-Bold", spaceAfter=6)
    heading_style = ParagraphStyle("heading", fontSize=12, fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10)
    body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica", spaceAfter=3, leading=14)

    def clean(text):
        if not text:
            return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "").replace("#", "")

    story = []

    # Title + metadata
    story.append(Paragraph(clean(paper['title']), title_style))
    story.append(Paragraph(f"Authors: {clean(', '.join(paper['authors'][:5]))}", body_style))
    story.append(Paragraph(f"Published: {clean(str(paper['published']))}", body_style))
    story.append(Paragraph(f"ArXiv: https://arxiv.org/abs/{paper_id}", body_style))
    story.append(Spacer(1, 6*mm))

    # Summary
    story.append(Paragraph("Summary", heading_style))
    for line in clean(summary).split("\n"):
        if line.strip():
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 4*mm))

    # Citations
    if citations:
        story.append(Paragraph("Key Citations", heading_style))
        for cite in citations:
            line = f"- {cite['title']} — {cite.get('reason', '')}"
            if cite.get('arxiv_id'):
                line += f" [arxiv.org/abs/{cite['arxiv_id']}]"
            story.append(Paragraph(clean(line), body_style))
        story.append(Spacer(1, 4*mm))

    # Timeline
    if timeline:
        story.append(Paragraph("Timeline", heading_style))
        for line in clean(timeline).split("\n"):
            if line.strip():
                story.append(Paragraph(line, body_style))

    doc.build(story)
    return buffer.getvalue()

@st.cache_data(show_spinner=False)
def get_comparison(title1, abstract1, title2, abstract2, style_input):
    return invoke_with_fallback(compare_template, {
        "title1": title1,
        "abstract1": abstract1,
        "title2": title2,
        "abstract2": abstract2,
        "style_input": style_input,
    })

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

    # APA Citation
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
            full_text = paper['abstract']  # fallback to abstract

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
        st.session_state.last_summary = summary_result[0]  # store for export
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
    with st.spinner("Building timeline..."):
        try:
            timeline = get_timeline(paper['title'], full_text, paper['published'], style_input)
            st.markdown(timeline)
        except Exception as err:
            st.error(f"Timeline error: {err}")

    if summary_result[0]:
        md = export_as_markdown(paper, summary_result[0], citations_result[0], timeline or "", paper_id)
        pdf = export_as_pdf(paper, summary_result[0], citations_result[0], timeline or "", paper_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Markdown",
                data=md,
                file_name=f"{paper_id}_report.md",
                mime="text/markdown"
            )
        with col2:
            st.download_button(
                "Download PDF",
                data=pdf,
                file_name=f"{paper_id}_report.pdf",
                mime="application/pdf"
            )



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

        st.subheader("🎙️ Generate Podcast")
        if st.button("Generate Podcast Script + Audio"):
            papers_text = f"""
---
Title: {paper1['title']}
Summary: {paper1['abstract']}

---
Title: {paper2['title']}
Summary: {paper2['abstract']}
"""
            with st.spinner("Writing podcast script..."):
                script = get_podcast_script(papers_text)
            st.session_state.podcast_script = script
            st.rerun()

        if st.session_state.get("podcast_script"):
            st.subheader("Script")
            st.markdown(st.session_state.podcast_script)

            if st.button("Generate Audio"):
                with st.spinner("Generating audio... 2-3 minutes..."):
                    try:
                        audio_bytes = generate_podcast_audio(st.session_state.podcast_script)
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button(
                            "Download Podcast",
                            data=audio_bytes,
                            file_name="podcast.wav",
                            mime="audio/wav"
                        )
                    except Exception as e:
                        st.error(f"Audio failed: {e}")

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

        papers_text = f"""
---
Title: {paper['title']}
Abstract: {paper['abstract']}
"""
        with st.spinner("Writing podcast script..."):
                script = get_podcast_script(paper['title'], paper['abstract'])
        
        with st.spinner("Generating audio... 2-3 minutes..."):
            try:
                audio_bytes = generate_podcast_audio(script)
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "⬇️ Download Podcast",
                    data=audio_bytes,
                    file_name="podcast.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Audio failed: {e}")