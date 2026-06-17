import os
import json
import threading
import arxiv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from prompts import (
    summary_template, citation_template, compare_template,
    timeline_template, podcast_template
)

def get_models_in_order():
    models = []
    models.append(("Gemini 2.5 Flash", ChatGoogleGenerativeAI(model="gemini-2.5-flash")))
    models.append(("Groq LLaMA 3.3", ChatGroq(model="llama-3.3-70b-versatile")))
    models.append(("HuggingFace Mistral", HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=1024
    )))
    return models

MODELS = get_models_in_order()

def invoke_with_fallback(template, inputs: dict) -> str:
    last_error = None
    for model_name, model in MODELS:
        try:
            chain = template | model
            result = chain.invoke(inputs)
            content = result.content
            if isinstance(content, list):
                content = " ".join(
                    block["text"] if isinstance(block, dict) else str(block)
                    for block in content
                    if not isinstance(block, dict) or block.get("type") == "text"
                )
            st.caption(f"Response from: {model_name}")
            return content
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")

def extract_arxiv_id(raw: str) -> str:
    raw = raw.strip()
    if "arxiv.org/abs/" in raw:
        return raw.split("arxiv.org/abs/")[-1].split("v")[0].strip("/")
    if "arxiv.org/pdf/" in raw:
        return raw.split("arxiv.org/pdf/")[-1].replace(".pdf", "").split("v")[0].strip("/")
    return raw

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
    import fitz
    import tempfile

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        urllib.request.urlretrieve(pdf_url, f.name)
        tmp_path = f.name

    doc = fitz.open(tmp_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    os.unlink(tmp_path)
    return text[:15000]

@st.cache_data(show_spinner=False)
def search_arxiv_by_title(title: str):
    client = arxiv.Client()
    search = arxiv.Search(query=title, max_results=1)
    results = list(client.results(search))
    if results:
        return results[0].entry_id.split("/abs/")[-1].split("v")[0]
    return None

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

@st.cache_data(show_spinner=False)
def get_comparison(title1, abstract1, title2, abstract2, style_input):
    return invoke_with_fallback(compare_template, {
        "title1": title1,
        "abstract1": abstract1,
        "title2": title2,
        "abstract2": abstract2,
        "style_input": style_input,
    })

@st.cache_data(show_spinner=False)
def get_timeline(paper_title, abstract, published, style_input):
    return invoke_with_fallback(timeline_template, {
        "paper_title": paper_title,
        "abstract": abstract,
        "published": published,
        "style_input": style_input,
    })

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

def export_as_markdown(paper, summary, citations, timeline, paper_id):
    authors = ', '.join(paper['authors'][:5])
    citations_md = ""
    if citations:
        for cite in citations:
            link = f"https://arxiv.org/abs/{cite['arxiv_id']}" if cite.get('arxiv_id') else "Not on ArXiv"
            citations_md += f"- **{cite['title']}** — {cite.get('reason', '')} [{link}]\n"
    return f"""# {paper['title']}

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

def export_as_pdf(paper, summary, citations, timeline, paper_id):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    title_style = ParagraphStyle("title", fontSize=15, fontName="Helvetica-Bold", spaceAfter=6)
    heading_style = ParagraphStyle("heading", fontSize=12, fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10)
    body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica", spaceAfter=3, leading=14)

    def clean(text):
        if not text:
            return ""
        return