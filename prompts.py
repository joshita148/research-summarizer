from langchain_core.prompts import PromptTemplate

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
    input_variables=["paper_title", "abstract", "style_input", "length_input"]
)

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
    input_variables=["paper_title", "abstract"]
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
    input_variables=["title1", "abstract1", "title2", "abstract2", "style_input"]
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
- For "After": 3 real papers published after {published} that this work influenced. Only use "Too recent to assess influence yet" if the paper was published in the current year. For older papers, you must find real follow-up work — do NOT invent papers, but do NOT give up early either.
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