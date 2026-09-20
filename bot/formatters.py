import html
import re
from typing import Dict, List, Optional, Tuple
from core.models import NewsFactSheet, ReliabilityState
from core.synthesis.explainer import StructuredDigestItem
from core.synthesis.tutor import HowToGuide, VillageExplanation

RELIABILITY_BADGE_MAP: Dict[ReliabilityState, str] = {
    ReliabilityState.OFFICIAL: "🛡️ [Verified Official]",
    ReliabilityState.CROSS_CHECKED: "🔍 [Cross-Checked]",
    ReliabilityState.REPORTED: "📰 [Reported]",
    ReliabilityState.UNCERTAIN: "⚠️ [Uncertain / Disputed]",
}


def strip_html(text: str) -> str:
    """Remove raw HTML tags and decode HTML entities from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html.unescape(clean)
    return re.sub(r"\s+", " ", clean).strip()


def escape_telegram_markdown(text: str) -> str:
    """Escape special control characters for Telegram legacy Markdown (parse_mode='Markdown').

    Telegram legacy Markdown reserves: '_', '*', '`', '['
    We escape existing backslashes first, then escape control characters.
    Brackets '[' and ']' are escaped so arbitrary dynamic text cannot form
    unintended link entities or break entity parsing.
    """
    if not text:
        return ""
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("_", "\\_")
    escaped = escaped.replace("*", "\\*")
    escaped = escaped.replace("`", "\\`")
    escaped = escaped.replace("[", "\\[")
    escaped = escaped.replace("]", "\\]")
    return escaped


def sanitize_markdown_url(url: str) -> str:
    """Safely format URLs for use inside Markdown [anchor](url).
    
    Percent-encodes parentheses and spaces according to RFC 3986 so that
    unbalanced or query-string parentheses do not prematurely terminate
    the Markdown inline link syntax [text](url).
    """
    if not url:
        return "#"
    return (
        url.strip()
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )


def find_protected_spans(text: str) -> List[Tuple[int, int]]:
    """Identify character spans that must not be split (markdown links, code blocks)."""
    spans: List[Tuple[int, int]] = []
    
    # 1. Markdown links: [anchor](url)
    link_pattern = re.compile(r'\[[^\]]*\]\([^\)]*\)')
    for match in link_pattern.finditer(text):
        spans.append((match.start(), match.end()))
        
    # 2. Fenced code blocks: ```...```
    code_pattern = re.compile(r'```.*?```', re.DOTALL)
    for match in code_pattern.finditer(text):
        spans.append((match.start(), match.end()))
        
    # 3. Inline code: `...`
    inline_code_pattern = re.compile(r'`[^`\n]+`')
    for match in inline_code_pattern.finditer(text):
        spans.append((match.start(), match.end()))

    return sorted(spans, key=lambda x: x[0])


def is_inside_span(index: int, spans: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Return the (start, end) span if index falls strictly inside it."""
    for start, end in spans:
        if start < index < end:
            return (start, end)
    return None


def chunk_telegram_message(text: str, max_chars: int = 4000) -> List[str]:
    """Safely split text into chunks within Telegram's message size limit.
    
    Guarantees:
    1. Each chunk is <= max_chars.
    2. Does not split inside Markdown links [anchor](url) or code blocks.
    3. Prefers natural split boundaries (paragraphs, newlines, sentences, spaces).
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        protected_spans = find_protected_spans(remaining)

        # Look for natural split points before max_chars
        candidate = max_chars

        # Check if candidate falls inside a protected span
        inside = is_inside_span(candidate, protected_spans)
        if inside:
            candidate = inside[0]  # Backtrack to before the protected span

        # Search backward from candidate for preferred delimiters
        sub = remaining[:candidate]
        split_idx = -1

        # Priority 1: Double newline (paragraph)
        p_idx = sub.rfind("\n\n")
        if p_idx > candidate // 3:
            if not is_inside_span(p_idx, protected_spans):
                split_idx = p_idx + 2

        # Priority 2: Single newline
        if split_idx == -1:
            n_idx = sub.rfind("\n")
            if n_idx > candidate // 3:
                if not is_inside_span(n_idx, protected_spans):
                    split_idx = n_idx + 1

        # Priority 3: Sentence ending (period / question / exclamation followed by space)
        if split_idx == -1:
            match = None
            for m in re.finditer(r'([.?!])\s', sub):
                m_end = m.end()
                if not is_inside_span(m_end, protected_spans):
                    match = m_end
            if match and match > candidate // 3:
                split_idx = match

        # Priority 4: Whitespace
        if split_idx == -1:
            s_idx = sub.rfind(" ")
            if s_idx > candidate // 4:
                if not is_inside_span(s_idx, protected_spans):
                    split_idx = s_idx + 1

        # Fallback: candidate boundary before protected span
        if split_idx == -1 or split_idx <= 0:
            split_idx = candidate

        # Safety: ensure progress
        if split_idx <= 0:
            split_idx = max_chars

        chunk = remaining[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].strip()

    return chunks


def format_digest_message(
    items: List[StructuredDigestItem],
    fact_sheets: List[NewsFactSheet],
) -> str:
    """Format the morning digest into readable Telegram Markdown with clickable links and badges.
    
    Source URLs and reliability states are deterministically bound from fact_sheets.
    All dynamic article text is stripped of HTML and escaped for Telegram legacy Markdown.
    """
    fact_map = {fs.event_id: fs for fs in fact_sheets}
    
    lines = [
        "🌅 *AI Mitra Morning Intelligence Digest*",
        "Here are today's verified developments and practical tools:\n",
    ]

    for idx, item in enumerate(items, 1):
        fs = fact_map.get(item.event_id)
        badge = RELIABILITY_BADGE_MAP.get(fs.reliability, "📰 [Reported]") if fs else "📰 [Reported]"
        source_name = fs.primary_source_name if fs else "Original Source"
        source_url = fs.primary_source_url if fs else "#"

        safe_headline = escape_telegram_markdown(strip_html(item.headline))
        safe_summary = escape_telegram_markdown(strip_html(item.concise_summary))
        safe_takeaway = escape_telegram_markdown(strip_html(item.key_takeaway))
        safe_source_name = escape_telegram_markdown(strip_html(source_name))
        safe_source_url = sanitize_markdown_url(source_url)

        lines.append(f"*{idx}. {safe_headline}*")
        lines.append(f"{safe_summary}")
        lines.append(f"💡 *Takeaway:* {safe_takeaway}")
        # Deterministic clickable Markdown link and reliability badge
        lines.append(f"{badge} • [{safe_source_name}]({safe_source_url})\n")

    lines.append("────────────────────────")
    lines.append("💬 _Options:_")
    lines.append("• `/explain <number>` — Plain-English village analogy breakdown")
    lines.append("• `/howto <tool>` — Evidence-based practical quickstart")
    lines.append("• Or simply ask any AI doubt or paste a news link!")
    
    return "\n".join(lines)


def format_village_explanation(explanation: VillageExplanation) -> str:
    """Format a village-level explanation with clear separation between facts and analogy."""
    safe_topic = escape_telegram_markdown(strip_html(explanation.concept_or_topic))
    lines = [
        f"🌾 *AI Mitra Breakdown: {safe_topic}*\n",
        "📢 *Source Facts (What actually happened):*",
    ]
    for fact in explanation.source_facts:
        safe_fact = escape_telegram_markdown(strip_html(fact))
        lines.append(f"• {safe_fact}")

    safe_analogy = escape_telegram_markdown(strip_html(explanation.village_analogy))
    lines.append("\n🌾 *Everyday Village Analogy (How to think about it):*")
    lines.append(f"{safe_analogy}\n")

    safe_what = escape_telegram_markdown(strip_html(explanation.what_is_it))
    lines.append("🔍 *What It Is:*")
    lines.append(f"{safe_what}\n")

    safe_how = escape_telegram_markdown(strip_html(explanation.how_it_works))
    lines.append("⚙️ *How It Works:*")
    lines.append(f"{safe_how}\n")

    safe_why = escape_telegram_markdown(strip_html(explanation.why_it_matters))
    lines.append("🌟 *Why It Matters to You:*")
    lines.append(f"{safe_why}")

    return "\n".join(lines)


def format_howto_guide(guide: HowToGuide) -> str:
    """Format an evidence-based tool guide with documentation facts strictly separated from steps."""
    safe_name = escape_telegram_markdown(strip_html(guide.tool_name))
    safe_doc_url = sanitize_markdown_url(guide.official_documentation_url)
    lines = [
        f"🛠 *AI Mitra Quickstart: {safe_name}*\n",
        "📄 *Verified Documentation Facts:*",
        f"• Official Documentation: [{safe_name} Docs]({safe_doc_url})",
    ]

    if guide.is_free is not None:
        free_status = "Free / Open Source" if guide.is_free else "Paid / Subscription"
        lines.append(f"• Access: {free_status}")
    if guide.pricing_details:
        safe_pricing = escape_telegram_markdown(strip_html(guide.pricing_details))
        lines.append(f"• Pricing: {safe_pricing}")

    if guide.verified_prerequisites:
        lines.append("\n📋 *Prerequisites:*")
        for prereq in guide.verified_prerequisites:
            lines.append(f"• {escape_telegram_markdown(strip_html(prereq))}")

    if guide.verified_install_commands:
        lines.append("\n💻 *Verified Setup Commands:*")
        for cmd in guide.verified_install_commands:
            lines.append(f"```bash\n{cmd}\n```")

    if guide.step_by_step_usage:
        lines.append("\n🚀 *Step-by-Step Usage:*")
        for idx, step in enumerate(guide.step_by_step_usage, 1):
            lines.append(f"{idx}. {escape_telegram_markdown(strip_html(step))}")

    if guide.caveats_or_unverified:
        lines.append("\n⚠️ *Important Caveats / Unverified Details:*")
        for caveat in guide.caveats_or_unverified:
            lines.append(f"• {escape_telegram_markdown(strip_html(caveat))}")

    return "\n".join(lines)

