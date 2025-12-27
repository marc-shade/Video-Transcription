"""
Task Presets for AI Persona
Provides one-click task buttons that generate structured outputs from transcriptions.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TaskPreset:
    """Defines a task preset with its configuration."""
    id: str
    name: str
    icon: str
    description: str
    prompt_template: str
    category: str


# Define all task presets
TASK_PRESETS: Dict[str, TaskPreset] = {
    "chapters": TaskPreset(
        id="chapters",
        name="Generate Chapters",
        icon="📋",
        description="Create timestamped chapter markers for the video",
        category="analysis",
        prompt_template="""Based on the following transcript, generate a list of chapter markers for this video.

TRANSCRIPT:
{transcript}

Please provide:
1. A list of chapters with timestamps (use the format [HH:MM:SS] if timestamps are available in the transcript, otherwise estimate based on content flow)
2. Each chapter should have a clear, descriptive title
3. Include a brief 1-sentence description of what each chapter covers

Format your response as:
## Video Chapters

| Timestamp | Chapter Title | Description |
|-----------|---------------|-------------|
| [00:00:00] | Introduction | Brief description... |
| [00:02:30] | Topic Name | Brief description... |

Include at least 5-10 chapters depending on content length."""
    ),

    "key_points": TaskPreset(
        id="key_points",
        name="Extract Key Points",
        icon="📊",
        description="Bullet-point summary with citations",
        category="analysis",
        prompt_template="""Analyze the following transcript and extract the key points and main takeaways.

TRANSCRIPT:
{transcript}

Please provide:
1. A concise summary (2-3 sentences)
2. 5-10 key points as bullet points
3. For each key point, include a citation reference to where in the transcript it's discussed (use timestamps if available)

Format your response as:
## Summary
[2-3 sentence overview]

## Key Points

- **Point 1**: [Key insight] - *Referenced at [timestamp/section]*
- **Point 2**: [Key insight] - *Referenced at [timestamp/section]*
...

## Main Takeaways
1. [Most important takeaway]
2. [Second most important]
3. [Third most important]"""
    ),

    "faqs": TaskPreset(
        id="faqs",
        name="Find FAQs",
        icon="💬",
        description="Questions the content answers",
        category="analysis",
        prompt_template="""Based on the following transcript, identify the questions that this content effectively answers.

TRANSCRIPT:
{transcript}

Please provide:
1. A list of 5-10 frequently asked questions that this video/content addresses
2. For each question, provide a concise answer based on the transcript
3. Include relevant timestamps or sections where the answer is discussed

Format your response as:
## FAQs from this Content

### Q1: [Question]
**Answer**: [Concise answer from the transcript]
*Discussed at: [timestamp/section]*

### Q2: [Question]
**Answer**: [Concise answer from the transcript]
*Discussed at: [timestamp/section]*

Continue for all identified questions..."""
    ),

    "clip_hooks": TaskPreset(
        id="clip_hooks",
        name="Create Clip Hooks",
        icon="✂️",
        description="10 short-form video hook suggestions with timestamps",
        category="repurposing",
        prompt_template="""Analyze the following transcript and identify the best moments for short-form video clips (TikTok, Reels, Shorts).

TRANSCRIPT:
{transcript}

Please provide:
1. 10 potential clip hooks - moments that would make engaging short-form content
2. For each, include the timestamp (if available) and a suggested hook/caption
3. Explain why each moment would work well as a short clip

Format your response as:
## Short-Form Clip Suggestions

### Clip 1: [Catchy Title]
- **Timestamp**: [00:00:00] to [00:00:30]
- **Hook**: "[Opening line or caption]"
- **Why it works**: [Brief explanation]

### Clip 2: [Catchy Title]
- **Timestamp**: [00:02:15] to [00:02:45]
- **Hook**: "[Opening line or caption]"
- **Why it works**: [Brief explanation]

Continue for all 10 clips...

## Top 3 Viral Potential
1. [Best clip for reach]
2. [Second best]
3. [Third best]"""
    ),

    "social_posts": TaskPreset(
        id="social_posts",
        name="Social Media Posts",
        icon="📱",
        description="Twitter/LinkedIn posts from key insights",
        category="repurposing",
        prompt_template="""Create social media posts based on the following transcript content.

TRANSCRIPT:
{transcript}

Please provide posts for multiple platforms:

## Twitter/X Thread (5-7 tweets)
Create a thread that summarizes the key insights. Each tweet should be under 280 characters.

## LinkedIn Post
Create a professional, longer-form post (1000-1300 characters) suitable for LinkedIn. Include:
- Attention-grabbing opening
- Key insights with bullet points
- Call to action
- Relevant hashtags

## Instagram Caption
Create an engaging caption (under 2200 characters) with:
- Hook in first line
- Value-packed middle
- Call to action
- Hashtags (10-15 relevant ones)

Format clearly with headers for each platform."""
    ),

    "blog_draft": TaskPreset(
        id="blog_draft",
        name="Blog Draft",
        icon="📝",
        description="Long-form article from transcript",
        category="repurposing",
        prompt_template="""Convert the following transcript into a well-structured blog article.

TRANSCRIPT:
{transcript}

Please create a complete blog post with:

## Blog Article

### Title
[Compelling, SEO-friendly title]

### Meta Description
[150-160 character description for search engines]

### Introduction
[2-3 paragraphs introducing the topic and hooking the reader]

### Main Content
[Organize the transcript content into logical sections with headers]
[Include subheadings (H2, H3) for better readability]
[Add bullet points and lists where appropriate]
[Include quotes from the original content where impactful]

### Key Takeaways
[Summarize main points as a bulleted list]

### Conclusion
[2-3 paragraphs wrapping up and providing a call to action]

### SEO Keywords
[List of 5-10 relevant keywords/phrases]

Target word count: 1500-2000 words"""
    ),

    "claims_evidence": TaskPreset(
        id="claims_evidence",
        name="Extract Claims",
        icon="🔍",
        description="Factual claims with supporting timestamps",
        category="research",
        prompt_template="""Analyze the following transcript and extract all factual claims and their supporting evidence.

TRANSCRIPT:
{transcript}

Please provide:

## Factual Claims Analysis

### Verified/Verifiable Claims
For each claim that can be fact-checked:
| Claim | Evidence/Source Given | Timestamp | Verification Status |
|-------|----------------------|-----------|---------------------|
| [Claim] | [Any evidence cited] | [Time] | Needs verification / Has source |

### Opinions vs Facts
Distinguish between:
- **Stated Facts**: Claims presented as objective truth
- **Expert Opinions**: Claims based on expertise/experience
- **Personal Views**: Subjective statements

### Statistics & Numbers
List all numerical claims:
- [Statistic 1] - Context: [where mentioned]
- [Statistic 2] - Context: [where mentioned]

### Sources Mentioned
List any sources, studies, or references cited in the content.

### Claims Requiring Fact-Check
Highlight claims that should be verified before republishing."""
    ),

    "quotes": TaskPreset(
        id="quotes",
        name="Find Quotes",
        icon="💎",
        description="Notable quotable moments",
        category="research",
        prompt_template="""Identify the most quotable and shareable moments from the following transcript.

TRANSCRIPT:
{transcript}

Please extract:

## Notable Quotes

### Top 10 Quotable Moments
For each quote:
1. **Quote**: "[Exact words from transcript]"
   - **Timestamp**: [00:00:00]
   - **Context**: [Brief context of when this was said]
   - **Best use**: [Social media / Article pull quote / Presentation slide]

### Categorized Quotes

#### Inspirational/Motivational
- "[Quote]" - [Timestamp]

#### Insightful/Educational
- "[Quote]" - [Timestamp]

#### Controversial/Bold
- "[Quote]" - [Timestamp]

#### Humorous/Memorable
- "[Quote]" - [Timestamp]

### Quote Graphics Suggestions
Top 3 quotes that would work well as visual quote graphics:
1. "[Quote]" - Suggested style: [Minimal/Bold/Professional]
2. "[Quote]" - Suggested style: [Minimal/Bold/Professional]
3. "[Quote]" - Suggested style: [Minimal/Bold/Professional]"""
    ),

    "topics": TaskPreset(
        id="topics",
        name="Identify Topics",
        icon="🏷️",
        description="Topic/keyword extraction",
        category="research",
        prompt_template="""Perform a comprehensive topic and keyword analysis on the following transcript.

TRANSCRIPT:
{transcript}

Please provide:

## Topic Analysis

### Primary Topics
1. [Main Topic 1] - Coverage: [Extensive/Moderate/Brief]
2. [Main Topic 2] - Coverage: [Extensive/Moderate/Brief]
3. [Main Topic 3] - Coverage: [Extensive/Moderate/Brief]

### Secondary Topics
- [Topic] - Mentioned at [timestamps]
- [Topic] - Mentioned at [timestamps]

### Keywords & Phrases

#### High-Frequency Keywords
| Keyword | Frequency | Context |
|---------|-----------|---------|
| [word] | [count] | [how it's used] |

#### Long-tail Keywords/Phrases
- "[phrase 1]"
- "[phrase 2]"

### Topic Timeline
How topics flow through the content:
- [00:00-05:00]: [Topics covered]
- [05:00-10:00]: [Topics covered]
- [Continue...]

### SEO Opportunities
Based on this content, target these keywords:
1. Primary keyword: [keyword]
2. Secondary keywords: [list]
3. Related searches: [list]

### Content Categorization
- **Industry/Niche**: [Category]
- **Content Type**: [Tutorial/Interview/Discussion/etc.]
- **Audience Level**: [Beginner/Intermediate/Advanced]
- **Suggested Tags**: [tag1, tag2, tag3, ...]"""
    ),
}


def get_task_presets_by_category() -> Dict[str, List[TaskPreset]]:
    """Group task presets by category for UI display."""
    categories = {}
    for preset in TASK_PRESETS.values():
        if preset.category not in categories:
            categories[preset.category] = []
        categories[preset.category].append(preset)
    return categories


def get_category_info() -> Dict[str, Dict[str, str]]:
    """Get display information for categories."""
    return {
        "analysis": {
            "name": "Content Analysis",
            "icon": "📋",
            "description": "Analyze and structure the content"
        },
        "repurposing": {
            "name": "Content Repurposing",
            "icon": "✂️",
            "description": "Transform content for different platforms"
        },
        "research": {
            "name": "Research & Extraction",
            "icon": "🔍",
            "description": "Deep analysis and data extraction"
        }
    }


def generate_task_prompt(preset_id: str, transcript: str) -> str:
    """
    Generate the full prompt for a task preset.

    Args:
        preset_id: The ID of the task preset
        transcript: The transcript text to analyze

    Returns:
        The formatted prompt ready to send to the AI
    """
    if preset_id not in TASK_PRESETS:
        raise ValueError(f"Unknown task preset: {preset_id}")

    preset = TASK_PRESETS[preset_id]
    return preset.prompt_template.format(transcript=transcript)


def export_content_to_markdown(content: str, task_name: str, filename: str = None) -> str:
    """
    Export generated content to markdown format.

    Args:
        content: The generated content
        task_name: Name of the task that generated this content
        filename: Optional source filename

    Returns:
        Formatted markdown string
    """
    header = f"# {task_name}\n\n"
    if filename:
        header += f"*Source: {filename}*\n\n"
    header += "---\n\n"
    return header + content
