"""
DescriptoAI — Shopify AI Product Description Generator
--------------------------------------------------------
A single-file Streamlit app that uses Google's Gemini API to generate
high-converting, e-commerce-ready product descriptions for Shopify
and dropshipping stores.

Design System: "Velvet & Lavender"
v2 Update:
- FIXED: input text / placeholder contrast (text boxes were unreadable)
- API key now hidden inside a sidebar expander
- 10 new SaaS-grade features: audience targeting, SEO keywords, word
  count control, formatting style, emoji toggle, competitor link context,
  matching social ad copy, copy-to-clipboard, session history, CSV export

Tech Stack: Python, Streamlit, google-generativeai (Gemini API)
"""

import csv
import io
import datetime as dt

import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DescriptoAI — Shopify Product Description Generator",
    page_icon="🛍️",
    layout="centered",
)


# =========================================================
# 2. SESSION STATE INIT
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {time, name, tone, result, social}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_social" not in st.session_state:
    st.session_state.last_social = None


# =========================================================
# 3. PROMPT ENGINEERING
# =========================================================
SYSTEM_PROMPT_TEMPLATE = """
You are an elite e-commerce copywriter and direct-response marketing expert.
You specialize in writing high-converting product descriptions for Shopify
and dropshipping stores. Your copy is known for being persuasive, scannable,
and optimized to turn casual browsers into buyers.

Write a complete product description using the details below.

PRODUCT NAME: {product_name}
KEY FEATURES / KEYWORDS: {key_features}
TONE OF VOICE: {tone}
TARGET AUDIENCE: {audience}
SEO KEYWORDS TO NATURALLY INCLUDE: {seo_keywords}
DESIRED LENGTH: {length_instruction}
OUTPUT LAYOUT STYLE: {format_style}
EMOJI USAGE: {emoji_instruction}
{competitor_block}

STRICT FORMATTING RULES:

1. A short, catchy, attention-grabbing HEADLINE using a level-2 Markdown
   heading (##). It must spark curiosity or desire — not just restate the
   product name.
2. A 2-3 sentence engaging INTRODUCTION paragraph right after the headline,
   written with the TARGET AUDIENCE specifically in mind.
3. Body content formatted according to the OUTPUT LAYOUT STYLE:
   - "Bullet Points" -> a bolded sub-heading "**Key Benefits:**" followed by
     4-6 short, punchy bullet points, each leading with the BENEFIT not the
     raw feature.
   - "Paragraph Style" -> 2-3 flowing persuasive paragraphs (no bullets).
   - "Storytelling Format" -> a short narrative/scenario showing someone
     using the product and benefiting from it, woven naturally into the copy.
4. A short closing paragraph (1-2 sentences) that builds urgency or trust.
5. A final, bolded CALL TO ACTION on its own line, formatted like:
   **👉 [Strong action-driving CTA sentence]**

TONE INSTRUCTIONS:
- Match the "{tone}" tone of voice precisely throughout the entire piece.
- Professional = polished, credible, confident, no slang.
- Persuasive = benefit-driven, emotional triggers, classic sales psychology.
- Casual = friendly, conversational, like talking to a friend, light slang ok.
- Urgent = scarcity, FOMO, time-sensitive language, high energy.

OTHER RULES:
- Naturally weave in the SEO keywords above; never just dump them in a list.
- Respect the EMOJI USAGE instruction strictly.
- Do not invent fake statistics, certifications, or claims not implied by
  the provided features.
- Do not include any explanations, notes, or text about what you are doing.
- Output ONLY the final Markdown product description, nothing else.
"""

SOCIAL_PROMPT_TEMPLATE = """
You are a senior paid-social copywriter. Based on the product description
below, write ONE short, scroll-stopping Facebook/Instagram ad caption.

PRODUCT DESCRIPTION:
{description}

TONE OF VOICE: {tone}
TARGET AUDIENCE: {audience}

RULES:
- Max 5 short lines.
- Hook in the very first line.
- End with a clear, punchy call to action.
- {emoji_instruction}
- Output ONLY the caption text, nothing else (no labels, no quotes).
"""


def length_instruction_from_slider(value: str) -> str:
    mapping = {
        "Short / Punchy": "Keep the ENTIRE description short and punchy — roughly 60-100 words total.",
        "Medium": "Keep the ENTIRE description medium length — roughly 120-180 words total.",
        "Long / Detailed": "Write a longer, detailed description — roughly 220-300 words total.",
    }
    return mapping.get(value, mapping["Medium"])


def build_prompt(product_name, key_features, tone, audience, seo_keywords,
                  length_choice, format_style, emojis_on, competitor_url) -> str:
    emoji_instruction = (
        "Use relevant emojis naturally throughout (especially in bullets/headlines)."
        if emojis_on else
        "Do NOT use any emojis anywhere in the output."
    )
    competitor_block = (
        f"COMPETITOR REFERENCE (for positioning/context only, do not copy or mention the URL): {competitor_url}"
        if competitor_url.strip() else ""
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        product_name=product_name.strip(),
        key_features=key_features.strip(),
        tone=tone,
        audience=audience,
        seo_keywords=seo_keywords.strip() if seo_keywords.strip() else "None provided",
        length_instruction=length_instruction_from_slider(length_choice),
        format_style=format_style,
        emoji_instruction=emoji_instruction,
        competitor_block=competitor_block,
    )


def build_social_prompt(description, tone, audience, emojis_on) -> str:
    emoji_instruction = (
        "Include a few relevant emojis."
        if emojis_on else
        "Do NOT use any emojis."
    )
    return SOCIAL_PROMPT_TEMPLATE.format(
        description=description,
        tone=tone,
        audience=audience,
        emoji_instruction=emoji_instruction,
    )


# =========================================================
# 4. GEMINI API CALL
# =========================================================
def call_gemini(api_key: str, prompt: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    if not response or not getattr(response, "text", None):
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise PermissionError(
                "This request was blocked by Gemini's safety filters. "
                "Try rephrasing the product details."
            )
        raise ValueError("The model returned an empty response. Please try again.")

    return response.text


# =========================================================
# 5. "VELVET & LAVENDER" DESIGN SYSTEM — GLOBAL CSS (v2: contrast-fixed)
# =========================================================
GLOBAL_CSS = """
<style>
:root{
    --velvet:#2a1240;
    --velvet-deep:#1a0e2e;
    --charcoal:#16121f;
    --lavender:#7a4fd6;
    --lavender-soft:#d8cdff;
    --lavender-glow:rgba(122,79,214,0.55);
    --glass-bg:rgba(255,255,255,0.05);
    --glass-border:rgba(216,205,255,0.18);
    --input-bg:#f5f2ff;      /* light, distinct from dark page bg */
    --input-text:#1a1a1a;    /* explicit dark text, fixes invisibility bug */
    --input-placeholder:#5b5468;
}

html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at 20% 0%, var(--velvet) 0%, var(--charcoal) 55%, var(--velvet-deep) 100%) !important;
    color:#f1ecff;
}
[data-testid="stHeader"]{ background:rgba(0,0,0,0); }
#MainMenu, footer{ visibility:hidden; }

.block-container{
    position:relative;
    z-index:2;
    background:var(--glass-bg);
    border:1px solid var(--glass-border);
    border-radius:22px;
    padding:2.4rem 2.2rem 2.6rem 2.2rem !important;
    box-shadow:0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

/* ---------- Shimmering 3D title ---------- */
.descripto-title{
    font-size:2.6rem;
    font-weight:800;
    text-align:center;
    margin-bottom:0.2rem;
    background:linear-gradient(100deg,#7a4fd6 0%, #cdb8ff 25%, #f3edff 50%, #cdb8ff 75%, #7a4fd6 100%);
    background-size:300% auto;
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    text-shadow:0 4px 18px rgba(185,163,255,0.35);
    animation:shimmer 5s linear infinite;
    letter-spacing:0.5px;
}
@keyframes shimmer{
    0%{ background-position:0% 50%; }
    100%{ background-position:300% 50%; }
}
.descripto-subtitle{
    text-align:center;
    color:var(--lavender-soft);
    opacity:0.85;
    margin-bottom:1.4rem;
    font-size:1.02rem;
}

/* ---------- CRITICAL FIX: Inputs / Textareas / Selects ----------
   Light, solid background + explicitly dark text so typed content is
   ALWAYS readable, independent of the dark page theme. */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input{
    background:var(--input-bg) !important;
    color:var(--input-text) !important;
    border:1px solid var(--glass-border) !important;
    border-radius:14px !important;
    box-shadow:inset 0 1px 4px rgba(0,0,0,0.12);
    transition:border-color .25s ease, box-shadow .25s ease;
    caret-color:var(--lavender);
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder{
    color:var(--input-placeholder) !important;
    opacity:1 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus{
    border:1px solid var(--lavender) !important;
    box-shadow:0 0 0 3px var(--lavender-glow) !important;
}

/* Selectbox: force the closed-box + dropdown text dark-on-light too */
.stSelectbox div[data-baseweb="select"] > div{
    background:var(--input-bg) !important;
    border:1px solid var(--glass-border) !important;
    border-radius:14px !important;
}
.stSelectbox div[data-baseweb="select"] *{
    color:var(--input-text) !important;
}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] *{
    color:var(--input-text) !important;
    background:var(--input-bg) !important;
}

/* Slider value/labels readable on dark bg */
.stSlider label, .stCheckbox label{ color:#e6defc !important; }
.stSlider [data-baseweb="slider"] div{ color:#f1ecff; }

label, .stMarkdown p, .stCaption, .stExpander summary{ color:#e6defc !important; }

/* ---------- Generate button ---------- */
.stButton > button{
    width:100%;
    border:none !important;
    border-radius:16px !important;
    padding:0.85rem 1rem !important;
    font-weight:700;
    font-size:1.05rem;
    letter-spacing:0.3px;
    color:#fff !important;
    background:linear-gradient(135deg, #5b2a86 0%, #8c5fd6 45%, #b9a3ff 100%) !important;
    box-shadow:0 6px 22px rgba(140,95,214,0.45), 0 0 0 1px rgba(216,205,255,0.25) inset;
    transition:transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover{
    transform:translateY(-2px) scale(1.01);
    box-shadow:0 10px 28px rgba(185,163,255,0.55), 0 0 0 1px rgba(216,205,255,0.35) inset;
}
.stButton > button:active{ transform:translateY(0px) scale(0.99); }

/* ---------- Download button ---------- */
.stDownloadButton > button{
    border-radius:14px !important;
    border:1px solid var(--glass-border) !important;
    background:rgba(185,163,255,0.12) !important;
    color:#f1ecff !important;
}

/* ---------- Output / result card ---------- */
.output-card{
    position:relative;
    margin-top:0.6rem;
    padding:1.6rem 1.8rem;
    border-radius:20px;
    background:rgba(255,255,255,0.045);
    border:1px solid rgba(216,205,255,0.25);
    box-shadow:0 0 28px rgba(185,163,255,0.28), 0 8px 30px rgba(0,0,0,0.4);
    animation:glowPulse 4s ease-in-out infinite;
}
@keyframes glowPulse{
    0%,100%{ box-shadow:0 0 22px rgba(185,163,255,0.22), 0 8px 30px rgba(0,0,0,0.4); }
    50%{ box-shadow:0 0 36px rgba(185,163,255,0.4), 0 8px 30px rgba(0,0,0,0.4); }
}
.output-card h2{ color:var(--lavender-soft) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg, rgba(42,18,64,0.92) 0%, rgba(20,14,32,0.96) 100%) !important;
    border-right:1px solid var(--glass-border);
}
[data-testid="stSidebar"] .block-container{
    background:transparent;
    box-shadow:none;
    backdrop-filter:none;
    border:none;
    padding-top:1.5rem !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1{
    color:var(--lavender-soft) !important;
}
.sidebar-divider{
    height:2px;
    width:100%;
    margin:1.1rem 0;
    background:linear-gradient(90deg, transparent, var(--lavender) 50%, transparent);
    opacity:0.8;
    border-radius:2px;
}
[data-testid="stSidebar"] a{
    color:var(--lavender-soft) !important;
    text-decoration:none;
    border-bottom:1px dashed rgba(216,205,255,0.5);
}
[data-testid="stSidebar"] a:hover{ color:#fff !important; }

/* Expander header styling */
.stExpander{
    border:1px solid var(--glass-border) !important;
    border-radius:14px !important;
    background:rgba(255,255,255,0.04) !important;
}

/* ---------- History card ---------- */
.history-card{
    padding:0.8rem 1rem;
    margin-bottom:0.6rem;
    border-radius:12px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(216,205,255,0.15);
}
.history-card .hist-title{ color:var(--lavender-soft); font-weight:700; }
.history-card .hist-meta{ color:#bdb2dd; font-size:0.8rem; }

/* ---------- Alerts / divider ---------- */
.stAlert{ border-radius:14px !important; border:1px solid var(--glass-border) !important; backdrop-filter:blur(8px); }
hr{ border-color:rgba(216,205,255,0.18) !important; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# =========================================================
# 6. 3D PARTICLE BACKGROUND (mouse-reactive, optimized canvas)
# =========================================================
PARTICLE_BG_JS = """
<script>
(function(){
    const parentDoc = window.parent.document;
    if (parentDoc.getElementById('descripto-particle-bg')) return;

    const canvas = parentDoc.createElement('canvas');
    canvas.id = 'descripto-particle-bg';
    canvas.style.position = 'fixed';
    canvas.style.top = 0;
    canvas.style.left = 0;
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.zIndex = '0';
    canvas.style.pointerEvents = 'none';
    parentDoc.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let W, H;
    function resize(){
        W = canvas.width = window.parent.innerWidth;
        H = canvas.height = window.parent.innerHeight;
    }
    resize();
    window.parent.addEventListener('resize', resize);

    const COLORS = ['rgba(185,163,255,', 'rgba(122,79,214,', 'rgba(216,205,255,'];
    const COUNT = Math.min(70, Math.floor((window.parent.innerWidth * window.parent.innerHeight) / 18000));
    const particles = [];
    for(let i=0;i<COUNT;i++){
        particles.push({
            x: Math.random()*W,
            y: Math.random()*H,
            z: Math.random()*1 + 0.3,
            vx: (Math.random()-0.5)*0.25,
            vy: (Math.random()-0.5)*0.25,
            r: Math.random()*2 + 0.6,
            c: COLORS[Math.floor(Math.random()*COLORS.length)]
        });
    }

    let mouseX = W/2, mouseY = H/2;
    window.parent.addEventListener('mousemove', function(e){
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    let visible = true;
    parentDoc.addEventListener('visibilitychange', function(){
        visible = !parentDoc.hidden;
    });

    function tick(){
        if(visible){
            ctx.clearRect(0,0,W,H);
            for(let i=0;i<particles.length;i++){
                const p = particles[i];

                const dx = (mouseX - p.x);
                const dy = (mouseY - p.y);
                const dist = Math.sqrt(dx*dx + dy*dy) + 0.001;
                const force = Math.min(40 / dist, 0.6);
                p.vx -= (dx/dist) * force * 0.02;
                p.vy -= (dy/dist) * force * 0.02;

                p.x += p.vx;
                p.y += p.vy;
                p.vx *= 0.985;
                p.vy *= 0.985;

                if(p.x < 0) p.x = W; if(p.x > W) p.x = 0;
                if(p.y < 0) p.y = H; if(p.y > H) p.y = 0;

                const alpha = 0.25 + p.z*0.35;
                ctx.beginPath();
                ctx.fillStyle = p.c + alpha + ')';
                ctx.shadowColor = 'rgba(185,163,255,0.6)';
                ctx.shadowBlur = 6;
                ctx.arc(p.x, p.y, p.r * p.z * 2, 0, Math.PI*2);
                ctx.fill();
            }
        }
        requestAnimationFrame(tick);
    }
    tick();
})();
</script>
"""
components.html(PARTICLE_BG_JS, height=0, width=0)


# =========================================================
# 7. LIGHTWEIGHT "LOTTIE-STYLE" ANIMATED ICON HELPERS (CSS/SVG)
# =========================================================
def holographic_book_icon():
    html = """
    <div style="display:flex;align-items:center;justify-content:center;height:64px;margin:-6px 0 6px 0;">
      <style>
        @keyframes bookSpin{0%{transform:rotateY(0deg);}100%{transform:rotateY(360deg);}}
        @keyframes penGlow{0%,100%{filter:drop-shadow(0 0 2px #b9a3ff);}50%{filter:drop-shadow(0 0 9px #d8cdff);}}
        .holo-wrap{perspective:300px;}
        .holo-book{animation:bookSpin 6s linear infinite;transform-style:preserve-3d;}
        .holo-pen{animation:penGlow 1.8s ease-in-out infinite;}
      </style>
      <div class="holo-wrap">
        <svg class="holo-book" width="46" height="46" viewBox="0 0 64 64">
          <defs>
            <linearGradient id="bookGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#7a4fd6"/>
              <stop offset="100%" stop-color="#d8cdff"/>
            </linearGradient>
          </defs>
          <rect x="10" y="14" width="44" height="36" rx="4" fill="url(#bookGrad)" opacity="0.85"/>
          <line x1="32" y1="14" x2="32" y2="50" stroke="#1a0e2e" stroke-width="1.5"/>
          <path class="holo-pen" d="M40 8 L46 14 L26 34 L20 36 L22 30 Z" fill="#f1ecff"/>
        </svg>
      </div>
    </div>
    """
    components.html(html, height=64)


def data_stream_icon():
    html = """
    <div style="height:60px;margin:-4px 0 6px 0;">
      <canvas id="dsCanvas" width="500" height="60" style="width:100%;height:60px;"></canvas>
      <script>
      (function(){
        const cv = document.getElementById('dsCanvas');
        const ctx = cv.getContext('2d');
        const pts = [];
        for(let i=0;i<26;i++){
            pts.push({x:Math.random()*500, y:30+Math.sin(i)*10, s:Math.random()*1.6+0.6, sp:Math.random()*0.7+0.3});
        }
        function draw(){
            ctx.clearRect(0,0,500,60);
            for(const p of pts){
                p.x += p.sp;
                if(p.x>500) p.x=-5;
                const y = 30 + Math.sin(p.x/40)*12;
                const grad = ctx.createRadialGradient(p.x,y,0,p.x,y,6);
                grad.addColorStop(0,'rgba(216,205,255,0.95)');
                grad.addColorStop(1,'rgba(122,79,214,0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(p.x,y,p.s*3,0,Math.PI*2);
                ctx.fill();
            }
            requestAnimationFrame(draw);
        }
        draw();
      })();
      </script>
    </div>
    """
    components.html(html, height=60)


def ai_processing_orb():
    html = """
    <div style="display:flex;align-items:center;justify-content:center;height:80px;">
      <style>
        @keyframes orbPulse{0%{transform:scale(0.85);opacity:0.6;}50%{transform:scale(1.15);opacity:1;}100%{transform:scale(0.85);opacity:0.6;}}
        @keyframes orbSpin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
        .orb-core{width:34px;height:34px;border-radius:50%;
            background:radial-gradient(circle at 35% 30%, #f1ecff, #b9a3ff 40%, #5b2a86 100%);
            animation:orbPulse 1.3s ease-in-out infinite;
            box-shadow:0 0 24px 6px rgba(185,163,255,0.65);}
        .orb-ring{position:absolute;width:60px;height:60px;border:2px dashed rgba(216,205,255,0.55);
            border-radius:50%;animation:orbSpin 4s linear infinite;}
      </style>
      <div style="position:relative;display:flex;align-items:center;justify-content:center;">
        <div class="orb-ring"></div>
        <div class="orb-core"></div>
      </div>
    </div>
    """
    components.html(html, height=80)


def particle_burst():
    html = """
    <div style="height:46px;">
      <canvas id="burstCanvas" width="500" height="46" style="width:100%;height:46px;"></canvas>
      <script>
      (function(){
        const cv = document.getElementById('burstCanvas');
        const ctx = cv.getContext('2d');
        const cx = 250, cy = 23;
        const parts = [];
        for(let i=0;i<40;i++){
            const ang = Math.random()*Math.PI*2;
            const speed = Math.random()*3+1;
            parts.push({x:cx,y:cy,vx:Math.cos(ang)*speed,vy:Math.sin(ang)*speed,life:1});
        }
        function draw(){
            ctx.clearRect(0,0,500,46);
            let alive = false;
            for(const p of parts){
                if(p.life<=0) continue;
                alive = true;
                p.x+=p.vx; p.y+=p.vy; p.life-=0.025;
                ctx.beginPath();
                ctx.fillStyle = 'rgba(216,205,255,'+Math.max(p.life,0)+')';
                ctx.arc(p.x,p.y,2.2,0,Math.PI*2);
                ctx.fill();
            }
            if(alive) requestAnimationFrame(draw);
        }
        draw();
      })();
      </script>
    </div>
    """
    components.html(html, height=46)


def copy_to_clipboard_button(text: str, key: str, label: str = "📋 Copy to Clipboard"):
    """Pure HTML/JS copy button — works without extra Streamlit components."""
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/")
    html = f"""
    <div style="margin-top:0.4rem;">
      <button id="copyBtn_{key}" style="
          width:100%; padding:0.6rem 1rem; border-radius:12px; border:1px solid rgba(216,205,255,0.35);
          background:rgba(185,163,255,0.14); color:#f1ecff; font-weight:600; cursor:pointer;
          font-family:inherit; font-size:0.95rem;">
        {label}
      </button>
      <script>
        const btn_{key} = document.getElementById('copyBtn_{key}');
        btn_{key}.addEventListener('click', function(){{
          const txt = `{safe_text}`;
          navigator.clipboard.writeText(txt).then(function(){{
              btn_{key}.innerText = '✅ Copied!';
              setTimeout(function(){{ btn_{key}.innerText = '{label}'; }}, 1800);
          }});
        }});
      </script>
    </div>
    """
    components.html(html, height=56)


# =========================================================
# 8. SIDEBAR — API KEY (hidden in expander) + SESSION HISTORY
# =========================================================
with st.sidebar:
    st.header("⚙️ DescriptoAI")

    with st.sidebar.expander("🔑 API Key Configuration", expanded=False):
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            placeholder="Paste your API key here...",
            help="Your key is only used for this session and is never stored or shared.",
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            "**Don't have a key?**\n\n"
            "Get one for free from [Google AI Studio](https://aistudio.google.com/app/apikey)."
        )
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.caption(
            "🔒 Your API key stays in your browser session only. "
            "It is never logged or sent anywhere except directly to Google's API."
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.subheader("🕓 Recent Descriptions")
    if not st.session_state.history:
        st.caption("Your last 5 generations will appear here.")
    else:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""<div class="history-card">
                        <div class="hist-title">{item['name']}</div>
                        <div class="hist-meta">{item['time']} · {item['tone']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# =========================================================
# 9. MAIN PAGE — HEADER
# =========================================================
st.markdown('<div class="descripto-title">🛍️ DescriptoAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="descripto-subtitle">Turn raw product details into '
    '<b>scroll-stopping, high-converting</b> product descriptions — '
    'ready to paste straight into your Shopify store.</div>',
    unsafe_allow_html=True,
)
st.divider()


# =========================================================
# 10. MAIN PAGE — CORE INPUTS
# =========================================================
holographic_book_icon()
product_name = st.text_input(
    "Product Name",
    placeholder="e.g. Aurora Glow LED Desk Lamp",
)

data_stream_icon()
key_features = st.text_area(
    "Key Features / Keywords",
    placeholder="e.g. wireless charging, 3 light modes, touch control, "
                "adjustable arm, eye-care lighting, USB-C powered",
    height=120,
)

col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox(
        "Tone of Voice",
        options=["Professional", "Persuasive", "Casual", "Urgent"],
        index=1,
    )
with col2:
    audience = st.selectbox(
        "🎯 Target Audience",
        options=[
            "General Shoppers", "Gen Z", "Millennials", "Tech Enthusiasts",
            "Parents", "Fitness & Wellness", "Luxury Buyers", "Budget Shoppers",
            "Pet Owners", "Gamers",
        ],
        index=0,
    )

st.markdown("#### 🛠️ Advanced Options")

seo_keywords = st.text_input(
    "🔍 SEO Keywords (comma-separated)",
    placeholder="e.g. eco-friendly desk lamp, LED reading light, USB-C lamp",
)

competitor_url = st.text_input(
    "🔗 Competitor Product URL (optional, for positioning context)",
    placeholder="https://competitor-store.com/products/example",
)

col3, col4 = st.columns(2)
with col3:
    length_choice = st.select_slider(
        "📏 Word Count",
        options=["Short / Punchy", "Medium", "Long / Detailed"],
        value="Medium",
    )
with col4:
    format_style = st.selectbox(
        "🧱 Formatting Style",
        options=["Bullet Points", "Paragraph Style", "Storytelling Format"],
        index=0,
    )

col5, col6 = st.columns(2)
with col5:
    emojis_on = st.checkbox("😀 Include Emojis", value=True)
with col6:
    generate_social = st.checkbox("📱 Also generate Facebook/Instagram ad caption", value=True)

generate_clicked = st.button("✨ Generate Description", type="primary", use_container_width=True)


# =========================================================
# 11. GENERATION LOGIC + ERROR HANDLING
# =========================================================
if generate_clicked:
    particle_burst()

    if not api_key.strip():
        st.warning("⚠️ Please enter your Gemini API Key (open '🔑 API Key Configuration' in the sidebar).")
    elif not product_name.strip():
        st.warning("⚠️ Please enter a Product Name.")
    elif not key_features.strip():
        st.warning("⚠️ Please enter at least a few Key Features or Keywords.")
    else:
        ai_processing_orb()
        with st.spinner("Crafting your product description with AI... ✍️"):
            try:
                prompt = build_prompt(
                    product_name, key_features, tone, audience, seo_keywords,
                    length_choice, format_style, emojis_on, competitor_url,
                )
                result = call_gemini(api_key.strip(), prompt)

                social_copy = None
                if generate_social:
                    social_prompt = build_social_prompt(result, tone, audience, emojis_on)
                    social_copy = call_gemini(api_key.strip(), social_prompt)

                st.session_state.last_result = result
                st.session_state.last_social = social_copy
                st.session_state.history.append({
                    "time": dt.datetime.now().strftime("%H:%M:%S"),
                    "name": product_name.strip(),
                    "tone": tone,
                    "result": result,
                    "social": social_copy,
                })
                st.session_state.history = st.session_state.history[-5:]

                st.success("✅ Description generated successfully!")

            except PermissionError as e:
                st.error(f"🚫 {e}")
                st.session_state.last_result = None

            except Exception as e:
                error_text = str(e).lower()
                if "api_key" in error_text or "api key not valid" in error_text or "permission" in error_text:
                    st.error("🔑 Invalid Gemini API Key. Please double-check the key in the sidebar and try again.")
                elif "quota" in error_text or "rate limit" in error_text or "429" in error_text:
                    st.error("⏳ You've hit the API rate limit or quota. Please wait a moment and try again.")
                elif "network" in error_text or "connection" in error_text or "timeout" in error_text:
                    st.error("🌐 A network error occurred while contacting the Gemini API. Please check your connection.")
                elif "404" in error_text or "not found" in error_text:
                    st.error("❌ The AI model could not be found. It may have been retired.")
                else:
                    st.error(f"❌ Something went wrong while generating the description. Details: {e}")
                st.session_state.last_result = None


# =========================================================
# 12. OUTPUT DISPLAY (description + social copy + export tools)
# =========================================================
if st.session_state.last_result:
    result = st.session_state.last_result
    social_copy = st.session_state.last_social

    st.divider()
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    st.markdown(result)
    st.markdown('</div>', unsafe_allow_html=True)

    copy_to_clipboard_button(result, key="desc", label="📋 Copy Description")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📥 Download as .txt",
            data=result,
            file_name=f"{product_name.strip().replace(' ', '_').lower() or 'description'}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl_col2:
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["product_name", "tone", "audience", "description"])
        writer.writerow([product_name.strip(), tone, audience, result])
        st.download_button(
            label="📊 Export as .csv",
            data=csv_buffer.getvalue(),
            file_name=f"{product_name.strip().replace(' ', '_').lower() or 'description'}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if social_copy:
        st.markdown("#### 📱 Matching Social Ad Caption")
        st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown(social_copy)
        st.markdown('</div>', unsafe_allow_html=True)
        copy_to_clipboard_button(social_copy, key="social", label="📋 Copy Social Caption")


# =========================================================
# 13. FOOTER
# =========================================================
st.divider()
st.caption("Built with Python, Streamlit & Google Gemini API · DescriptoAI · Portfolio Project")
