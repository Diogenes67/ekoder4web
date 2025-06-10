import json, re, io, string
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------
#  Streamlit page config MUST BE FIRST
# -----------------------------------------------------------------
st.set_page_config(
    page_title="EKoder Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",  # Changed to collapsed
)

# Now do other imports after page config is set
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the enhanced matcher
try:
    from enhanced_clinical_matcher import EnhancedClinicalMatcher
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# -----------------------------------------------------------------
#  Ensure cache-busting salt exists before anything else uses it
# -----------------------------------------------------------------
def get_tfidf_salt() -> int:
    """Return the current salt value, initialising it to 0 if absent."""
    if "tfidf_salt" not in st.session_state:
        st.session_state["tfidf_salt"] = 0
    return st.session_state["tfidf_salt"]
# -----------------------------------------------------------------

st.title("EKoder Pro – ED Diagnosis Coder")
st.markdown(
    """
<div style='background-color:#fff3cd; padding:12px; border-radius:6px;
            border:1px solid #ffe69c; margin-bottom:20px;'>
<b>Disclaimer:</b> <i>EKoder-4o</i> is a research and educational tool.  
It does <u>not</u> provide medical advice, and outputs must be reviewed by
qualified clinical staff. Use only with fully de-identified notes and in
compliance with your organisation's privacy and governance policies.
</div>
""",
    unsafe_allow_html=True,
)

# Show warning here if enhanced matcher not available
if not ENHANCED_AVAILABLE:
    st.warning("Enhanced Clinical Matcher not found. Using original TF-IDF only.")

# -----------------------------------------------------------------
#  Sidebar controls
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

# API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif "OPENAI_API_KEY" not in os.environ:
    st.error("⚠️ Please enter your OpenAI API key in the sidebar to use EKoder.")
    st.stop()

debug_mode = st.sidebar.checkbox("🪛 Verbose debug", value=False)  # Changed to False

# Enhanced ranking controls
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Ranking System")
use_enhanced_ranking = st.sidebar.checkbox(
    "Use Enhanced Ranking",
    value=True,  # Changed to True - ON by default
    disabled=not ENHANCED_AVAILABLE,
    help="New system with better medical term understanding"
)

if use_enhanced_ranking:
    ranking_method = st.sidebar.selectbox(
        "Method",
        options=['hybrid', 'tfidf', 'exact'],
        index=0,  # Already set to hybrid (index 0)
        help=(
            "• Hybrid: Combines TF-IDF with exact matching (recommended)\n"
            "• TF-IDF: Statistical ranking only\n"
            "• Exact: Term matching only"
        )
    )
    show_ranking_comparison = st.sidebar.checkbox(
        "Show Ranking Comparison",
        value=False,  # Already False - OFF by default
        help="Compare original vs enhanced rankings"
    )
else:
    ranking_method = None
    show_ranking_comparison = False

st.sidebar.markdown("---")

DATA_DIR        = Path(__file__).resolve().parent / "data"
CODES_XLSX      = DATA_DIR / "FinalEDCodes_Complexity.xlsx"
EXAMPLES_JSONL  = DATA_DIR / "edcode_finetune_v7_more_notes.jsonl"

if debug_mode:
    st.sidebar.write("**Debug: Build Info**")
    st.sidebar.write(f"Working dir: {Path.cwd()}")
    st.sidebar.write(f"Data dir  : {DATA_DIR}")
    st.sidebar.write(f"Codes file exists: {CODES_XLSX.exists()}")
    st.sidebar.write(f"Enhanced matcher available: {ENHANCED_AVAILABLE}")

gpt_model   = st.sidebar.selectbox(
    "GPT Model", ["gpt-4o", "gpt-4-turbo-preview", "gpt-4"], index=0
)
temperature = st.sidebar.slider("GPT Temperature", 0.0, 1.0, 0.0, 0.1)

# -----------------------------------------------------------------
#  Utility imports
# -----------------------------------------------------------------
try:
    from ek_utils import load_codes, parse_gpt
except ImportError as e:
    st.error(f"❌ Failed to import ek_utils: {e}")
    st.stop()

def tokenize(txt: str) -> set[str]:
    """Simple alphanumeric tokeniser ≥4 chars."""
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", txt)}

# -----------------------------------------------------------------
#  Few-shot example loader
# -----------------------------------------------------------------
@st.cache_resource
def load_examples(path: Path, n: int = 3) -> str:
    if not path.exists():
        st.warning(f"Examples file not found: {path}")
        return ""
    blocks = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            blocks.append(
                f"Casenote:\n{d['messages'][0]['content']}\n"
                f"Answer:\n{d['messages'][1]['content']}"
            )
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n"

fewshot = load_examples(EXAMPLES_JSONL, 3)

# -----------------------------------------------------------------
#  TF-IDF builder (improved parameters)
# -----------------------------------------------------------------
@st.cache_resource(show_spinner="Building TF-IDF …")
def build_tfidf(df: pd.DataFrame, salt: int):
    df = df.copy()
    df["combined_text"] = (
        df["ED Short List Term"].fillna("") + " " +
        df["description"].fillna("")       + " " +
        df["ED Short List Included conditions"].fillna("")
    )
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),   # 1-grams + 2-grams + 3-grams
        sublinear_tf=True,    # dampen high TF
        min_df=2              # drop singletons
    )
    mat = vec.fit_transform(df["combined_text"])
    return vec, mat, df

# -----------------------------------------------------------------
#  Enhanced matcher builder
# -----------------------------------------------------------------
@st.cache_resource(show_spinner="Building Enhanced Matcher...")
def build_enhanced_matcher(df: pd.DataFrame, salt: int):
    """Build and cache the enhanced clinical matcher"""
    if not ENHANCED_AVAILABLE:
        return None
    
    # Prepare dataframe for enhanced matcher
    df = df.copy()
    # The enhanced matcher expects these exact columns
    df['ICD_10_CM_diagnosis_codes'] = df['ED Short List code']
    df['ED Short List Included'] = df['ED Short List Included conditions'].fillna('')
    df['description'] = df['ED Short List Term'].fillna('')
    
    # Create the matcher
    matcher = EnhancedClinicalMatcher(df)
    return matcher

# -----------------------------------------------------------------
#  Load everything (TF-IDF with salt)
# -----------------------------------------------------------------
try:
    codes_df_raw = load_codes(CODES_XLSX)

    # Build original TF-IDF
    vectorizer, tfidf_matrix, codes_df_with_combined = build_tfidf(
        codes_df_raw,
        get_tfidf_salt()
    )
    codes_df = codes_df_with_combined

    # Build enhanced matcher if available
    enhanced_matcher = None
    if ENHANCED_AVAILABLE:
        enhanced_matcher = build_enhanced_matcher(codes_df_raw, get_tfidf_salt())

    desc_lookup = dict(
        zip(
            codes_df["ED Short List code"],
            codes_df["ED Short List Included conditions"].fillna("")
        )
    )
    
    try:
        client = OpenAI()
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client. Please check your API key.")
        st.stop()

    if debug_mode:
        st.sidebar.write("**Loaded successfully:**")
        st.sidebar.write(f"- Codes           : {len(codes_df)}")
        st.sidebar.write(f"- combined_text ok: {'combined_text' in codes_df.columns}")
        st.sidebar.write(f"- TF-IDF features : {len(vectorizer.get_feature_names_out())}")
        if enhanced_matcher:
            st.sidebar.write(f"- Enhanced matcher: ✅ Ready")

except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# -----------------------------------------------------------------
#  Helper: extract key symptoms
# -----------------------------------------------------------------
def extract_key_symptoms(note_text: str) -> str:
    patterns = [
        r'complain(?:s|ing|ed)?\s+of\s+([^.]+)',
        r'present(?:s|ing|ed)?\s+with\s+([^.]+)',
        r'chief complaint[:\s]+([^.]+)',
        r'c/o\s+([^.]+)',
        r'diagnosis[:\s]+([^.]+)',
        r'impression[:\s]+([^.]+)',
        r'assessment[:\s]+([^.]+)',
    ]
    hits = []
    for p in patterns:
        hits.extend(re.findall(p, note_text, re.IGNORECASE))
    return " ".join(hits[:3]) if hits else ""

# -----------------------------------------------------------------
#  Main processing function (MODIFIED FOR ENHANCED RANKING)
# -----------------------------------------------------------------
def process_note(note_text: str):
    if not note_text.strip():
        return None

    with st.spinner("Analyzing casenote…"):
        key_symptoms = extract_key_symptoms(note_text)

        # Choose ranking method based on settings
        if use_enhanced_ranking and enhanced_matcher and ranking_method:
            # Use enhanced matcher
            enhanced_results = enhanced_matcher.rank_codes(
                note_text,
                method=ranking_method,
                top_k=50,  # Changed from 100 to 50
                return_diagnostics=False
            )
            
            # Convert back to expected format
            codes_df_work = codes_df.copy()
            
            # Check if enhanced_results is a DataFrame
            if hasattr(enhanced_results, 'iterrows'):
                # Map enhanced results back to original dataframe
                for idx, row in enhanced_results.iterrows():
                    mask = codes_df_work['ED Short List code'] == row['ICD_10_CM_diagnosis_codes']
                    if mask.any():
                        codes_df_work.loc[mask, 'Score'] = row['similarity_score']
            
            # Fill any missing scores with 0
            codes_df_work['Score'] = codes_df_work['Score'].fillna(0)
            
            # Also get original TF-IDF scores for comparison if requested
            if show_ranking_comparison:
                note_vec = vectorizer.transform([note_text.lower()])
                codes_df_work['Original_Score'] = cosine_similarity(note_vec, tfidf_matrix).flatten()
            
        else:
            # Use original TF-IDF
            note_vec = vectorizer.transform([note_text.lower()])
            codes_df_work = codes_df.copy()
            codes_df_work["Score"] = cosine_similarity(note_vec, tfidf_matrix).flatten()

        # ---------- verbose debug ----------
        if debug_mode:
            st.sidebar.subheader("🔍 Ranking Debug")
            st.sidebar.write(f"Method: {'Enhanced ' + (ranking_method or '') if use_enhanced_ranking else 'Original TF-IDF'}")
            st.sidebar.write("Max score:", f"{codes_df_work['Score'].max():.4f}")
            st.sidebar.write("Top 10 avg:", f"{codes_df_work['Score'].nlargest(10).mean():.4f}")

        # ---------- shortlist ----------
        shortlist = codes_df_work.sort_values("Score", ascending=False).head(50)[  # Changed to 50
            ["ED Short List code", "ED Short List Term", "Score"]
        ]

        opts_text = "\n".join(
            f"{row['ED Short List code']} — {row['ED Short List Term']}"
            for _, row in shortlist.iterrows()
        )

        # ---------- GPT prompt (FIXED VERSION) ----------
        prompt = f"""{fewshot}You are an expert Australian emergency physician and senior clinical coder with 20+ years of experience.

CRITICAL INSTRUCTIONS:
1. Your response MUST start with "1. "
2. Choose the SINGLE BEST ED principal diagnosis from the provided shortlist
3. Format: "1. CODE — Term — Brief explanation"
4. Provide differentials (lines 2–4) for ANY of these situations:
   - Clinical presentation has overlapping features with other conditions
   - Mentioned differentials in your explanation
   - Diagnostic certainty is <85%
   - Standard ED practice would consider ruling out other conditions
5. If you mention other conditions in your explanation, you MUST list them as differentials

KEY SYMPTOMS IDENTIFIED: {key_symptoms if key_symptoms else "None clearly identified"}

AVAILABLE ED CODES (top 50 by relevance):
{opts_text}

Casenote:
{note_text}

IMPORTANT: Your response must follow this exact format:
1. CODE — Term from list — Your clinical explanation
2. CODE — Term from list — Why this is a differential
3. CODE — Term from list — Why this is a differential

Include the full term from the provided list for each code."""

    # ---------- GPT call ----------
    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system",
                 "content": "You are an expert emergency medicine coder. Always follow the exact format requested: CODE — Term — Explanation"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        gpt_resp = response.choices[0].message.content
        if gpt_resp:
            gpt_resp = gpt_resp.strip()
        else:
            gpt_resp = ""
    except Exception as e:
        st.error(f"{gpt_model} error: {e}")
        return None

    # ---------- parse GPT ----------
    try:
        parsed = parse_gpt(gpt_resp, codes_df)
    except Exception:
        parsed = []

    note_tokens = tokenize(note_text)
    validated   = []
    for item in parsed:
        if len(item) < 3:
            continue
        code, term, expl = item[:3]
        if code in codes_df["ED Short List code"].values:
            hits = (tokenize(term) | tokenize(desc_lookup.get(code, ""))) & note_tokens
            kw   = ", ".join(sorted(hits)) if hits else "code match"
            validated.append((code, term, expl, kw))

    return {
        "validated": validated,
        "shortlist": shortlist,
        "gpt_raw"  : gpt_resp,
        "codes_df" : codes_df_work,
        "key_symptoms": key_symptoms,
    }

# -----------------------------------------------------------------
#  Sidebar button – rebuild TF-IDF
# -----------------------------------------------------------------
if st.sidebar.button("🔄 Re-build TF-IDF"):
    st.session_state["tfidf_salt"] = get_tfidf_salt() + 1
    st.rerun()

if debug_mode:
    st.sidebar.write("**TF-IDF Salt Debug**")
    st.sidebar.write("Current salt:", get_tfidf_salt())
    st.sidebar.write("TF-IDF features:", len(vectorizer.get_feature_names_out()))
    st.sidebar.write("Codes with combined_text:", 'combined_text' in codes_df.columns)

# -----------------------------------------------------------------
#  Session-state initialisation
# -----------------------------------------------------------------
if "processed_note" not in st.session_state:
    st.session_state.processed_note = ""
if "processing_result" not in st.session_state:
    st.session_state.processing_result = None

# -----------------------------------------------------------------
#  Tabs: Paste, Upload, Batch
# -----------------------------------------------------------------
tab_paste, tab_upload, tab_batch = st.tabs(
    ["📝 Paste Note", "📁 Upload File", "📚 Batch Process"]
)

# ----- Paste tab -----
with tab_paste:
    note_input = st.text_area(
        "Paste or enter de-identified casenote here:",
        height=300,
        placeholder="Enter the emergency department clinical note here…",
        key="paste_input",
    )

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("🚀 Process Note", type="primary", key="process_paste"):
            if note_input:
                st.session_state.processed_note  = note_input
                st.session_state.processing_result = process_note(note_input)
                st.rerun()
            else:
                st.warning("Please enter a note first")
    with col2:
        if st.button("🗑️ Clear", key="clear_paste"):
            st.session_state.processed_note  = ""
            st.session_state.processing_result = None
            st.rerun()

# ----- Upload tab -----
with tab_upload:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file:
        file_content = uploaded_file.getvalue().decode(errors="ignore")
        st.success(f"✅ Loaded: {uploaded_file.name}")

        with st.expander("📄 File Preview", expanded=True):
            st.text_area(
                "File content:", value=file_content, height=200, disabled=True
            )

        if st.button("🚀 Process This File", type="primary", key="process_upload"):
            st.session_state.processed_note  = file_content
            st.session_state.processing_result = process_note(file_content)
            st.rerun()

# ----- Batch tab -----
with tab_batch:
    st.header("📁 Batch processing")
    files = st.file_uploader(
        "Select de-identified .txt files",
        type="txt",
        accept_multiple_files=True,
        key="batch_upload",
    )
    if files:
        st.info(f"{len(files)} file(s) ready")
        if st.button("🚀 Run batch"):
            rows, dbg_rows = [], []
            prog = st.progress(0.0, text="Classifying …")

            for i, f in enumerate(files, 1):
                txt    = f.getvalue().decode(errors="ignore")
                result = process_note(txt)

                if result:
                    row = {"File": f.name}
                    for j, (c, t, e, kw) in enumerate(result["validated"], 1):
                        row[f"Code {j}"] = c
                        # scale lookup
                        if c in codes_df["ED Short List code"].values:
                            scale_match = codes_df.loc[
                                codes_df["ED Short List code"] == c, "Scale"
                            ]
                        if not scale_match.empty and len(scale_match) > 0:
                            try:
                                row[f"Scale {j}"] = int(scale_match.iloc[0])
                            except Exception:
                                row[f"Scale {j}"] = ""
                    rows.append(row)

                    if debug_mode:
                        dbg_rows.append(
                            {
                                "File"       : f.name,
                                "Codes"      : len(result["validated"]),
                                "Key Symptoms": result.get("key_symptoms", "")[:50],
                            }
                        )

                prog.progress(i / len(files))

            prog.empty()
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            if debug_mode and dbg_rows:
                with st.expander("🔍 Debug details"):
                    st.dataframe(
                        pd.DataFrame(dbg_rows), hide_index=True, use_container_width=True
                    )

            # Excel download
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                df.to_excel(xw, index=False, sheet_name="ED codes")
            buf.seek(0)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ Download results (.xlsx)",
                data=buf,
                file_name=f"ekoder_batch_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Upload .txt files to enable batch processing.")

# -----------------------------------------------------------------
#  Display results (if any)
# -----------------------------------------------------------------
if st.session_state.processing_result:
    st.divider()
    st.header("🏥 EKoder-4o Recommendations")

    result    = st.session_state.processing_result
    validated = result["validated"]

    if validated:
        for i, (c, t, e, kw) in enumerate(validated, 1):
            scale_val = 0
            scale_match = codes_df.loc[
                codes_df["ED Short List code"] == c, "Scale"
            ]
            if not scale_match.empty:
                try:
                    scale_val = int(scale_match.iloc[0])
                except Exception:
                    scale_val = 0
            dollars = "$" * scale_val

            st.markdown(
                f"### {i}. **{c} — {t}** {dollars} "
                f"<span style='color:#888'>({kw})</span>",
                unsafe_allow_html=True,
            )
            st.write(e)
    else:
        st.warning("No valid codes parsed.")
        with st.expander("GPT-4 raw output"):
            st.code(result["gpt_raw"])

    # ------ show shortlist ------
    st.divider()
    st.subheader("📊 Top 50 Codes by TF-IDF Similarity")
    st.markdown("*These are the codes sent to GPT-4 for consideration:*")

    shortlist_df          = result["shortlist"].copy()
    shortlist_df["Rank"]  = range(1, len(shortlist_df) + 1)
    shortlist_df["Score"] = shortlist_df["Score"].apply(lambda x: f"{x:.3f}")

    with st.expander("View all 50 codes", expanded=True):
        search_term = st.text_input("🔍 Search codes:", key="code_search")
        display_df  = shortlist_df[
            ["Rank", "ED Short List code", "ED Short List Term", "Score"]
        ]
        if search_term:
            display_df = display_df[
                display_df["ED Short List code"].str.contains(search_term, case=False)
                | display_df["ED Short List Term"].str.contains(search_term, case=False)
            ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ------ verbose debug ------
        if debug_mode:
            st.subheader("Key Symptoms Extracted")
            st.write(result.get("key_symptoms", "None"))

            st.subheader("Raw GPT Response")
            st.code(result["gpt_raw"])

    # ------ Ranking comparison if enabled ------
    if show_ranking_comparison and 'Original_Score' in result['codes_df'].columns:
        st.divider()
        st.subheader("🔄 Ranking Method Comparison")
        
        codes_work = result['codes_df'].copy()
        
        # Get top codes from each method
        enhanced_top = codes_work.nlargest(20, 'Score')[['ED Short List code', 'ED Short List Term']].copy()
        enhanced_top['Enhanced_Rank'] = range(1, len(enhanced_top) + 1)
        
        original_top = codes_work.nlargest(20, 'Original_Score')[['ED Short List code', 'ED Short List Term']].copy()
        original_top['Original_Rank'] = range(1, len(original_top) + 1)
        
        # Show top 10 from each method
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original TF-IDF Top 10**")
            for i, row in original_top.head(10).iterrows():
                st.write(f"{row['Original_Rank']}. {row['ED Short List code']} - {row['ED Short List Term']}")
        
        with col2:
            st.write("**Enhanced Ranking Top 10**")
            for i, row in enhanced_top.head(10).iterrows():
                st.write(f"{row['Enhanced_Rank']}. {row['ED Short List code']} - {row['ED Short List Term']}")
        
        # Show general statistics about ranking changes
        st.write("**Ranking Statistics:**")
        
        # Calculate average rank changes for codes that appear in both top 20s
        merged_ranks = pd.merge(
            original_top[['ED Short List code', 'Original_Rank']], 
            enhanced_top[['ED Short List code', 'Enhanced_Rank']], 
            on='ED Short List code', 
            how='inner'
        )
        
        if not merged_ranks.empty:
            merged_ranks['Rank_Change'] = merged_ranks['Original_Rank'] - merged_ranks['Enhanced_Rank']
            avg_change = merged_ranks['Rank_Change'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Codes in both top 20", len(merged_ranks))
            with col2:
                st.metric("Average rank change", f"{avg_change:+.1f}")
            with col3:
                improved = (merged_ranks['Rank_Change'] > 0).sum()
                st.metric("Codes that improved", improved)

# -----------------------------------------------------------------
#  Sidebar session-state debug
# -----------------------------------------------------------------
if debug_mode:
    st.sidebar.divider()
    st.sidebar.subheader("Session State")
    st.sidebar.write(
        f"Has processed note: {bool(st.session_state.processed_note)}"
    )
    st.sidebar.write(
        f"Has result       : {st.session_state.processing_result is not None}"
    )
    if st.session_state.processed_note:
        st.sidebar.write(f"Note length: {len(st.session_state.processed_note)}")
    st.sidebar.write(
        f"Few-shot examples loaded: {'Yes' if fewshot else 'No'}"
    )
if debug_mode:
    st.sidebar.write("Sample combined_text for 'I30.9':")
    sample = codes_df[codes_df["ED Short List code"] == "I30.9"]["combined_text"].values
    st.sidebar.write(sample[0] if len(sample) else "Not found")
if debug_mode and st.session_state.get("processing_result"):
    shortlist = st.session_state.processing_result.get("shortlist")
    if shortlist is not None:
        st.sidebar.write("Top 10 TF-IDF matches:")
        st.sidebar.dataframe(shortlist.head(10))