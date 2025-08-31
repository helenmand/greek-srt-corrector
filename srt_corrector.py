import streamlit as st
import re
import os
import time
import hashlib
from typing import List, Dict, Tuple
import requests
import json
import logging

# set up logging
logging.basicConfig(level=logging.INFO)

# --- constants & configuration ---
class Constants:
    """centralized configuration and constants."""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    PULL_URL = "http://localhost:11434/api/pull"
    # common greek stop words 
    SKIP_WORDS = {
        "ο", "η", "το", "οι", "τα", "του", "της", "των", "πολλά",
        "στο", "στη", "στην", "στον", "και", "ή", "με", "δεν",
        "για", "σε", "από", "ως", "σαν", "ένα", "την", "τον",
        "πολύ", "πολλά", "αλλά", "άλλα"
    }
    DEFAULT_MODEL = "ilsp/meltemi-instruct-v1.5"

# --- utility functions ---
def get_text_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def estimate_tokens(text: str) -> int:
    """roughly estimate tokens based on word count."""
    words = len(text.split())
    return int(words / 0.75)

# --- core classes ---
class TextProcessor:
    """handles all text processing operations."""
    
    def __init__(self):
        self.skip_words = Constants.SKIP_WORDS
    
    def strip_html_tags(self, text: str) -> str:
        """remove html tags from text."""
        return re.sub(r'<[^>]+>', '', text)
    
    def is_trivial_text(self, text: str) -> bool:
        """check if the text is a trivial word/phrase to skip."""
        cleaned = re.sub(r'[^\wάέήίόύώΆΈΉΊΌΎΏϊΐϋΰ]+', '', text.lower(), flags=re.UNICODE)
        return cleaned in self.skip_words or not cleaned
    
    def remove_punctuation(self, text: str) -> str:
        """remove punctuation from text while preserving greek characters."""
        return re.sub(r'[^\w\s\'-]', '', text, flags=re.UNICODE)
    
    def apply_lowercase_first_letter(self, text: str) -> str:
        """lowercase the first letter of the text."""
        if text:
            return text[0].lower() + text[1:]
        return text
    
    def process_text_pipeline(
        self,
        text: str,
        model_client,
        remove_punct: bool = False,
        lowercase_first: bool = False,
        do_spell_check: bool = False
    ) -> Tuple[str, float, int]:
        """
        processes text through a pipeline.
        """
        start_time = time.time()
        
        clean_text = self.strip_html_tags(text)
        corrected_text = clean_text
        
        if do_spell_check and model_client and not self.is_trivial_text(clean_text):
            corrected_text, _, _ = model_client.correct_text(clean_text)
        
        if remove_punct:
            corrected_text = self.remove_punctuation(corrected_text)
        
        if lowercase_first:
            corrected_text = self.apply_lowercase_first_letter(corrected_text)
            
        final_text = corrected_text.strip()
        
        total_time = time.time() - start_time
        output_tokens = estimate_tokens(final_text)
        
        return final_text, total_time, output_tokens

class ModelClient:
    """handles communication with the llm (ollama)."""
    
    def __init__(self, model_name: str = Constants.DEFAULT_MODEL):
        self.model_name = model_name
    
    @st.cache_resource
    def load_model(_self, model_name):
        """loads (and pulls) the model from ollama using st.cache_resource."""
        print(f"modelclient: attempting to load model '{model_name}'. this will only run once per model name.")
        try:
            with requests.post(Constants.PULL_URL, json={"name": model_name}, stream=True, timeout=600) as resp:
                resp.raise_for_status()
            print(f"modelclient: model '{model_name}' pull successful.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"modelclient: error pulling model: {e}")
            st.error(f"error connecting to ollama or pulling model: {e}")
            logging.error(f"ollama request error: {e}")
            return False
    
    def correct_text(self, text: str) -> Tuple[str, float, int]:
        """corrects spelling using the model."""
        start_time = time.time()

        if not st.session_state.get('model_loaded', False):
            return text, time.time() - start_time, estimate_tokens(text)

        try:
            safe_context = re.sub(r'[\r\n]+', ' ', st.session_state.model_context.strip())
            safe_context = re.sub(r'\s+', ' ', safe_context)[:200]
            if not safe_context:
                safe_context = "γενικό περιεχόμενο"  # default fallback

            prompt = (
                f"ο ρόλος σου είναι να διορθώσεις ορθογραφικά το κείμενο που θα σου δοθεί με βάση την σύγχρονη ελληνική γλώσσα. "
                f"διόρθωσε την παρακάτω φράση/λέξη έτσι ώστε να μην περιέχει ορθογραφικά λάθη. "
                f"μην αλλάζεις την κλίση των λέξεων και μην προσθέτεις λέξεις για να ολοκληρωθεί το νόημα. "
                f"η θεματική του κειμένου που θα σου δοθεί αφορά: {safe_context}"
                f"απάντησε μονο με την φράση/λέξη διορθωμένη χωρίς καμία εξήγηση και χωρίς να προσθέσεις νέες λέξεις για να ολοκληρωθεί το νόημα.\n"
                f"παράδειγμα: γειατί\n"
                f"διορθωμένο κείμενο: γιατί\n"
                f"παράδειγμα: ρογμή.\n"
                f"διορθωμένο κείμενο: ρωγμή.\n"
                f"το κείμενο προς ορθογραφική διόρθωση είναι: '{text}'\n"
                f"διορθωμένο κείμενο:"
            )
            print(f"modelclient: sending request to ollama for text: {text}")

            resp = requests.post(
                Constants.OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": st.session_state.get("temperature", 0),
                    "num_predict": st.session_state.get("num_predict", 10),
                    "stream": True,
                },
                timeout=30
            )
            resp.raise_for_status()

            generated_text = "".join(json.loads(line).get("response", "") for line in resp.text.strip().splitlines())
            
            corrected = generated_text.strip()
            corrected = corrected.split('\n')[0].strip().strip("'\"")
            final_corrected = corrected if corrected else text
            
            processing_time = time.time() - start_time
            output_tokens = estimate_tokens(final_corrected)
            print(f"modelclient: received corrected text: '{final_corrected}' in {processing_time:.2f}s.")
            
            return final_corrected, processing_time, output_tokens

        except requests.exceptions.RequestException as e:
            print(f"modelclient: ollama request failed: {e}. returning original text.")
            st.warning(f"ollama request failed: {e}. using original text.")
            logging.error(f"ollama correction error: {e}")
            return text, time.time() - start_time, estimate_tokens(text)
        except json.JSONDecodeError as e:
            print(f"modelclient: failed to parse json: {e}. returning original text.")
            st.warning(f"failed to parse json from ollama: {e}. using original text.")
            logging.error(f"json decode error: {e} - response text: {resp.text}")
            return text, time.time() - start_time, estimate_tokens(text)
        except Exception as e:
            print(f"modelclient: an unexpected error occurred: {e}. returning original text.")
            st.warning(f"an unexpected error occurred during correction: {e}. using original text.")
            logging.error(f"unexpected error: {e}")
            return text, time.time() - start_time, estimate_tokens(text)

class SRTProcessor:
    """handles srt file parsing and processing."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.model_client = ModelClient()
        
    def parse_srt(self, srt_content: str) -> List[Dict]:
        """parse srt content into structured data."""
        blocks = []
        srt_blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in srt_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    timestamp = lines[1]
                    subtitle_text = '\n'.join(lines[2:])
                    
                    clean_original_text = self.text_processor.strip_html_tags(subtitle_text)
                    
                    blocks.append({
                        'index': index,
                        'timestamp': timestamp,
                        'original_text': clean_original_text,
                        'corrected_text': clean_original_text,
                        'processed': False,
                        'processing_time': 0.0,
                        'output_tokens': 0
                    })
                except (ValueError, IndexError):
                    logging.warning(f"skipping malformed block: {lines}")
                    continue
        print(f"srtprocessor: parsed {len(blocks)} blocks.")
        return blocks
    
    def blocks_to_srt(self, blocks: List[Dict]) -> str:
        """convert blocks back to srt format with bolded text."""
        srt_content_list = []
        for block in blocks:
            text = f"<b>{block['corrected_text']}</b>"
            srt_content_list.append(f"{block['index']}\n{block['timestamp']}\n{text}\n")
        
        return '\n'.join(srt_content_list)

# --- main application logic ---
def _format_time(seconds: float) -> str:
    """helper function to format time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_metrics(blocks: List[Dict]) -> Tuple[float, str, bool]:
    """calculate average tps (estimation) and estimated time remaining."""
    processed_blocks = [b for b in blocks if b['processed'] and b['processing_time'] > 0]
    
    if not processed_blocks:
        return 0.0, "n/a", False
    
    total_tokens = sum(b['output_tokens'] for b in processed_blocks)
    total_time = sum(b['processing_time'] for b in processed_blocks)
    
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    
    unprocessed_blocks_count = len([b for b in blocks if not b['processed']])
    is_complete = unprocessed_blocks_count == 0
    
    if is_complete:
        return avg_tps, _format_time(total_time), True
    
    avg_tokens_per_block = total_tokens / len(processed_blocks) if processed_blocks else 0
    remaining_tokens = unprocessed_blocks_count * avg_tokens_per_block
    remaining_seconds = remaining_tokens / avg_tps if avg_tps > 0 else 0
    
    return avg_tps, _format_time(remaining_seconds), False

def reset_for_new_file():
    print("resetting for new file.")
    st.session_state.srt_blocks = []
    st.session_state.current_index = 0
    st.session_state.processing_paused = True
    st.session_state.current_file_hash = None

def _init_session_state():
    """initializes all necessary session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = SRTProcessor()
    if 'srt_blocks' not in st.session_state:
        st.session_state.srt_blocks = []
    if 'processing_paused' not in st.session_state:
        st.session_state.processing_paused = True
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'current_file_hash' not in st.session_state:
        st.session_state.current_file_hash = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = Constants.DEFAULT_MODEL
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.0
    if 'num_predict' not in st.session_state:
        st.session_state.num_predict = 10
    if 'remove_punctuation' not in st.session_state:
        st.session_state.remove_punctuation = True
    if 'spell_check' not in st.session_state:
        st.session_state.spell_check = True
    if 'lower_first_letter' not in st.session_state:
        st.session_state.lower_first_letter = True
    if 'model_context' not in st.session_state:
        st.session_state.model_context = "γενικό περιεχόμενο"
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False


def main():
    _init_session_state()

    st.set_page_config(
        page_title="srt subtitle corrector",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📝 srt subtitle corrector")
    st.markdown("upload an srt file, select processing options, and start when ready.")
    
    # sidebar model settings
    with st.sidebar:
        st.header("model settings")
        new_model_name = st.text_input(
            "model name", value=st.session_state.model_name
        )
        if new_model_name != st.session_state.model_name:
            print(f"model name changed from '{st.session_state.model_name}' to '{new_model_name}'.")
            st.session_state.model_name = new_model_name
            st.session_state.model_loaded = False
        st.session_state.temperature = st.slider(
            "temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.temperature, step=0.05
        )
        st.session_state.num_predict = st.number_input(
            "number of predictions", min_value=1, max_value=100,
            value=st.session_state.num_predict, step=1
        )
        # context for the model
        st.session_state.model_context = st.text_area(
            "context",
            value=st.session_state.model_context,
            help="provide thematic/contextual information for better corrections (e.g., 'οφθαλμολογικό περιεχόμενο')."
        )


    # file upload
    st.header("📁 upload a file")
    uploaded_file = st.file_uploader(
        "choose an srt file", type=['srt'], help="upload your subtitle file (.srt format)"
    )

    # processing options
    st.subheader("⚙️ options")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        st.session_state.remove_punctuation = st.checkbox(
            "remove punctuation", value=st.session_state.remove_punctuation,
            help="removes all punctuation marks from the subtitle text."
        )
    with col_opt2:
        st.session_state.spell_check = st.checkbox(
            "spell check", value=st.session_state.spell_check,
            help="uses the ai model to check and correct greek spelling."
        )
    with col_opt3:
        st.session_state.lower_first_letter = st.checkbox(
            "lowercase first letter", value=st.session_state.lower_first_letter,
            help="changes the first letter of the subtitle text to lowercase."
        )

    # handle file upload and parsing
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_hash = get_text_hash(file_bytes)

        if file_hash != st.session_state.current_file_hash:
            reset_for_new_file()
            srt_content = file_bytes.decode('utf-8')
            st.session_state.srt_blocks = st.session_state.processor.parse_srt(srt_content)
            st.session_state.current_file_hash = file_hash
        
        # model loading logic
        if st.session_state.spell_check:
            if not st.session_state.model_loaded:
                with st.spinner(f"loading model {st.session_state.model_name}..."):
                    if st.session_state.processor.model_client.load_model(st.session_state.model_name):
                        st.session_state.model_loaded = True
                        st.success(f"model {st.session_state.model_name} loaded successfully!")
                        print("model successfully loaded and state updated.")
                    else:
                        st.error("failed to load model. spell check will be skipped.")
                        st.session_state.spell_check = False
                        st.session_state.model_loaded = False
                        print("failed to load model. spell check disabled.")
        else:
            print("spell check is disabled.")
        
        if st.session_state.srt_blocks:
            st.header("🔄 processing controls")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("▶️ start/resume processing", type="primary"):
                    st.session_state.processing_paused = False
                    st.rerun()
            with col2:
                if st.button("⏸️ pause processing"):
                    st.session_state.processing_paused = True
            with col3:
                if st.button("🔄 reset progress"):
                    for block in st.session_state.srt_blocks:
                        block.update({
                            'processed': False,
                            'corrected_text': block['original_text'],
                            'processing_time': 0.0,
                            'output_tokens': 0
                        })
                    st.session_state.current_index = 0
                    st.session_state.processing_paused = True
                    st.rerun()

            # progress bar & metrics
            total_blocks = len(st.session_state.srt_blocks)
            processed_blocks_count = sum(1 for block in st.session_state.srt_blocks if block['processed'])
            st.progress(processed_blocks_count / total_blocks,
                        text=f"progress: {processed_blocks_count}/{total_blocks} blocks processed")

            avg_tps, time_info, is_complete = calculate_metrics(st.session_state.srt_blocks)
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("tokens/second" if not is_complete else "average tps",
                          f"{avg_tps:.1f}" if avg_tps > 0 else "n/a")
            with metric_col2:
                st.metric("est. time remaining" if not is_complete else "total time", time_info)

            # --- preview/edit section first ---
            with st.expander("👁️ preview / edit text", expanded=True):
                if st.session_state.model_context:
                    st.info(f"📝 using context: {st.session_state.model_context}")

                for i, block in enumerate(st.session_state.srt_blocks):
                    text_content = block['corrected_text']
                    edited_text = st.text_area(
                        label=f"block {block['index']} ({block['timestamp']}){' ✅' if block['processed'] else ' ⏳'}",
                        value=text_content,
                        key=f"edit_block_{i}",
                        height=80
                    )
                    block['corrected_text'] = edited_text.strip()
                    if edited_text.strip() != block['original_text'].strip() and not block['processed']:
                        block['processed'] = True

            # --- process one block after ui render ---
            if not st.session_state.processing_paused and st.session_state.current_index < total_blocks:
                idx = st.session_state.current_index
                block_to_process = st.session_state.srt_blocks[idx]
                
                corrected_text, processing_time, output_tokens = \
                    st.session_state.processor.text_processor.process_text_pipeline(
                        block_to_process['original_text'],
                        st.session_state.processor.model_client if st.session_state.spell_check else None,
                        remove_punct=st.session_state.remove_punctuation,
                        lowercase_first=st.session_state.lower_first_letter,
                        do_spell_check=st.session_state.spell_check
                    )
                
                block_to_process['corrected_text'] = corrected_text
                block_to_process['processed'] = True
                block_to_process['processing_time'] = processing_time
                block_to_process['output_tokens'] = output_tokens

                st.session_state.current_index += 1
                time.sleep(0.1)
                st.rerun()

            # download button
            if st.session_state.srt_blocks:
                st.markdown(
                    """
                    <style>
                    div.stDownloadButton > button {
                        display: block;
                        margin: 0 auto; /* center horizontally */
                        width: 100% !important; /* full width */
                        background-color: #28a745 !important; /* green */
                        color: white !important;
                        font-size: 18px !important;
                        border-radius: 8px !important;
                        height: 3em !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                srt_output = st.session_state.processor.blocks_to_srt(st.session_state.srt_blocks)
                st.download_button(
                    label="📥 download subtitles",
                    data=srt_output.encode('utf-8'),
                    file_name=f"edited_{os.path.splitext(uploaded_file.name)[0]}.srt",
                    mime="application/x-subrip",
                )

if __name__ == "__main__":
    main()