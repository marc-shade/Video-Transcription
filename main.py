import streamlit as st
import os
import requests
import warnings
import logging
import io
import time
import pandas as pd

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*is not a valid config option.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import torch safely
try:
    import torch
    # Suppress specific PyTorch warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
except ImportError:
    logger.warning("PyTorch not found. Some ML features may be limited.")
    torch = None

from utils import (
    extract_audio, transcribe_audio, is_valid_video_format,
    translate_text, get_available_languages, export_to_srt, export_to_vtt,
    export_to_markdown, export_to_json
)
from database import TranscriptionDB
import ai_persona
from video_player import render_interactive_player, render_simple_transcript_with_timestamps
from task_presets import (
    TASK_PRESETS, get_task_presets_by_category, get_category_info,
    generate_task_prompt, export_content_to_markdown
)
from batch_processor import (
    BatchJobQueue, JobStatus, get_batch_queue, start_batch_processing,
    process_video_job
)
from typing import Tuple, Optional
import json
import tempfile

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Batch processing state
    if 'batch_queue' not in st.session_state:
        st.session_state.batch_queue = get_batch_queue()
    if 'batch_processing_started' not in st.session_state:
        st.session_state.batch_processing_started = False
    if 'batch_max_concurrent' not in st.session_state:
        st.session_state.batch_max_concurrent = 1
    
    # Default settings
    default_settings = {
        'api_base': "http://localhost:11434",
        'model': "mistral:instruct",
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'repeat_penalty': 1.1,
        'max_tokens': 1024,
        'context_window': 4096
    }
    
    # Try to load saved settings, fall back to defaults if not found
    if 'ollama_settings' not in st.session_state:
        settings_file = "settings.json"
        try:
            with open(settings_file, 'r') as f:
                saved_settings = json.load(f)
                # Merge saved settings with defaults (in case new settings were added)
                st.session_state.ollama_settings = {**default_settings, **saved_settings}
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.ollama_settings = default_settings.copy()
    
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False

def save_settings():
    """Save current settings to a JSON file."""
    settings_file = "settings.json"
    try:
        with open(settings_file, 'w') as f:
            json.dump(st.session_state.ollama_settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving settings: {str(e)}")
        return False

def load_settings():
    """Load settings from JSON file."""
    settings_file = "settings.json"
    try:
        with open(settings_file, 'r') as f:
            saved_settings = json.load(f)
            # Update session state with saved settings
            st.session_state.ollama_settings.update(saved_settings)
            # No rerun needed here as this is called during initialization
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return False

def render_sidebar_settings():
    """Render Ollama settings in the sidebar."""
    
    # Ollama API settings in a collapsible section
    with st.sidebar.expander("🤖 AI Settings", expanded=False):
        st.subheader("Ollama Configuration")
        
        # API Base URL
        api_base = st.text_input(
            "API Base URL",
            value=st.session_state.ollama_settings['api_base'],
            help="The base URL for your Ollama instance"
        )
        
        # Fetch available models
        try:
            available_models = ai_persona.PersonaAnalyzer.get_available_models(api_base)
            if not available_models:
                available_models = ["mistral:instruct"]
                st.error("⚠️ No models found. Is Ollama running?")
        except requests.RequestException as e:
            available_models = ["mistral:instruct"]
            st.error(f"⚠️ Network error fetching models: {e}")
        except Exception as e:
            available_models = ["mistral:instruct"]
            st.error(f"⚠️ Unexpected error fetching models: {e}")
        
        # Ensure the current model is in the list
        current_model = st.session_state.ollama_settings['model']
        if current_model not in available_models:
            available_models.append(current_model)
        
        # Model selection
        model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(current_model),
            help="Select the AI model to use for persona generation and chat"
        )
        
        # Advanced model parameters
        st.subheader("Model Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.ollama_settings['temperature'],
                step=0.1,
                help="Controls randomness in responses"
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.ollama_settings['top_p'],
                step=0.1,
                help="Nucleus sampling threshold"
            )
            
            repeat_penalty = st.slider(
                "Repeat Penalty",
                min_value=1.0,
                max_value=2.0,
                value=st.session_state.ollama_settings['repeat_penalty'],
                step=0.1,
                help="Penalty for repeating tokens"
            )
        
        with col2:
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=st.session_state.ollama_settings['top_k'],
                help="Limits vocabulary in responses"
            )
            
            max_tokens = st.slider(
                "Max Tokens",
                min_value=128,
                max_value=4096,
                value=st.session_state.ollama_settings['max_tokens'],
                step=128,
                help="Maximum response length"
            )
            
            context_window = st.slider(
                "Context Window",
                min_value=512,
                max_value=8192,
                value=st.session_state.ollama_settings['context_window'],
                step=512,
                help="Token context window size"
            )
        
        # Save settings button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Settings"):
                if save_settings():
                    st.success("Settings saved!")
                else:
                    st.error("Failed to save settings")
        
        with col2:
            if st.button("🔄 Reset Defaults"):
                st.session_state.ollama_settings = {
                    'api_base': "http://localhost:11434",
                    'model': "mistral:instruct",
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                    'max_tokens': 1024,
                    'context_window': 4096
                }
                save_settings()  # Save the default settings
                st.rerun()
        
        # Update settings if changed
        current_settings = {
            'api_base': api_base,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': repeat_penalty,
            'max_tokens': max_tokens,
            'context_window': context_window
        }
        
        # Only compare keys that exist in current_settings to avoid infinite rerun
        settings_changed = any(
            st.session_state.ollama_settings.get(k) != v
            for k, v in current_settings.items()
        )
        if settings_changed:
            st.session_state.ollama_settings.update(current_settings)
            save_settings()  # Auto-save when settings change
            st.rerun()
    
    # Initialize database
    db = TranscriptionDB()
    
    # Client Management section
    st.sidebar.header("Client Management")
    client_management_tab = st.sidebar.expander("Manage Clients", expanded=False)
    
    with client_management_tab:
        render_client_management(db)

def render_client_management(db):
    """Render the client management section."""
    st.header("🤝 Client Management")
    
    # Get current list of clients
    clients = db.get_all_clients()
    
    # Client List Section
    st.subheader("Existing Clients")
    if not clients:
        st.info("No clients found. Add a new client below.")
    else:
        # Create a DataFrame for clients
        client_df = pd.DataFrame(clients, columns=['ID', 'Name', 'Email'])
        st.dataframe(client_df, hide_index=True)
    
    # Add Client Form with a unique key
    st.subheader("Add New Client")
    with st.form(key=f"add_client_form_{int(time.time())}"):
        name = st.text_input("Client Name")
        email = st.text_input("Client Email")
        
        submit_button = st.form_submit_button("Add Client")
        
        if submit_button:
            if not name or not email:
                st.error("Please provide both name and email.")
            else:
                try:
                    client_id = db.add_client(name, email)
                    st.success(f"Client '{name}' added successfully!")
                    # Force a rerun to refresh the client list
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding client: {str(e)}")
    
    # Delete Client Section
    st.subheader("Delete Client")
    client_options = [f"{client[1]} ({client[2]})" for client in clients] if clients else []
    
    if client_options:
        selected_client = st.selectbox("Select Client to Delete", client_options)
        
        # Two-step confirmation for deletion
        if 'delete_client_confirmed' not in st.session_state:
            st.session_state.delete_client_confirmed = False
        
        if st.button("Delete Client"):
            if not st.session_state.delete_client_confirmed:
                st.warning(f"Are you sure you want to delete {selected_client}? This action cannot be undone.")
                st.session_state.delete_client_confirmed = True
            else:
                # Find the client ID
                client_to_delete = next(
                    client for client in clients 
                    if f"{client[1]} ({client[2]})" == selected_client
                )
                try:
                    db.delete_client(client_to_delete[0])
                    st.success(f"Client {selected_client} deleted successfully!")
                    # Reset confirmation and force rerun
                    st.session_state.delete_client_confirmed = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting client: {str(e)}")
                    st.session_state.delete_client_confirmed = False
    else:
        st.info("No clients available to delete.")

def get_persona_analyzer() -> ai_persona.PersonaAnalyzer:
    """Get a PersonaAnalyzer instance with current settings."""
    options = {
        "temperature": st.session_state.ollama_settings['temperature'],
        "top_p": st.session_state.ollama_settings['top_p'],
        "top_k": st.session_state.ollama_settings['top_k'],
        "repeat_penalty": st.session_state.ollama_settings['repeat_penalty'],
        "num_predict": st.session_state.ollama_settings['max_tokens'],
        "num_ctx": st.session_state.ollama_settings['context_window']
    }
    return ai_persona.PersonaAnalyzer(
        model=st.session_state.ollama_settings['model'],
        api_base=st.session_state.ollama_settings['api_base'],
        options=options
    )

def get_client_list():
    """Get list of clients for dropdown."""
    db = TranscriptionDB()
    clients = db.get_all_clients()
    return {f"{client[1]} ({client[2]})": client[0] for client in clients}

def check_environment():
    """Check if all required environment variables are set."""
    try:
        # Check if Ollama is running by making a test request
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code != 200:
            st.error("❌ Ollama server is not running. Please start Ollama first.")
            st.stop()
        st.success("✅ Ollama server is running")
        return True
    except requests.exceptions.ConnectionError:
        st.error("❌ Ollama server is not running. Please start Ollama first.")
        st.stop()
    return False

def regenerate_persona_for_transcription(transcription_id: int, original_text: str, db: TranscriptionDB) -> Tuple[bool, Optional[str]]:
    """Regenerate persona prompt for an existing transcription."""
    analyzer = get_persona_analyzer()
    try:
        persona_name, system_prompt = analyzer.analyze_transcript(original_text)
        
        if persona_name and system_prompt and len(system_prompt.strip()) > 0:
            success = db.update_persona_prompt(transcription_id, persona_name, system_prompt)
            if success:
                return True, persona_name
        return False, None
    except Exception as e:
        print(f"Error regenerating persona: {str(e)}")
        return False, None

def render_task_presets(db: TranscriptionDB, transcription_id: int, original_text: str, context: str = "default"):
    """
    Render the task presets section with one-click action buttons.

    Args:
        db (TranscriptionDB): Database connection
        transcription_id (int): ID of the transcription
        original_text (str): Original transcribed text
        context (str, optional): Context of where this is being called.
    """
    st.markdown("### ⚡ Quick Actions")
    st.caption("One-click tasks to generate structured content from the transcript")

    # Get presets by category
    categories = get_task_presets_by_category()
    category_info = get_category_info()

    # Create tabs for each category
    category_tabs = st.tabs([f"{category_info[cat]['icon']} {category_info[cat]['name']}" for cat in categories.keys()])

    for tab, (category_key, presets) in zip(category_tabs, categories.items()):
        with tab:
            # Create columns for preset buttons (3 per row)
            cols = st.columns(3)
            for idx, preset in enumerate(presets):
                with cols[idx % 3]:
                    button_key = f"preset_{preset.id}_{transcription_id}_{context}"
                    if st.button(f"{preset.icon} {preset.name}", key=button_key, help=preset.description, use_container_width=True):
                        # Store which preset was clicked
                        st.session_state[f"active_preset_{transcription_id}_{context}"] = preset.id
                        st.session_state[f"generating_{transcription_id}_{context}"] = True

    # Handle preset generation
    generating_key = f"generating_{transcription_id}_{context}"
    active_preset_key = f"active_preset_{transcription_id}_{context}"

    if st.session_state.get(generating_key, False):
        preset_id = st.session_state.get(active_preset_key)
        if preset_id and preset_id in TASK_PRESETS:
            preset = TASK_PRESETS[preset_id]

            with st.spinner(f"Generating {preset.name}..."):
                try:
                    # Generate the prompt with transcript
                    full_prompt = generate_task_prompt(preset_id, original_text)

                    # Use the AI to generate content
                    analyzer = get_persona_analyzer()
                    generated_content = analyzer.generate_response(
                        f"You are an expert content analyst. Analyze the provided transcript and generate the requested output format.",
                        full_prompt
                    )

                    # Save to database
                    content_id = db.add_generated_content(transcription_id, preset_id, generated_content)

                    st.success(f"✅ {preset.name} generated and saved!")

                    # Reset generation state
                    st.session_state[generating_key] = False
                    st.session_state[active_preset_key] = None
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating content: {str(e)}")
                    st.session_state[generating_key] = False

    # Display previously generated content
    st.markdown("---")
    st.markdown("### 📂 Generated Content")

    generated_items = db.get_generated_content(transcription_id)

    if not generated_items:
        st.info("No content generated yet. Click a Quick Action button above to get started!")
    else:
        for item in generated_items:
            content_id, _, task_type, content, created_at = item
            preset = TASK_PRESETS.get(task_type)
            preset_name = preset.name if preset else task_type
            preset_icon = preset.icon if preset else "📄"

            with st.expander(f"{preset_icon} {preset_name} - {created_at}", expanded=False):
                st.markdown(content)

                # Export and delete buttons
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    # Export as Markdown
                    md_export = export_content_to_markdown(content, preset_name)
                    st.download_button(
                        label="📥 Export MD",
                        data=md_export,
                        file_name=f"{task_type}_{created_at.replace(':', '-').replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"export_md_{content_id}_{context}"
                    )

                with col2:
                    # Delete button
                    delete_key = f"delete_content_{content_id}_{context}"
                    if st.button("🗑️ Delete", key=delete_key):
                        st.session_state[f"confirm_delete_{content_id}"] = True

                # Confirmation for delete
                if st.session_state.get(f"confirm_delete_{content_id}", False):
                    st.warning("Are you sure you want to delete this content?")
                    conf_col1, conf_col2 = st.columns(2)
                    with conf_col1:
                        if st.button("Yes, delete", key=f"confirm_yes_{content_id}_{context}"):
                            db.delete_generated_content(content_id)
                            st.session_state[f"confirm_delete_{content_id}"] = False
                            st.rerun()
                    with conf_col2:
                        if st.button("Cancel", key=f"confirm_no_{content_id}_{context}"):
                            st.session_state[f"confirm_delete_{content_id}"] = False
                            st.rerun()


def render_batch_upload(db: TranscriptionDB, client_id: int):
    """Render the batch upload and job queue interface."""
    st.markdown("### 📁 Batch Upload Queue")
    st.caption("Upload multiple videos at once and process them in the background")

    batch_queue = st.session_state.batch_queue

    # Settings row
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        max_concurrent = st.selectbox(
            "Processing Mode",
            options=[1, 2, 3],
            index=0,
            format_func=lambda x: "Sequential (1 at a time)" if x == 1 else f"Parallel ({x} at a time)",
            help="How many videos to process simultaneously"
        )
        st.session_state.batch_max_concurrent = max_concurrent

    with col2:
        include_timestamps = st.checkbox("Include Timestamps", value=True, key="batch_timestamps")

    with col3:
        # Start/Stop processing button
        if not st.session_state.batch_processing_started:
            if st.button("▶️ Start Processing", use_container_width=True):
                batch_queue.set_max_concurrent(max_concurrent)
                batch_queue.start_workers(process_video_job)
                st.session_state.batch_processing_started = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Processing", use_container_width=True):
                batch_queue.stop_workers()
                st.session_state.batch_processing_started = False
                st.rerun()

    st.markdown("---")

    # File uploader with multiple files
    uploaded_files = st.file_uploader(
        "Drop multiple videos here",
        type=['mp4', 'avi', 'mov', 'mkv', 'm4a'],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files:
        if st.button(f"📤 Add {len(uploaded_files)} file(s) to Queue", use_container_width=True):
            added_count = 0
            for uploaded_file in uploaded_files:
                try:
                    # Save file to temp location
                    ext = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Add to queue
                    batch_queue.add_job(
                        filename=uploaded_file.name,
                        file_path=tmp_path,
                        client_id=client_id,
                        include_timestamps=include_timestamps
                    )
                    added_count += 1
                except Exception as e:
                    st.error(f"Error adding {uploaded_file.name}: {str(e)}")

            if added_count > 0:
                st.success(f"Added {added_count} file(s) to the queue!")
                # Start processing if not already started
                if not st.session_state.batch_processing_started:
                    batch_queue.set_max_concurrent(max_concurrent)
                    batch_queue.start_workers(process_video_job)
                    st.session_state.batch_processing_started = True
                st.rerun()

    st.markdown("---")

    # Queue statistics
    stats = batch_queue.get_queue_stats(client_id)
    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.metric("🕐 Pending", stats['pending'])
    with stat_cols[1]:
        st.metric("⏳ Processing", stats['processing'])
    with stat_cols[2]:
        st.metric("✅ Complete", stats['complete'])
    with stat_cols[3]:
        st.metric("❌ Failed", stats['failed'])
    with stat_cols[4]:
        st.metric("📊 Total", stats['total'])

    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col1:
        if stats['complete'] + stats['failed'] + stats['cancelled'] > 0:
            if st.button("🧹 Clear Completed", use_container_width=True):
                batch_queue.clear_completed_jobs(client_id)
                st.rerun()

    st.markdown("---")

    # Job list
    jobs = batch_queue.get_all_jobs(client_id=client_id)

    if not jobs:
        st.info("No jobs in queue. Upload some videos to get started!")
        return

    for job in jobs:
        with st.container():
            # Status icon and color
            status_icons = {
                JobStatus.PENDING: ("🕐", "Pending"),
                JobStatus.PROCESSING: ("⏳", "Processing"),
                JobStatus.COMPLETE: ("✅", "Complete"),
                JobStatus.FAILED: ("❌", "Failed"),
                JobStatus.CANCELLED: ("🚫", "Cancelled")
            }
            icon, status_label = status_icons.get(job.status, ("❓", "Unknown"))

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{icon} {job.filename}**")
                if job.status == JobStatus.PROCESSING:
                    st.progress(job.progress / 100, text=f"Processing... {job.progress:.0f}%")
                elif job.status == JobStatus.FAILED and job.error_message:
                    st.error(f"Error: {job.error_message}")
                elif job.status == JobStatus.COMPLETE and job.transcription_id:
                    st.caption(f"Transcription ID: {job.transcription_id}")

            with col2:
                st.caption(status_label)
                if job.created_at:
                    st.caption(job.created_at.strftime("%H:%M:%S"))

            with col3:
                if job.status == JobStatus.COMPLETE and job.transcription_id:
                    if st.button("👁️ View", key=f"view_job_{job.id}"):
                        st.session_state['view_transcription_id'] = job.transcription_id
                        st.rerun()
                elif job.status == JobStatus.FAILED:
                    if st.button("🔄 Retry", key=f"retry_job_{job.id}"):
                        batch_queue.retry_job(job.id)
                        st.rerun()
                elif job.status == JobStatus.PENDING:
                    if st.button("❌ Cancel", key=f"cancel_job_{job.id}"):
                        batch_queue.cancel_job(job.id)
                        st.rerun()

            st.markdown("---")


def render_persona_chat(db: TranscriptionDB, transcription_id: int, original_text: str, context: str = "default"):
    """
    Render the persona chat interface with task presets.

    Args:
        db (TranscriptionDB): Database connection
        transcription_id (int): ID of the transcription
        original_text (str): Original transcribed text
        context (str, optional): Context of where this is being called. Defaults to "default".
    """
    # Initialize session state for messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Get persona data
    persona_data = db.get_persona_prompt(transcription_id)

    # If no persona exists, try to generate one
    if not persona_data:
        with st.spinner("Generating persona..."):
            success, new_name = generate_persona_for_transcription(transcription_id, original_text, db)
            if success:
                persona_data = db.get_persona_prompt(transcription_id)
            else:
                st.error("Failed to generate persona. Please try again.")
                return

    # If still no persona, exit
    if not persona_data:
        st.warning("No persona could be generated for this transcription.")
        return

    persona_name, system_prompt = persona_data

    # Create a unique session state key for this transcription's messages
    messages_key = f"messages_{transcription_id}_{context}"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    # Display the system prompt in a container with toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Persona Name:** {persona_name}")
    with col2:
        # Use a unique key combining transcription_id and context
        regen_key = f"regen_{transcription_id}_{context}"
        if st.button("🔄 Regenerate", key=regen_key):
            with st.spinner("Regenerating persona..."):
                success, new_name = regenerate_persona_for_transcription(transcription_id, original_text, db)
                if success:
                    st.success(f"Regenerated persona as '{new_name}'!")
                    st.rerun()
                else:
                    st.error("Failed to regenerate persona. Please try again.")

    if "show_prompt" not in st.session_state:
        st.session_state.show_prompt = False

    # Use a unique key for toggle button
    toggle_key = f"toggle_{transcription_id}_{context}"
    if st.button("Toggle Persona Details", key=toggle_key):
        st.session_state.show_prompt = not st.session_state.show_prompt

    if st.session_state.show_prompt:
        st.markdown("**System Prompt:**")
        # Use a unique key for text area
        prompt_key = f"prompt_{transcription_id}_{context}"
        st.text_area("", system_prompt, height=100, disabled=True, key=prompt_key)

    st.markdown("---")

    # Add Task Presets section
    render_task_presets(db, transcription_id, original_text, context)

    st.markdown("---")
    st.markdown("### 💬 Chat with Persona")

    # Chat interface
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Use a unique key for chat input
    chat_input_key = f"chat_input_{transcription_id}_{context}"
    if prompt := st.chat_input("Ask a question...", key=chat_input_key):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state[messages_key].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            analyzer = get_persona_analyzer()
            response = analyzer.generate_response(system_prompt, prompt)
            st.markdown(response)
            st.session_state[messages_key].append({"role": "assistant", "content": response})

def generate_persona_for_transcription(transcription_id: int, original_text: str, db: TranscriptionDB):
    """Generate a persona prompt for an existing transcription."""
    analyzer = get_persona_analyzer()
    try:
        persona_name, system_prompt = analyzer.analyze_transcript(original_text)
        
        if persona_name and system_prompt and len(system_prompt.strip()) > 0:
            db.add_persona_prompt(transcription_id, persona_name, system_prompt)
            return True, persona_name
        else:
            print("Error: Empty persona or system prompt generated")
            return False, None
    except Exception as e:
        print(f"Error generating persona: {str(e)}")
        return False, None

def process_video(video_path, client_id, include_timestamps, target_language, languages, db, progress_container, progress_bar, status_text, filename):
    """Process a single video file with support for chunked transcription."""
    try:
        # Stage 1: Audio extraction (20% of progress)
        status_text.text("Extracting audio from video...")
        audio_path = extract_audio(video_path)
        progress_bar.progress(0.2)

        # Stage 2: Chunked Transcription (40% of progress)
        status_text.text("Preparing audio chunks...")
        
        # Import pydub here to avoid circular imports
        from pydub import AudioSegment
        
        # Load audio and split into chunks
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 5 * 60 * 1000  # 5-minute chunks
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_length_ms, len(audio))
            chunks.append(audio[start:end])
            start = end
        
        # Transcribe each chunk
        status_text.text("Transcribing audio chunks...")
        final_transcript = ""
        for i, chunk in enumerate(chunks, 1):
            # Create a temporary file for each chunk
            chunk_file = f"temp_chunk_{i}.wav"
            chunk.export(chunk_file, format="wav")
            
            # Transcribe the chunk
            chunk_text = transcribe_audio(chunk_file, include_timestamps)
            final_transcript += f"\n\n[CHUNK {i}]\n{chunk_text}"
            
            # Remove temporary chunk file
            os.remove(chunk_file)
            
            # Update progress (20-60% range)
            progress_bar.progress(0.2 + 0.4 * (i / len(chunks)))
        
        # Clean up original audio file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

        # Stage 3: Translation if needed (60% of progress)
        translated_text = None
        if target_language and target_language != "None":
            status_text.text(f"Translating to {languages[target_language]}...")
            translated_text = translate_text(final_transcript, target_language)
        progress_bar.progress(0.6)

        # Stage 4: Save to database (80% of progress)
        status_text.text("Saving transcription...")
        transcription_id = db.add_transcription(
            client_id, filename, final_transcript,
            translated_text, target_language
        )
        progress_bar.progress(0.8)

        # Stage 5: Generate AI Persona (100% of progress)
        status_text.text("Generating AI persona...")
        analyzer = get_persona_analyzer()
        persona_result = analyzer.analyze_transcript(final_transcript)
        
        # Handle tuple return from analyze_transcript
        if isinstance(persona_result, tuple):
            persona_name, persona_prompt = persona_result
        else:
            # Fallback if the return type is unexpected
            persona_name = "Unknown Persona"
            persona_prompt = persona_result.get("persona_prompt", "")
        
        db.add_persona_prompt(
            transcription_id,
            persona_name,
            persona_prompt
        )
        progress_bar.progress(1.0)

        return final_transcript, translated_text, transcription_id

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        # Cleanup in case of error
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None, None, None

def check_file_size(uploaded_file, max_size_mb=2000):
    """
    Check if the uploaded file size is within acceptable limits.
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file object
        max_size_mb (int): Maximum file size in megabytes
    
    Returns:
        bool: True if file is within size limit, False otherwise
    """
    return True  # Let Streamlit handle file size checks

def handle_large_file_upload(uploaded_file):
    """
    Handle large file uploads by saving to a temporary file.
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file object
    
    Returns:
        str: Path to the saved temporary file
    """
    import tempfile
    import os

    # Create a temporary file with the same extension as the uploaded file
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    return temp_file_path

def export_transcriptions(transcriptions):
    """Create a formatted text file for bulk export."""
    output = io.StringIO()
    
    for filename, text, language, date in transcriptions:
        output.write(f"\n{'='*50}\n")
        output.write(f"File: {filename}\n")
        output.write(f"Date: {date}\n")
        output.write(f"Language: {language if language else 'Original'}\n")
        output.write(f"{'='*50}\n\n")
        output.write(text)
        output.write("\n\n")
    
    return output.getvalue()

def render_client_form(db, client_id=None):
    """Render form for adding/editing client details."""
    client = None if client_id is None else db.get_client_by_id(client_id)
    
    with st.form(key=f"client_form_{client_id if client_id else 'new'}"):
        st.subheader("Client Details")
        name = st.text_input("Name", value=client[1] if client else "")
        email = st.text_input("Email", value=client[2] if client else "")
        
        if st.form_submit_button("Save Client"):
            if not name or not email:
                st.error("Both name and email are required!")
                return False
            
            try:
                if client_id:
                    success = db.update_client(client_id, name, email)
                    if success:
                        st.success("Client updated successfully!")
                    else:
                        st.error("Failed to update client.")
                else:
                    client_id = db.add_client(name, email)
                    st.success("Client added successfully!")
                return True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return False
    return False

def render_transcription_interface():
    """Main interface for the transcription app."""
    
    check_environment()
    initialize_session_state()
    load_settings()  # Load saved settings if available
    
    # Display dragon image at the top of the sidebar
    st.sidebar.image('assets/dragon.png')
    
    render_sidebar_settings()
    st.title("🎥 Video to Text Transcription")
    
    # Initialize database
    db = TranscriptionDB()
    
    # Sidebar for client selection
    clients = get_client_list()
    
    if not clients:
        st.sidebar.warning("No clients found. Please add a new client.")
        with st.sidebar.form("add_client_form"):
            st.subheader("Add New Client")
            new_name = st.text_input("Client Name")
            new_email = st.text_input("Client Email")
            
            if st.form_submit_button("Add Client"):
                if new_name and new_email:
                    try:
                        db.add_client(new_name, new_email)
                        st.success("Client added successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding client: {str(e)}")
                else:
                    st.error("Please fill in all fields!")
        return

    # Client selection
    selected_client = st.sidebar.selectbox(
        "Select a client",
        options=[""] + list(clients.keys()),
        format_func=lambda x: x or "Select a Client",
        key="client_select"
    )

    # Determine client_id
    client_id = clients.get(selected_client) if selected_client else None

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎥 Upload & Transcribe", "📚 View Transcriptions", "📁 Batch Upload"])
    
    with tab1:
        st.markdown("""
        Upload your video file to get the transcribed text and AI persona analysis.
        Supported formats: MP4, AVI, MOV, MKV, M4A
        """)

        if not client_id:
            st.warning("Please select a client first!")
            return

        # Large file upload section
        uploaded_file = st.file_uploader(
            "Choose a video or audio file (supports large files up to 2 GB)", 
            type=['mp4', 'avi', 'mov', 'mkv', 'm4a'],
            accept_multiple_files=False
        )
        
        # Transcription settings
        col1, col2 = st.columns(2)
        with col1:
            include_timestamps = st.checkbox("Include Timestamps", value=False)
        with col2:
            target_language = st.selectbox(
                "Translate to", 
                options=["None"] + list(get_available_languages().keys()),
                key="translation_language"
            )
        
        # Progress tracking
        progress_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Transcription button
        if uploaded_file is not None:
            if st.button("Transcribe to Text"):
                try:
                    # Handle large file upload
                    video_path = handle_large_file_upload(uploaded_file)
                    
                    # Validate video format
                    if not is_valid_video_format(uploaded_file.name):
                        st.error("Unsupported video format. Please upload a valid video file.")
                        os.unlink(video_path)
                        return
                    
                    # Process video
                    languages = get_available_languages()
                    transcription, translated_text, transcription_id = process_video(
                        video_path, 
                        client_id, 
                        include_timestamps, 
                        target_language, 
                        languages, 
                        db, 
                        progress_container, 
                        progress_bar, 
                        status_text, 
                        uploaded_file.name
                    )
                    
                    # Remove temporary video file
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                    
                    # Display results
                    if transcription:
                        st.success("Transcription completed successfully!")

                        # Store video bytes for interactive playback
                        uploaded_file.seek(0)
                        video_bytes = uploaded_file.read()
                        st.session_state['current_video_bytes'] = video_bytes
                        st.session_state['current_video_name'] = uploaded_file.name

                        # Display results in tabs
                        result_tabs = st.tabs(["📝 Transcription", "🔄 Translation", "🤖 AI Persona"] if translated_text else ["📝 Transcription", "🤖 AI Persona"])

                        with result_tabs[0]:
                            # Check if timestamps are present in transcription
                            has_timestamps = '[' in transcription and ':' in transcription and include_timestamps

                            if has_timestamps and 'current_video_bytes' in st.session_state:
                                # Create temp file for video playback
                                ext = os.path.splitext(uploaded_file.name)[1]
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_video:
                                    tmp_video.write(st.session_state['current_video_bytes'])
                                    tmp_video_path = tmp_video.name

                                try:
                                    st.markdown("**Interactive Video Player** - Click timestamps to jump to that position")
                                    render_interactive_player(tmp_video_path, transcription, player_id="new_transcription")
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(tmp_video_path):
                                        os.unlink(tmp_video_path)
                            else:
                                st.text_area("Original Text", transcription, height=300)
                                if not include_timestamps:
                                    st.info("💡 Enable 'Include Timestamps' for interactive click-to-seek functionality")

                            # Export buttons for subtitles and formats
                            st.markdown("**Export Formats:**")
                            export_col1, export_col2, export_col3, export_col4, export_col5 = st.columns(5)

                            base_name = os.path.splitext(uploaded_file.name)[0]

                            with export_col1:
                                srt_content = export_to_srt(transcription)
                                st.download_button(
                                    label="SRT",
                                    data=srt_content,
                                    file_name=f"{base_name}.srt",
                                    mime="text/plain",
                                    key="download_srt_new"
                                )

                            with export_col2:
                                vtt_content = export_to_vtt(transcription)
                                st.download_button(
                                    label="VTT",
                                    data=vtt_content,
                                    file_name=f"{base_name}.vtt",
                                    mime="text/vtt",
                                    key="download_vtt_new"
                                )

                            with export_col3:
                                st.download_button(
                                    label="TXT",
                                    data=transcription,
                                    file_name=f"{base_name}.txt",
                                    mime="text/plain",
                                    key="download_txt_new"
                                )

                            with export_col4:
                                md_content = export_to_markdown(transcription, filename=uploaded_file.name)
                                st.download_button(
                                    label="Markdown",
                                    data=md_content,
                                    file_name=f"{base_name}.md",
                                    mime="text/markdown",
                                    key="download_md_new"
                                )

                            with export_col5:
                                json_content = export_to_json(transcription, filename=uploaded_file.name)
                                st.download_button(
                                    label="JSON",
                                    data=json_content,
                                    file_name=f"{base_name}.json",
                                    mime="application/json",
                                    key="download_json_new"
                                )
                            
                        if translated_text:
                            with result_tabs[1]:
                                st.text_area(f"Translation ({languages[target_language]})", translated_text, height=300)
                        
                        with result_tabs[-1]:
                            render_persona_chat(db, transcription_id, transcription, context="upload")
                
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")
                    # Ensure temporary file is removed in case of error
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.unlink(video_path)

    with tab2:
        if not client_id:
            st.warning("Please select a client to view their transcriptions!")
            return

        transcriptions = db.get_client_transcriptions(client_id)
        if not transcriptions:
            st.info("No transcriptions found for this client.")
            return

        # Track if a transcript deletion is needed
        if 'transcript_to_delete' not in st.session_state:
            st.session_state.transcript_to_delete = None

        for t in transcriptions:
            with st.expander(f"📝 {t[2]} - {t[6]}"):
                # Add delete button at the top of the expander
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"🗑️", key=f"delete_{t[0]}", type="primary"):
                        # Set the transcript to be deleted
                        st.session_state.transcript_to_delete = t[0]

                # Check if timestamps are present
                has_timestamps = '[' in t[3] and ':' in t[3]

                with col1:
                    if has_timestamps:
                        # Offer video re-upload for interactive playback
                        video_key = f"video_upload_{t[0]}"
                        reupload_file = st.file_uploader(
                            "Upload video for interactive playback",
                            type=['mp4', 'avi', 'mov', 'mkv', 'm4a'],
                            key=video_key,
                            help="Upload the original video to enable click-to-seek functionality"
                        )

                        if reupload_file:
                            # Create temp file for video playback
                            ext = os.path.splitext(reupload_file.name)[1]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_video:
                                tmp_video.write(reupload_file.read())
                                tmp_video_path = tmp_video.name

                            try:
                                st.markdown("**Interactive Video Player** - Click timestamps to jump to that position")
                                render_interactive_player(tmp_video_path, t[3], player_id=f"view_{t[0]}")
                            finally:
                                if os.path.exists(tmp_video_path):
                                    os.unlink(tmp_video_path)
                        else:
                            # Show transcript with timestamps highlighted
                            render_simple_transcript_with_timestamps(t[3])
                    else:
                        st.text_area("Original Text", t[3], height=150)

                # Export buttons for subtitles and formats
                st.markdown("**Export Formats:**")
                exp_col1, exp_col2, exp_col3, exp_col4, exp_col5 = st.columns(5)
                base_filename = os.path.splitext(t[2])[0]

                with exp_col1:
                    srt_data = export_to_srt(t[3])
                    st.download_button(
                        label="SRT",
                        data=srt_data,
                        file_name=f"{base_filename}.srt",
                        mime="text/plain",
                        key=f"download_srt_{t[0]}"
                    )

                with exp_col2:
                    vtt_data = export_to_vtt(t[3])
                    st.download_button(
                        label="VTT",
                        data=vtt_data,
                        file_name=f"{base_filename}.vtt",
                        mime="text/vtt",
                        key=f"download_vtt_{t[0]}"
                    )

                with exp_col3:
                    st.download_button(
                        label="TXT",
                        data=t[3],
                        file_name=f"{base_filename}.txt",
                        mime="text/plain",
                        key=f"download_txt_{t[0]}"
                    )

                with exp_col4:
                    md_data = export_to_markdown(t[3], filename=t[2])
                    st.download_button(
                        label="Markdown",
                        data=md_data,
                        file_name=f"{base_filename}.md",
                        mime="text/markdown",
                        key=f"download_md_{t[0]}"
                    )

                with exp_col5:
                    json_data = export_to_json(t[3], filename=t[2])
                    st.download_button(
                        label="JSON",
                        data=json_data,
                        file_name=f"{base_filename}.json",
                        mime="application/json",
                        key=f"download_json_{t[0]}"
                    )
                
                # Verification for transcript deletion
                if st.session_state.transcript_to_delete == t[0]:
                    st.warning(f"Are you sure you want to delete this transcript?")
                    col_confirm1, col_confirm2 = st.columns(2)
                    with col_confirm1:
                        if st.button("Confirm Delete", key=f"confirm_delete_{t[0]}"):
                            try:
                                # Delete the specific transcript
                                db.delete_transcript(t[0])
                                st.success("Transcript deleted successfully!")
                                # Reset the transcript to delete
                                st.session_state.transcript_to_delete = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting transcript: {str(e)}")
                    
                    with col_confirm2:
                        if st.button("Cancel", key=f"cancel_delete_{t[0]}"):
                            # Reset the transcript to delete
                            st.session_state.transcript_to_delete = None
                
                if t[4]:  # If there's a translation
                    language_display = get_available_languages().get(t[5], "Unknown")
                    st.text_area(f"Translation ({language_display})", t[4], height=150)
                
                # Add persona management section
                st.markdown("---")
                st.subheader("AI Persona")
                
                # Get existing persona data
                persona_data = db.get_persona_prompt(t[0])
                
                if persona_data:
                    render_persona_chat(db, t[0], t[3], context="view")
                else:
                    st.warning("No persona available for this transcription")
                    if st.button("Generate Persona", key=f"gen_{t[0]}"):
                        with st.spinner("Generating persona..."):
                            success, persona_name = generate_persona_for_transcription(t[0], t[3], db)
                            if success:
                                st.success(f"Generated persona '{persona_name}'!")
                                st.rerun()
                            else:
                                st.error("Failed to generate persona. Please try again.")

    with tab3:
        if not client_id:
            st.warning("Please select a client to use batch upload!")
        else:
            render_batch_upload(db, client_id)

if __name__ == "__main__":
    render_transcription_interface()