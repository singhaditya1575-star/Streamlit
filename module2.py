# import streamlit as st
# import requests
# import json
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from collections import Counter
# import re
# import random
# import time
# from math import exp
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # --- 1. INITIAL SETUP & CONFIGURATION ---
# # 1.1 Streamlit Page Config
# st.set_page_config(
#     page_title="Module 1: AI + Collaboration Analytics Foundations (Cerebras)",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# # 1.2 Session State Initialization 
# if 'progress' not in st.session_state:
#     st.session_state.progress = {f'C{i}': '🔴' for i in range(1, 6)} 
#     st.session_state.journal = []
#     st.session_state.lab_results = {}
#     st.session_state.current_tab = 'Intro'
#     st.session_state.guidance = "Welcome!" 
#     st.session_state.c1_step = 1 
#     st.session_state.c4_results = pd.DataFrame() 
#     st.session_state.onboarding_done = False 
# # --- Authentication State (Bypassed - Always Authenticated) ---
# if 'is_authenticated' not in st.session_state:
#     st.session_state.is_authenticated = True # <-- Set to True by default
# if 'user_info' not in st.session_state:
#     st.session_state.user_info = {"user_id": "DirectRunUser"} # <-- Set a default user info
# # ----------------------------
# if 'assistant_chat_history' not in st.session_state:
#     st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor for Collaboration Analytics. Ask me anything in simple terms about synthetic data, EDA, or what to do next!"}]
# if 'last_assistant_call' not in st.session_state:
#     st.session_state.last_assistant_call = 0
# # 1.3 AI Model API Configuration (Provider-Agnostic)
# AI_API_URL = "https://api.cerebras.ai/v1/chat/completions"
# API_KEY_NAME = "CEREBRAS_API_KEY" # Using the name you provided
# # --- MODEL SELECTION ---
# DEFAULT_MODEL = "qwen-3-32b" 
# MODEL_OPTIONS = [
#     DEFAULT_MODEL, 
#     "gpt-oss-120b", 
#     "llama-4-scout-17b-16e-instruct",
#     "llama-3.3-70b" 
# ]
# # -----------------------------
# # --- 1.4 CUSTOM STREAMLIT STYLING (Theme: Collaboration Blue) ---
# STYLING = """
# <style>
# /* Main Background and Text */
# .stApp {
#     background-color: #FFFDF7; 
#     color: #333; 
#     font-family: Inter, sans-serif;
# }
# /* Sidebar and Panel Colors */
# .st-emotion-cache-1c9v61q { 
#     background-color: #f0f0f0;
#     border-right: 2px solid #0d47a1; /* Deep Blue Accent */
# }
# /* Assistant Messages (For dynamic chat history) */
# .assistant-message {
#     background-color: #e0f7fa; /* Light cyan for assistant */
#     color: #004d40;
#     padding: 10px;
#     border-radius: 8px;
#     margin-bottom: 5px;
#     border-left: 3px solid #008080;
# }
# .user-message {
#     background-color: #fff9e0; /* Light yellow for user */
#     color: #795548;
#     padding: 10px;
#     border-radius: 8px;
#     margin-bottom: 5px;
#     border-left: 3px solid #B8860B;
# }
# /* Highlight for the current step button */
# .stButton>button[kind="primary"] {
#     border: 3px solid #FF5733 !important; /* Bright orange highlight */
#     animation: pulse 1.5s infinite;
# }
# @keyframes pulse {
#     0% {
#         box-shadow: 0 0 0 0 rgba(255, 87, 51, 0.7);
#     }
#     70% {
#         box-shadow: 0 0 0 10px rgba(255, 87, 51, 0);
#     }
#     100% {
#         box-shadow: 0 0 0 0 rgba(255, 87, 51, 0);
#     }
# }
# /* Card/Panel Styling (Glass Effect, Shadows) */
# div[data-testid*="stVerticalBlock"], div[data-testid*="stHorizontalBlock"], .stTextInput, .stTextArea, .stSelectbox {
#     border-radius: 15px;
#     box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
#     background: rgba(255, 255, 255, 0.9); /* Subtle Glass Effect */
#     border: 1px solid #eee;
#     padding: 10px;
# }
# /* Primary Buttons (Glass/Glow Effect) */
# .stButton>button {
#     background-color: #008080; /* Teal/Neon Accent */
#     color: white !important;
#     border: none;
#     border-radius: 8px;
#     transition: all 0.3s ease;
#     font-weight: 600;
#     box-shadow: 0 0 10px rgba(0, 128, 128, 0.5); /* Neon Glow */
# }
# .stButton>button:hover {
#     background-color: #006666;
#     box-shadow: 0 0 15px rgba(0, 128, 128, 0.8);
# }
# /* Title Styling */
# .title-header {
#     color: #0d47a1;
#     font-weight: 800;
#     font-size: 38px;
# }
# /* Progress Tracker Styling for Sidebar */
# .progress-tracker {
#     padding: 10px;
#     border-radius: 10px;
#     background-color: #ffffff;
#     border: 1px solid #ddd;
#     margin-bottom: 10px;
# }
# </style>
# """
# st.markdown(STYLING, unsafe_allow_html=True)
# # --- 2. UTILITY & ANALYSIS FUNCTIONS ---
# def get_progress_badge(key):
#     return st.session_state.progress.get(key, '🔴')
# def get_progress_percent():
#     """Calculates module completion percentage."""
#     completed_count = sum(1 for status in st.session_state.progress.values() if status == '🟢')
#     total_count = len(st.session_state.progress)
#     return int((completed_count / total_count) * 100)
# def update_progress(key, status):
#     """Sets the completion status for a specific lab key."""
#     st.session_state.progress[key] = status
# def update_guidance(message):
#     """Updates the dynamic instruction message."""
#     st.session_state.guidance = message
# def glossary_tooltip(term: str, definition: str):
#     """Helper for creating clickable tooltip-like text."""
#     return f'<span title="{definition}" style="cursor: pointer; border-bottom: 1px dotted #0d47a1;">{term} ℹ️</span>'
# def calculate_coherence_score(text):
#     """
#     Calculates a SIMULATED Coherence Score (0-100) for educational purposes.
#     Formula: Rewards word count and word complexity (long words), penalizes extreme brevity.
#     """
#     word_count = len(text.split())
#     long_word_count = len([w for w in text.split() if len(w) > 6])
#     score = 40 + (word_count / 10) + (long_word_count * 2)
#     return min(100, max(10, int(score)))
# def analyze_text_metrics(text):
#     """
#     Calculates tokens, length, and Flesch Reading Ease (Readability Score).
#     """
#     tokens = text.split()
#     word_count = len(tokens)
#     syllable_count = sum(len(re.findall('[aeiouy]+', w.lower())) for w in tokens)
#     sentence_count = len(re.split(r'[.!?]+', text))
#     if word_count == 0 or sentence_count == 0:
#         flesch_score = 100 
#     else:
#         # Standard Flesch Formula (simplified constant)
#         flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
#     return {
#         "tokens": len(tokens),
#         "flesch_score": max(0, min(100, int(flesch_score))),
#         "text_length": len(text)
#     }
# def save_to_journal(title, prompt, result, metrics=None):
#     """Saves the lab result and reflection to the session journal."""
#     st.session_state.journal.append({
#         'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
#         'lab': st.session_state.current_tab,
#         'title': title,
#         'prompt': prompt,
#         'result': result,
#         'metrics': metrics or {}
#     })
# def explain_llm_output(output):
#     """Simulated LLM summary for the learner (Simplified)."""
#     return "The AI Instructor has summarized the output: This response demonstrates the model's ability to generate text based on the given prompt. Analyze the metrics below to see its quality!"
# def get_c2_explanation(role, tone, metrics):
#     """
#     Generates an AI explanation specific to the C2 Role Prompt Lab.
#     """
#     flesch_score = metrics.get('flesch_score', 50)
#     word_count = metrics.get('tokens', 0)
#     base_explanation = f"The model successfully adopted the persona of a **{role}** using a **{tone}** tone. "
#     if "Skeptical Professor" in role or "Formal" in tone:
#         base_explanation += "The role constraint prioritized **detailed, complex reasoning**, which aligns with the observed "
#     elif "Child's Book Author" in role or "Whimsical" in tone:
#         base_explanation += "The model focused on **simple language and clarity** to match the persona, which should result in a higher readability score. "
#     elif "Urgent" in tone:
#         base_explanation += "The tone constraint enforced **short, direct sentences and focused length**, which contributes to the word count."
#     # --- Dynamic Metric Analysis ---
#     # 1. Readability Analysis
#     if flesch_score < 30:
#         base_explanation += f"Critically, the Flesch Readability Score is only **{flesch_score}/100**. This low score confirms the output uses **long sentences and complex vocabulary** (high word complexity), as is expected from a highly technical or academic persona like the **{role}**."
#     elif flesch_score > 70:
#         base_explanation += f"The high Flesch Readability Score of **{flesch_score}/100** indicates the language is **simple and easy to understand**, consistent with a less formal tone or audience."
#     else:
#         base_explanation += f"The Flesch score of **{flesch_score}/100** suggests a moderate, professional complexity, striking a balance between detail and accessibility."
#     # 2. Length Analysis (optional)
#     if word_count > 150:
#         base_explanation += f" Furthermore, the **{word_count} words** used indicates a comprehensive, verbose response, which is common when the role encourages high detail (like a professor or analyst)."
#     elif word_count < 50:
#         base_explanation += f" The brevity of **{word_count} words** shows the model strictly adhered to the tone and max token limit, focusing only on the core answer."
#     return base_explanation + " This lab highlights how role and style constraints fundamentally shift not just the content, but the measurable linguistic complexity of the output."

# # --- Updated C1 Data Generation Function with Parameters ---
# def generate_synthetic_collab_data(
#     n_groups=100,
#     comm_freq_min=1,
#     comm_freq_max=10,
#     success_bias=0.0,  # Extra randomness to success (±)
#     trust_influence=1.0,  # Weight of trust in success logic
#     participation_threshold=0.6
# ):
#     np.random.seed(42)
#     data = []
#     for i in range(n_groups):
#         group_id = f"G{i+1:03d}"
#         comm_freq = np.random.randint(comm_freq_min, comm_freq_max + 1)
#         task_clarity = np.random.randint(1, 6)
#         role_clarity = np.random.randint(1, 6)
#         conflict_freq = np.random.randint(0, 5)
#         equal_participation = np.random.rand()
#         trust_score = np.random.randint(1, 6)

#         # Adjust success logic with user-controlled weights
#         base_success = (
#             (comm_freq > 5) and 
#             (task_clarity >= 4) and 
#             (conflict_freq <= 1) and 
#             (equal_participation > participation_threshold)
#         )
#         # Add trust influence: if trust is high, slightly relax other rules
#         trust_boost = trust_score >= 5 and trust_influence > 0.5
#         success = int(base_success or trust_boost)

#         # Optional: add noise/bias
#         if np.random.rand() < success_bias:
#             success = 1 - success  # flip for realism

#         data.append([
#             group_id, comm_freq, task_clarity, role_clarity,
#             conflict_freq, equal_participation, trust_score, success
#         ])
#     df = pd.DataFrame(data, columns=[
#         'Group_ID', 'Communication_Frequency', 'Task_Clarity', 'Role_Clarity',
#         'Conflict_Frequency', 'Equal_Participation', 'Trust_Score', 'Successful_Collaboration'
#     ])
#     return df

# def llm_call_cerebras(messages, model=DEFAULT_MODEL, max_tokens=256, temperature=0.7):
#     """Handles the secure API call to the AI Model provider with process explanation."""
#     API_READ_TIMEOUT = 60
#     # Check for API Key (simplified for non-technical users)
#     try:
#         api_key = st.secrets[API_KEY_NAME] 
#     except KeyError:
#         if st.session_state.current_tab != 'Assistant':
#             st.error(f"⚠️ **API Key Missing!** Please configure the **{API_KEY_NAME}** in your `secrets.toml` file to run the labs.")
#         return {"error": f"API Error: {API_KEY_NAME} not configured."}
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
#     # START: Process Explanation 
#     if st.session_state.current_tab in [f'C{i}' for i in range(1, 6)]:
#         log_container = st.container(border=True)
#         log_container.subheader("💻 AI Model Processing Steps")
#         steps = [
#             ("✅ Input Sent", "Your prompt is securely sent."),
#             ("💻 AI Model Working", f"The chosen model ({model}) is calculating the best response."),
#             ("➡️ Generating Words", "The model predicts the output word by word."),
#             ("✨ Response Ready", "The final text is compiled and returned to the dashboard."),
#         ]
#         for i, (msg, detail) in enumerate(steps):
#             log_container.markdown(f"**Step {i+1}**: {msg}")
#             log_container.caption(detail)
#             time.sleep(0.05) 
#     # Final API Call
#     start_time = time.time() 
#     try:
#         response = requests.post(AI_API_URL, json=payload, headers=headers, timeout=API_READ_TIMEOUT)
#         end_time = time.time() 
#         if response.status_code != 200:
#             error_detail = response.json().get("message", response.text[:100])
#             if st.session_state.current_tab in [f'C{i}' for i in range(1, 6)]:
#                 log_container.error("🚨 **Error:** The AI connection failed. Please check your API Key or try a different model.")
#             return {"error": f"API Call Failed ({response.status_code}): {error_detail}"}
#         data = response.json()
#         content = data["choices"][0]["message"]["content"]
#         tokens_generated = len(content.split()) 
#         tokens_used = data.get("usage", {}).get("total_tokens", tokens_generated)
#         if st.session_state.current_tab in [f'C{i}' for i in range(1, 6)]:
#             log_container.success("✅ Response received successfully.")
#         time_to_generate = end_time - start_time
#         throughput_tps = tokens_generated / time_to_generate if time_to_generate > 0 else 0
#         return {
#             "content": content, 
#             "model": model, 
#             "tokens_used": tokens_used,
#             "latency": time_to_generate,
#             "throughput_tps": throughput_tps
#         }
#     except requests.exceptions.RequestException as e:
#         if st.session_state.current_tab in [f'C{i}' for i in range(1, 6)]:
#             log_container.error(f"🚨 **Network/Timeout Error:** {e}. This usually means the request took too long.")
#         return {"error": f"API Call Failed: {e}"}
# # --- 3. ONBOARDING & GETTING STARTED DASHBOARD ---
# def show_onboarding_modal():
#     """1. Onboarding Modal (First Login) - Popup version."""
#     if not st.session_state.get("onboarding_done", False):
#         st.session_state["onboarding_done"] = True
#         st.toast("👋 Welcome to AI Collaboration Analytics Explorer! Let's get started!", icon="🚀")
#         with st.popover("✨ **Welcome to the Collaboration Lab!** ✨", use_container_width=True):
#             st.markdown("""
#                 ### Here's Your Guided Flow:
#                 1.  **Data Generation (C1):** Learn to create synthetic collaboration datasets. 📊
#                 2.  **EDA (C2):** Learn to give the AI a role and tone for better results. 🔍
#                 3.  **ML Modeling (C3):** Learn to measure and improve the quality of the AI's response. 🤖
#                 4.  **Interpretability (C4):** Learn to compare and score different prompts. 📏
#                 5.  **Intervention (C5):** Learn to optimize for objective control. 🎯
#                 Click the tabs (C1, C2, etc.) above to begin your hands-on training!
#             """)
#             st.progress(0.1, text="Loading Core Concepts...")
# def render_getting_started():
#     """
#     Creates the main dashboard view detailing C1 through C5 labs.
#     """
#     st.markdown('<div class="title-header">🧭 Your Collaboration Analytics Journey</div>', unsafe_allow_html=True)
#     st.markdown("---")
#     st.header("💡 Module Overview: From Data to Intervention")
#     st.info("""
#         This module teaches you **AI for Collaboration Analytics**—the art of using synthetic data, EDA, and ML to understand and improve student group work. 
#         We move from basic data generation (C1) and exploring patterns (C2) to advanced techniques like building predictive models (C3), comparing model insights (C4), and optimizing for actionable educator strategies (C5).
#     """)
#     # --- C1: EXPLORING YOUR FIRST PROMPT ---
#     st.markdown("---")
#     st.subheader("⚙️ C1: Exploring Your First Dataset 🚀 (The Basics)")
#     col_def, col_goal = st.columns([1, 1])
#     with col_def:
#         st.markdown("#### What You'll Explore:")
#         st.markdown(f"""
#             - **Definition:** A **Synthetic Dataset** is artificially created data that mimics real-world patterns (like communication frequency, trust scores).
#             - **Key Concept:** You'll learn that even small changes to a dataset's parameters lead to big changes in its characteristics, including its success rate.
#             - **New Terms:** Learn about {glossary_tooltip('Features', 'The columns in your dataset (e.g., Communication_Frequency, Trust_Score).')} and **Targets** (the outcome variable to predict).
#         """)
#     with col_goal:
#         st.markdown("#### The Goal:")
#         st.success("""
#             **Goal:** Successfully run your first data generation and understand the **direct cause-and-effect** between your parameters and the AI's output, including its speed.
#         """)
#         st.markdown("---")
#         st.markdown("#### Step-by-Step Guide Preview:")
#         st.progress(0.0, text="Step 1/3: Set Parameters → Step 2/3: Generate → Step 3/3: Analyze Dataset")
#     # --- C2: ROLE PROMPT DESIGNER ---
#     st.markdown("---")
#     st.subheader("🎭 C2: Role Prompt Designer (Adding Personality)")
#     col_def_2, col_goal_2 = st.columns([1, 1])
#     with col_def_2:
#         st.markdown("#### What You'll Explore:")
#         st.markdown(f"""
#             - **Definition:** A **Role Prompt** gives the AI a persona, like "Act as a Data Analyst" or "You are an Educator."
#             - **Key Concept:** By adding **Role** and **Tone** (e.g., Formal, Humorous), you control the AI's personality, leading to better, more relevant outputs.
#             - **New Terms:** Learn how {glossary_tooltip('Readability Score', 'A metric (like Flesch) that measures how easy the text is for a general audience to read.')} changes based on the persona.
#         """)
#     with col_goal_2:
#         st.markdown("#### The Goal:")
#         st.success("""
#             **Goal:** Master prompt structure by forcing the AI to adopt a **specific role and tone**, demonstrating that the AI's style is fully controllable and measurable.
#         """)
#         st.markdown("---")
#         st.markdown("#### Step-by-Step Guide Preview:")
#         st.progress(0.0, text="Step 1/2: Select Role/Tone → Step 2/2: Run & Analyze Persona Shift")
#     # --- C3: TEMPERATURE & CONTEXT LAB ---
#     st.markdown("---")
#     st.subheader("🌡️ C3: ML Modeling Lab (Controlling Creativity)")
#     col_def_3, col_goal_3 = st.columns([1, 1])
#     with col_def_3:
#         st.markdown("#### Core Definition:")
#         st.markdown(f"""
#             - **Definition:** **Predictive Modeling** uses algorithms to learn patterns and predict outcomes.
#             - **Key Concept:** You'll explore **Accuracy** and **Classification Reports** (precision, recall) by training a model on your collaboration data.
#             - **Hands-on Action:** You'll run an experiment (training a model) and visualize the results (accuracy score, confusion matrix).
#         """)
#     with col_goal_3:
#         st.markdown("#### The Goal:")
#         st.success("""
#             **Goal:** Develop intuition for building and evaluating a simple ML model, showing how data-driven predictions can inform educational practice. You'll know how to train a model and interpret its performance metrics.
#         """)
#         st.markdown("---")
#         st.markdown("#### Step-by-Step Guide Preview:")
#         st.progress(0.0, text="Step 1/3: Load Data → Step 2/3: Train Model → Step 3/3: Evaluate Performance")
#     # --- C4: MULTI-PROMPT COMPARISON STUDIO ---
#     st.markdown("---")
#     st.subheader("⭐ C4: Model Interpretability Studio (Finding the Best Prompt)")
#     col_def_4, col_goal_4 = st.columns([1, 1])
#     with col_def_4:
#         st.markdown("#### Core Definition:")
#         st.markdown(f"""
#             - **Definition:** **Model Interpretability** is the process of understanding *why* a model made its prediction, often using techniques like feature importance.
#             - **Key Concept:** You'll use **Feature Importance Scores** and your own **Manual Analysis** (1-5 stars) to find the most influential factors.
#             - **Hands-on Action:** Input different model results (or re-run C3 with different parameters), run interpretability analysis, and rate the resulting insights in a comparative table.
#         """)
#     with col_goal_4:
#         st.markdown("#### The Goal:")
#         st.success("""
#             **Goal:** Determine which model interpretation method is **empirically superior** by scoring insights based on both automatic metrics and your manual judgment. You'll learn to think like a data scientist, identifying the structural elements (high importance scores) that consistently point to key collaboration factors.
#         """)
#         st.markdown("---")
#         st.markdown("#### Step-by-Step Guide Preview:")
#         st.progress(0.0, text="Step 1/3: Run Interpretability → Step 2/3: Score Insights → Step 3/3: Analyze Results Table")
#     # --- C5: PROMPT OPTIMIZATION CHALLENGE ---
#     st.markdown("---")
#     st.subheader("🎯 C5: Intervention Optimization Challenge (Achieving Objective Control)")
#     col_def_5, col_goal_5 = st.columns([1, 1])
#     with col_def_5:
#         st.markdown("#### Core Definition:")
#         st.markdown(f"""
#             - **Definition:** **Intervention Optimization** is the iterative process of refining a predictive model or its parameters until its insights meet specific, measurable quality standards for educator actionability.
#             - **Key Concept:** This is a test of your ability to control model performance ({glossary_tooltip('Accuracy', 'The percentage of correct predictions made by the model.')}) and interpretability (Feature Importance Thresholds).
#             - **Hands-on Action:** Set targets (e.g., Accuracy > 80%, Top Feature Importance > 20%), train a model, analyze the metrics, and refine parameters until you pass both targets.
#         """)
#     with col_goal_5:
#         st.markdown("#### The Goal:")
#         st.success("""
#             **Goal:** Iteratively refine your model to hit **two objective targets simultaneously**. You'll gain confidence in designing, testing, and controlling AI insights for real-world educational use, culminating in the completion of your Collaboration Analytics Module.
#         """)
#         st.markdown("---")
#         st.markdown("#### Step-by-Step Guide Preview:")
#         st.progress(0.0, text="Step 1/4: Set Targets → Step 2/4: Train Model → Step 3/4: Run & Check Metrics → Step 4/4: Refine & Repeat")
#     st.markdown("---")
#     st.markdown("### Ready to start? Click on the **C1: First Dataset 🚀** tab above!")

# # --- 4. LAB IMPLEMENTATION FUNCTIONS (C1 - C5) ---
# def render_lab1():
#     st.header("C1: Exploring Your First Dataset 🚀")
#     st.markdown("##### **Goal:** Learn how to create a synthetic dataset reflecting student collaboration dynamics and observe its characteristics.")
#     with st.expander("📝 Instructions: Definition & Process", expanded=True):
#         st.markdown("""
#         **What is Synthetic Data?** **Synthetic Data** is artificially created data that mimics the statistical properties of real-world data. Here, it simulates group project features.
#         **Metrics Explained (Dataset Quality):**
#         * **Success Rate:** The percentage of groups predicted to collaborate successfully. **(Target: 40-70%)**.
#         * **Feature Range:** The minimum and maximum values for each feature (e.g., Communication_Frequency 1-10).
#         **Action (Step 1):** Adjust the number of groups and parameters below. Try 100 or 200.
#         **Action (Step 2):** Click **Generate Dataset** to see the synthetic data!
#         """)
#     if 'c1_result' not in st.session_state:
#         st.session_state.c1_result = None
#     if 'c1_reflection' not in st.session_state:
#         st.session_state.c1_reflection = ""

#     # --- Step 1: Input and Parameters ---
#     n_groups = st.slider("Number of Synthetic Groups:", 50, 500, 200, step=50, key='c1_n_groups')

#     # --- Performance Metrics DataFrame Initialization (C1 specific) ---
#     if 'c1_performance_df' not in st.session_state:
#         st.session_state.c1_performance_df = pd.DataFrame(columns=['Metric', 'Value'])
    
#     # --- Advanced Settings Expander ---
#     with st.expander("⚙️ Adjust Data Generation Dials (Advanced Settings)", expanded=False):
#         st.markdown("### 🎚️ Tune How Your Synthetic Data Behaves")
        
#         col_a, col_b = st.columns(2)
        
#         with col_a:
#             st.markdown("#### Communication & Conflict")
#             comm_min = st.slider("Min Messages/Day", 1, 20, 1, key='c1_comm_min', help="Minimum communication frequency for a group.")
#             comm_max = st.slider("Max Messages/Day", 1, 20, 10, key='c1_comm_max', help="Maximum communication frequency for a group.")
#             conflict_max = st.slider("Max Conflicts/Week", 0, 10, 4, key='c1_conflict_max', help="Maximum conflict frequency allowed.")

#         with col_b:
#             st.markdown("#### Success Logic")
#             part_thresh = st.slider("Participation Fairness Threshold", 0.1, 0.9, 0.6, step=0.1, key='c1_part_thresh', help="How fairly must work be distributed for success?")
#             trust_influence = st.radio("Trust Score Impact", ["Low", "Medium", "High"], index=1, key='c1_trust', help="How much does trust affect success?")
#             trust_map = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
#             trust_val = trust_map[trust_influence]

#             noise_bias = st.slider("Realism Noise (%)", 0, 20, 0, key='c1_noise', help="Add randomness to make the data less predictable.") / 100.0

#         # --- Dataset Name Input ---
#         st.markdown("#### Dataset Configuration")
#         dataset_name = st.text_input("Name Your Dataset:", value="My_Synthetic_Collaboration_Data", key='c1_dataset_name', help="Give your generated dataset a custom name for identification.")

#         st.info("These settings shape how 'success' is determined in your dataset—great for simulating different classroom dynamics!")

#     st.markdown("---")
#     # --- Step 2: Run and Execute ---
#     if st.button("Generate Synthetic Dataset", key='c1_run', type='primary'):
#         with st.spinner("Generating synthetic collaboration data..."):
#             df = generate_synthetic_collab_data(
#                 n_groups=n_groups,
#                 comm_freq_min=comm_min,
#                 comm_freq_max=comm_max,
#                 participation_threshold=part_thresh,
#                 trust_influence=trust_val,
#                 success_bias=noise_bias
#             )
#             st.session_state.c1_result = df
#             st.session_state.c1_reflection = "" # Reset reflection
#             if df is not None:
#                 # Calculate metrics and store in DataFrame
#                 success_rate = df['Successful_Collaboration'].mean()
#                 data = {
#                     'Metric': ['Total Groups', 'Success Rate', 'Features', 'Dataset Name'],
#                     'Value': [
#                         len(df),
#                         f"{success_rate:.2%}",
#                         list(df.columns[:-1]), # Exclude target
#                         dataset_name # Include the custom name
#                     ]
#                 }
#                 st.session_state.c1_performance_df = pd.DataFrame(data)
#                 update_guidance("✅ C1 Step 2 Complete! Now, observe the generated dataset and performance metrics.")
#             update_progress('C1', '🟢')
#             st.rerun()

#     # --- Step 3: Output and Reflection ---
#     if st.session_state.c1_result is not None:
#         st.subheader("Generated Dataset Preview")
#         st.dataframe(st.session_state.c1_result.head(10), use_container_width=True)
#         # --- Performance Table ---
#         st.subheader("Dataset Metrics and Analysis")
#         st.dataframe(st.session_state.c1_performance_df.set_index('Metric'), use_container_width=True)
#         # AI Instructor Summary
#         metrics_df = st.session_state.c1_performance_df.set_index('Metric')
#         success_rate = metrics_df.loc['Success Rate', 'Value']
#         dataset_name = metrics_df.loc['Dataset Name', 'Value'] # Retrieve the name
#         summary = "The dataset has been generated with a realistic success rate, suitable for training predictive models."
#         st.subheader("🧠 AI Instructor Summary")
#         st.success(f"{summary} **Dataset Insight:** This dataset, named **'{dataset_name}'**, contains **{metrics_df.loc['Total Groups', 'Value']} groups** with a success rate of **{success_rate}**, which is ideal for modeling.")
#         st.markdown("---")
#         st.subheader("4. Your Reflection & Insights")
#         st.session_state.c1_reflection = st.text_area(
#             "What do you notice about the feature distributions or the success rate? How might this data reflect real student groups?",
#             value=st.session_state.c1_reflection,
#             height=100,
#             key='c1_reflection_input'
#         )
#         # --- Save Insight Button (Only for journal, no navigation) ---
#         if st.button("Save Insight & Complete Lab C1", key='c1_save_complete'):
#             # Pass the calculated metrics to the journal save function
#             metrics_data = {
#                 'reflection': st.session_state.c1_reflection,
#                 'Success Rate': success_rate,
#                 'Total Groups': int(metrics_df.loc['Total Groups', 'Value']),
#                 'Dataset Name': dataset_name # Include name in journal
#             }
#             save_to_journal("Synthetic Data Exploration", f"n_groups={n_groups}, name={dataset_name}", st.session_state.c1_result.to_dict(), metrics_data)
#             update_progress('C1', '🟢')
#             update_guidance("🎉 C1 Lab Complete! Move to the **C2: Role Prompt Designer** tab to learn about persona-based prompting.")
#             st.success("Insight saved! Your progress has been updated.")
#             st.rerun()

# def render_lab2():
#     st.header("C2: Role Prompt Designer 🎭")
#     st.markdown("##### **Definition:** A **Role Prompt** assigns a persona, expertise, or identity to the LLM (e.g., 'Act as a Data Analyst').")
#     st.markdown("##### **Goal:** Learn how **role**, **tone**, and **style** constraints steer the LLM's personality and output.")
#     with st.expander("📝 Instructions: Definition & Process", expanded=False):
#         st.markdown("""
#         **Action:** Select a professional role and an emotional tone. Then, click **Run Role Prompt** to submit your query.
#         **Output:** An LLM response filtered through the selected persona and tone.
#         **Learning Outcome:** Understand that defining a **role** forces the LLM to adopt a specific knowledge profile, improving relevance and style and helping the model understand intent.
#         """)
#     roles = ["Data Analyst", "Educator", "Researcher", "Project Manager"]
#     tones = ["Formal", "Casual", "Urgent", "Whimsical"]
#     col1, col2 = st.columns(2)
#     with col1:
#         role = st.selectbox("Choose a Role:", roles, key='c2_role')
#     with col2:
#         tone = st.selectbox("Choose a Tone/Style Constraint:", tones, key='c2_tone')

#     # --- Advanced Settings Expander for C2 ---
#     with st.expander("⚙️ Prompt & Output Settings (Advanced)", expanded=False):
#         col_a, col_b = st.columns(2)
#         with col_a:
#             st.markdown("#### Base Query Configuration")
#             query_length = st.radio("Base Query Length", ["Short", "Medium", "Long"], index=1, key='c2_query_len', help="How detailed is your base query?")
#             length_map = {"Short": "Analyze factors for success.", "Medium": "Analyze the key factors for successful student collaboration.", "Long": "Based on your expertise, analyze the key factors for successful student collaboration, including communication, trust, and task clarity, and provide a brief conclusion."}
#             base_query_default = length_map[query_length]
#         with col_b:
#             st.markdown("#### Response Configuration")
#             desired_tokens = st.slider("Desired Response Length (Tokens)", 50, 500, 250, key='c2_tokens', help="How long should the response be?")
#             temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, key='c2_temp', help="Higher values = more random, lower = more focused.")

#     user_query = st.text_input("Your Base Query:", base_query_default, key='c2_query')
#     full_prompt = f"Act as a **{role}**. Respond in a **{tone}** tone. Based on these instructions, address the following query: '{user_query}'"
#     st.markdown("---")
#     st.subheader("Final Prompt Construction:")
#     st.code(full_prompt, language='markdown')
#     if st.button("Run Role Prompt (Step 1)", key='c2_run', type='primary'):
#         if not user_query.strip():
#             st.warning("Please enter a base query.")
#             return
#         with st.spinner("Executing role-constrained prompt..."):
#             result = llm_call_cerebras([{"role": "user", "content": full_prompt}], max_tokens=desired_tokens, temperature=temperature, model=DEFAULT_MODEL)
#             st.session_state.c2_result = result
#             if 'content' in result:
#                 st.session_state.c2_metrics = analyze_text_metrics(result['content'])
#                 save_to_journal(f"Role Test: {role} in {tone} tone", full_prompt, result, st.session_state.c2_metrics)
#                 update_progress('C2', '🟢')
#                 update_guidance("✅ C2 Complete! Analyze how the tone shifted in the response.")
#             else:
#                 st.error(result['error'])

#     # --- CHECK ADDED: Ensure c1_result exists AND is not None ---
#     if 'c1_result' not in st.session_state or st.session_state.c1_result is None:
#         st.warning("Please complete C1 first to generate data.")
#         return
#     # --- END CHECK ---
#     if 'c2_result' in st.session_state and 'content' in st.session_state.c2_result:
#         st.subheader("Step 2: Analysis & Visualization")
#         col_res, col_vis = st.columns([2, 1])
#         with col_res:
#             st.info("LLM Response (Observe the role/tone shift):")
#             st.code(st.session_state.c2_result['content'], language='markdown')
#             # AI Explanation for C2
#             st.subheader("🧠 AI Instructor Explanation")
#             # PASS THE METRICS to the explanation function
#             ai_explanation = get_c2_explanation(role, tone, st.session_state.c2_metrics)
#             st.success(ai_explanation)
#         with col_vis:
#             metrics = st.session_state.c2_metrics
#             st.metric("Words Used", metrics['tokens'])
#             st.metric("Flesch Readability", f"{metrics['flesch_score']}/100")
#             st.markdown("##### Tone Histogram (Simulated)")
#             tone_score = {"Formal": 90, "Casual": 30, "Urgent": 70, "Whimsical": 50}[tone]
#             st.bar_chart({"Score": [tone_score]}, use_container_width=True)
        
#         # --- Guidance for Next Step (No Button) ---
#         st.markdown("---")
#         st.info("✅ C2 Complete! You can now proceed to **C3: ML Modeling Lab** to build a predictive model on your data.")

#     elif 'c2_result' in st.session_state and 'error' in st.session_state.c2_result:
#         st.error(st.session_state.c2_result['error'])

# def render_lab3():
#     st.header("C3: ML Modeling Lab 🤖")
#     st.markdown("##### **Definition:** **Predictive Modeling** uses algorithms to learn patterns and predict outcomes.")
#     st.markdown("##### **Goal:** Explore building and evaluating a model to predict collaboration success.")
#     with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
#         st.markdown("""**Action:** Load your generated data, select model parameters, and click **Train Model**. **Output:** Model performance metrics (Accuracy, Classification Report) and a performance visualization. **Learning Outcome:** Develop intuition for how model parameters and data quality affect prediction accuracy.""")

#     # --- CHECK ADDED: Ensure c1_result exists AND is not None ---
#     if 'c1_result' not in st.session_state or st.session_state.c1_result is None:
#         st.warning("Please complete C1 first to generate data.")
#         return
#     # --- END CHECK ---
#     df = st.session_state.c1_result.copy()
#     X = df.drop(['Group_ID', 'Successful_Collaboration'], axis=1)
#     y = df['Successful_Collaboration']

#     # --- Advanced Settings Expander for C3 ---
#     with st.expander("⚙️ Model Configuration (Advanced Settings)", expanded=False):
#         st.markdown("### 🎛️ Tune Your Model's Parameters")
#         col1, col2 = st.columns(2)
#         with col1:
#             test_size = st.slider("Test Set Size (%):", 10, 50, 30, key='c3_test_size', help="Percentage of data used for testing.")
#             test_size = test_size / 100
#             max_depth = st.slider("Max Depth of Trees:", 1, 20, 10, key='c3_max_depth', help="How deep can each tree grow? Higher = more complex.")
#         with col2:
#             n_estimators = st.slider("Number of Trees (Model Complexity):", 10, 200, 100, key='c3_n_est', help="More trees can improve accuracy but take longer.")
#             min_samples_split = st.slider("Min Samples to Split Node:", 2, 20, 5, key='c3_min_split', help="Minimum samples required to further split a node.")

#     st.subheader("1. Model Parameters and Data")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Test Set Size", f"{test_size*100:.0f}%")
#     with col2:
#         st.metric("Model Complexity (Trees)", n_estimators)

#     if st.button("Train Model (Step 1)", key='c3_run', type='primary'):
#         if 'c3_history' not in st.session_state:
#             st.session_state.c3_history = []
#         with st.spinner(f"Training model with {n_estimators} trees..."):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             report = classification_report(y_test, y_pred, output_dict=True)
#             result = {
#                 'model': 'RandomForest',
#                 'accuracy': accuracy,
#                 'classification_report': report,
#                 'predictions': y_pred,
#                 'actuals': y_test
#             }
#             st.session_state.c3_result = result
#             save_to_journal(f"ML Model Training: {n_estimators} trees, depth {max_depth}", f"Test Size: {test_size}, Min Split: {min_samples_split}", result)
#             update_progress('C3', '🟢') 
#             update_guidance("✅ C3 Complete! Analyze the model's performance metrics.")
#             st.rerun()

#     if 'c3_result' in st.session_state:
#         st.subheader("Step 2: Model Performance")
#         result = st.session_state.c3_result
#         col_acc, col_rep = st.columns([1, 1])
#         with col_acc:
#             st.metric("Model Accuracy", f"{result['accuracy']:.2%}")
#         with col_rep:
#             st.text("Classification Report:")
#             st.code(classification_report(result['actuals'], result['predictions']))
#         st.subheader("Step 3: Performance Visualization")
#         # Simple bar chart of accuracy vs. a baseline (e.g., predicting all 0s)
#         baseline_acc = 1 - result['actuals'].mean() # Accuracy if always predicting failure
#         fig = go.Figure(data=[
#             go.Bar(name='Model Accuracy', x=['Model'], y=[result['accuracy']], marker_color='green'),
#             go.Bar(name='Baseline Accuracy', x=['Baseline'], y=[baseline_acc], marker_color='red')
#         ])
#         fig.update_layout(title="Model vs. Baseline Accuracy", yaxis_title="Accuracy")
#         st.plotly_chart(fig, use_container_width=True)
#         # --- AI Instructor Graph Summary ---
#         st.subheader("🧠 AI Instructor Graph Summary")
#         summary_message = f"""
#         The **Model vs. Baseline Accuracy** chart shows how much better your AI model is compared to a simple guess.
#         * **Model Accuracy ({result['accuracy']:.2%}):** This is how often your model was correct.
#         * **Baseline Accuracy ({baseline_acc:.2%}):** This is how often you'd be correct if you just guessed 'Unsuccessful' every time.
#         **Key Insight:** Your model significantly outperforms the baseline, meaning it has learned meaningful patterns from the data to make better predictions.
#         """
#         st.success(summary_message)

#         # --- Guidance for Next Step (No Button) ---
#         st.markdown("---")
#         st.info("✅ C3 Complete! You can now proceed to **C4: Model Interpretability Studio** to analyze which features drive success.")

# def render_lab4():
#     st.header("C4: Model Interpretability Studio ⭐")
#     st.markdown("##### **Definition:** **Model Interpretability** explains *why* a model made its predictions.")
#     st.markdown("##### **Goal:** Compare and score the importance of features identified by the model.")
#     with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
#         st.markdown("""**Action:** If you have a trained model from C3, click **Analyze Feature Importance**. **Output:** A comparative table with feature importance scores and a space for your manual rating. **Learning Outcome:** Evaluate which features are most influential for predicting success.""")

#     # --- CHECK ADDED: Ensure c1_result and c3_result exist AND are not None ---
#     if 'c3_result' not in st.session_state or 'c1_result' not in st.session_state:
#         st.warning("Please complete C1 (Data Generation) and C3 (Model Training) first.")
#         return
#     if st.session_state.c1_result is None or st.session_state.c3_result is None:
#         st.warning("Please complete C1 (Data Generation) and C3 (Model Training) first.")
#         return
#     # --- END CHECK ---

#     df = st.session_state.c1_result.copy()
#     X = df.drop(['Group_ID', 'Successful_Collaboration'], axis=1)
#     y = df['Successful_Collaboration']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # --- Advanced Settings Expander for C4 ---
#     with st.expander("⚙️ Interpretability Settings (Advanced)", expanded=False):
#         st.markdown("### 🧠 Choose How to Analyze the Model")
#         interpretation_method = st.selectbox("Interpretation Method", ["Feature Importance (RF)", "SHAP (Simulated)"], key='c4_method', help="Which method should be used to determine feature importance?")
#         st.info("Currently, the application uses the built-in Feature Importance from the Random Forest model. SHAP simulation is planned for future versions.")

#     if st.button("Analyze Feature Importance (Step 2)", key='c4_run', type='primary'):
#         # Get feature importances from the trained model (using the same parameters as C3 for consistency)
#         # Re-train with the parameters from C3 session state if they exist, otherwise use defaults
#         n_est = st.session_state.get('c3_n_est', 100)
#         max_dep = st.session_state.get('c3_max_depth', 10)
#         min_split = st.session_state.get('c3_min_split', 5)
        
#         trained_model = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, min_samples_split=min_split, random_state=42)
#         trained_model.fit(X_train, y_train)

#         # Get feature importances
#         importances = trained_model.feature_importances_
#         feature_names = X.columns.tolist()
#         results = []
#         for i, (name, imp) in enumerate(zip(feature_names, importances)):
#             results.append({
#                 'Feature ID': i + 1,
#                 'Feature Name': name,
#                 'Importance Score': imp,
#                 'Manual Rating': 3 # Default rating
#             })
#         if results:
#             st.session_state.c4_results = pd.DataFrame(results) 
#             update_progress('C4', '🟡') 
#             update_guidance("➡️ C4: Review the table below and manually rate each feature's importance (Step 3).")
#             st.rerun()

#     if not st.session_state.c4_results.empty:
#         st.subheader("3. Feature Importance Results")
#         results_df = st.session_state.c4_results.copy() 
#         rating_inputs = []
#         for i, row in results_df.iterrows():
#             initial_rating = int(row.get('Manual Rating', 3)) 
#             rating = st.slider(f"Feature {row['Feature Name']} Rating (1=Not Important, 5=Very Important):", 1, 5, initial_rating, key=f'c4_rate_{row["Feature ID"]}')
#             rating_inputs.append(rating)
#         results_df['Manual Rating'] = rating_inputs
#         st.dataframe(results_df[['Feature Name', 'Importance Score', 'Manual Rating']], use_container_width=True)
#         # --- Post-Comparison Summary ---
#         if results_df['Manual Rating'].sum() > 0:
#             most_important_id = results_df.loc[results_df['Importance Score'].idxmax()]['Feature ID']
#             most_important_name = results_df.loc[results_df['Importance Score'].idxmax()]['Feature Name']
#             st.markdown("---")
#             st.subheader("Summary: Key Driver Insight")
#             st.info(f"""
#             Based on the model's analysis and your ratings:
#             * **Most Important Feature:** **{most_important_name}** (ID: {most_important_id}) was identified as the most influential by the model.
#             * **Key Metric:** The highest calculated **Importance Score** was **{results_df['Importance Score'].max():.3f}**.
#             * **Conclusion:** This feature is the strongest predictor of collaboration success according to the Random Forest model. Educators should focus on this aspect.
#             """)
#         # --- End Post-Comparison Summary ---
        
#         # --- Save Insight Button (Only for journal, no navigation) ---
#         if st.button("Save & Complete Lab C4", key='c4_complete'):
#             for _, row in results_df.iterrows():
#                 save_to_journal(f"C4 Feature Importance: {row['Feature Name']}", f"Score: {row['Importance Score']}", {'feature': row['Feature Name'], 'score': row['Importance Score']}, 
#                                  {'Importance Score': row['Importance Score'], 'Manual Rating': row['Manual Rating']})
#             update_progress('C4', '🟢')
#             st.success("C4 Complete! Results saved to Learning Journal.")
#             update_guidance("✅ C4 Complete! Move to C5: Intervention Optimization Challenge.")
#             st.rerun()

# def render_lab5():
#     st.header("C5: Intervention Optimization Challenge 🎯")
#     st.markdown("##### **Definition:** **Intervention Optimization** refines model parameters until insights meet specific, measurable standards.")
#     st.markdown("##### **Goal:** Iteratively adjust parameters to meet specific, measurable performance and interpretability thresholds.")
#     with st.expander("📝 Instructions: Action, Output, & Learning", expanded=False):
#         st.markdown("""
#         **Action:** Set your targets for model accuracy and feature importance. Click **Train & Evaluate Attempt**.
#         **Output:** Live feedback on whether your model meets the targets.
#         **Learning Outcome:** Practice systematic model refinement to achieve objective, metric-based goals.
#         """)
#     target_accuracy = st.slider("Target Model Accuracy (%):", 60, 100, 80, key='c5_target_acc', help="Your model's accuracy must be above this percentage.")
#     target_accuracy = target_accuracy / 100
#     target_top_feature_imp = st.slider("Target Top Feature Importance (%):", 10, 50, 25, key='c5_target_imp', help="The most important feature's score must be above this percentage.")
#     target_top_feature_imp = target_top_feature_imp / 100

#     # --- Advanced Settings Expander for C5 ---
#     with st.expander("⚙️ Optimization Parameters (Advanced Settings)", expanded=False):
#         st.markdown("### 🎛️ Fine-tune your Model for the Challenge")
#         col1, col2 = st.columns(2)
#         with col1:
#             n_estimators = st.slider("Number of Trees (Model Complexity for C5):", 10, 200, 100, key='c5_n_est', help="Adjust this to try and meet your targets.")
#             max_depth = st.slider("Max Tree Depth for C5:", 1, 20, 10, key='c5_max_depth', help="Limit the depth of each tree.")
#         with col2:
#             min_samples_split = st.slider("Min Samples to Split (C5):", 2, 20, 5, key='c5_min_split', help="Minimum samples required to split a node in C5.")
#             test_size = st.slider("Test Set Size (C5):", 10, 50, 30, key='c5_test_size', help="Percentage of data used for testing in C5.") / 100

#     st.subheader("1. The Challenge")
#     st.info(f"Challenge: Train a model where Accuracy is **above {target_accuracy:.0%}** AND the top feature's importance is **above {target_top_feature_imp:.0%}**.")

#     # --- CHECK ADDED: Ensure c1_result exists AND is not None ---
#     if 'c1_result' not in st.session_state or st.session_state.c1_result is None:
#         st.warning("Please complete C1 (Data Generation) first.")
#         return
#     # --- END CHECK ---
#     df = st.session_state.c1_result.copy()
#     X = df.drop(['Group_ID', 'Successful_Collaboration'], axis=1)
#     y = df['Successful_Collaboration']

#     if 'c5_attempts' not in st.session_state:
#         st.session_state.c5_attempts = []
#     if st.button("Train & Evaluate Attempt (Step 2)", key='c5_run', type='primary'):
#         with st.spinner("Training and evaluating optimization attempt..."):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             importances = model.feature_importances_
#             top_feature_imp = importances.max() # Get the highest importance score
#             # Check metrics against targets
#             passed_acc = accuracy >= target_accuracy
#             passed_imp = top_feature_imp >= target_top_feature_imp
#             passed = passed_acc and passed_imp
#             attempt_result = {
#                 'id': len(st.session_state.c5_attempts) + 1,
#                 'n_estimators': n_estimators,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'test_size': test_size,
#                 'accuracy': accuracy,
#                 'top_feature_importance': top_feature_imp,
#                 'passed': passed,
#                 'model_details': model # This might be too large to store, consider storing summary only
#             }
#             st.session_state.c5_attempts.append(attempt_result)
#             save_to_journal(f"Optimization Attempt {len(st.session_state.c5_attempts)}", f"n_est={n_estimators}, depth={max_depth}, min_split={min_samples_split}", attempt_result, {'accuracy': accuracy, 'top_feature_imp': top_feature_imp})
#             if passed:
#                 update_progress('C5', '🟢')
#                 update_guidance(f"🥳 Success! C5 complete on attempt {len(st.session_state.c5_attempts)}.")
#             else:
#                 update_progress('C5', '🟡')
#                 update_guidance("🟡 C5: Attempt failed. Analyze the metrics below and adjust parameters (Step 1).")
#             st.rerun()

#     if st.session_state.c5_attempts:
#         st.subheader("3. Optimization Trajectory")
#         last_attempt = st.session_state.c5_attempts[-1]
#         if last_attempt['passed']:
#             st.balloons()
#             st.success(f"🥳 CHALLENGE PASSED on Attempt {last_attempt['id']}! Metrics met the targets.")
#         else:
#             st.error(f"❌ Attempt {last_attempt['id']} Failed. Analyze the metrics below and adjust parameters and click 'Train & Evaluate Attempt' again.")
#         col_f, col_t = st.columns(2)
#         # Calculate deltas for C5 metrics
#         delta_acc = last_attempt['accuracy'] - target_accuracy
#         delta_imp = last_attempt['top_feature_importance'] - target_top_feature_imp
#         col_f.metric(
#             r"Current Accuracy (Target $\geq$ " + f"{target_accuracy:.0%}" + ")", 
#             f"{last_attempt['accuracy']:.2%}", 
#             delta=f"{delta_acc:.2%} from target",
#             delta_color='normal' if delta_acc >= 0 else 'inverse'
#         )
#         col_t.metric(
#             r"Top Feature Imp (Target $\geq$ " + f"{target_top_feature_imp:.0%}" + ")", 
#             f"{last_attempt['top_feature_importance']:.2%}",
#             delta=f"{delta_imp:.2%} from target",
#             delta_color='normal' if delta_imp >= 0 else 'inverse'
#         )
#         st.markdown("---")
#         # Show model details or summary
#         st.info(f"Model used {last_attempt['n_estimators']} trees with max depth {last_attempt['max_depth']}. Top feature importance score was {last_attempt['top_feature_importance']:.3f}.")
#         df_attempts = pd.DataFrame(st.session_state.c5_attempts)
#         fig = px.line(df_attempts, x='id', y=['accuracy', 'top_feature_importance'], 
#                       title="Performance Metrics Over Attempts",
#                       labels={'id': 'Attempt Number', 'value': 'Score'},
#                       color_discrete_map={'accuracy': '#007BFF', 'top_feature_importance': '#28a745'})
#         # Add target lines
#         fig.add_hline(y=target_accuracy, line_dash="dash", line_color="blue", annotation_text="Accuracy Target", annotation_position="bottom right")
#         fig.add_hline(y=target_top_feature_imp, line_dash="dash", line_color="green", annotation_text="Feature Imp Target", annotation_position="top right")
#         st.plotly_chart(fig, use_container_width=True)

# def render_learning_journal():
#     st.header("📘 Learning Journal & Progress")
#     st.markdown("##### **Goal:** Review and reflect on the key experiments you've run in each lab.")
#     st.markdown("---")
#     st.subheader("Your Module Progress")
#     progress_percent = get_progress_percent()
#     st.progress(progress_percent, text=f"Module Completion: **{progress_percent}%**")
#     cols = st.columns(5)
#     for i in range(1, 6):
#         lab_key = f'C{i}'
#         status = get_progress_badge(lab_key)
#         cols[i-1].metric(f"Lab {i}", f"{lab_key} Status", status)
#     st.markdown("---")
#     st.subheader("Saved Experiments & Reflections")
#     if st.session_state.journal:
#         # Reverse the journal to show the newest entries first
#         reversed_journal = st.session_state.journal[::-1]
#         for entry in reversed_journal:
#             with st.expander(f"**[{entry['timestamp'].split(' ')[0]}] {entry['lab']}: {entry['title']}**", expanded=False):
#                 st.markdown(f"**Prompt Used:**")
#                 st.code(entry['prompt'], language='markdown')
#                 st.markdown(f"**AI Response:**")
#                 st.info(entry['result'].get('content', 'N/A'))
#                 if 'reflection' in entry['metrics']:
#                     st.markdown(f"**Your Reflection:** *{entry['metrics']['reflection']}*")
#                 if entry['metrics']:
#                     st.markdown(f"**Metrics:** {entry['metrics']}")
#     else:
#         st.info("Your journal is empty! Start with the C1 lab to save your first experiment.")

# # --- 5. AI ASSISTANT FUNCTION (LLM INTEGRATED - REFINED) ---
# def render_ai_assistant_sidebar():
#     """Renders the persistent AI Assistant and Progress Tracker in the sidebar."""
#     # 1. Progress Tracker
#     st.sidebar.markdown('<div class="progress-tracker">', unsafe_allow_html=True)
#     st.sidebar.markdown("#### 🎯 Your Learning Progress")
#     progress_percent = get_progress_percent()
#     st.sidebar.progress(progress_percent, text=f"**Module Complete: {progress_percent}%**")
#     lab_statuses = [f"**{k}** {v}" for k, v in st.session_state.progress.items()]
#     st.sidebar.caption(f"Status: {' | '.join(lab_statuses)}")
#     # 2. Guidance Message Display
#     guidance_message = st.session_state.get('guidance', "Welcome! Select a lab tab (C1-C5) to begin your module.")
#     st.sidebar.markdown('**Current Goal:**')
#     st.sidebar.info(guidance_message)
#     st.sidebar.markdown('</div>', unsafe_allow_html=True)
#     # 3. AI Assistant Chat Interface
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("#### 💬 AI Instructor Chatbot")
#     st.sidebar.caption("Ask simple, non-technical questions here!")
#     # Predefined System Context for the AI Assistant (Simplified)
#     SYSTEM_PROMPT = """
#     You are the **AI Instructor Assistant** for absolute beginners learning AI for Student Collaboration Analytics. 
#     Your tone must be non-technical, extremely simple, and encouraging. 
#     Your goal is to define concepts (like 'What is Synthetic Data?', 'What is Feature Importance?'), suggest next steps, and give simple troubleshooting help.
#     The user is currently working on labs C1 through C5 (Data Generation, EDA, ML Modeling, Interpretability, Intervention Optimization).
#     """
#     user_query = st.sidebar.text_input("Ask about the module, steps, or concepts:", key="assistant_query")
#     if st.sidebar.button("Ask Instructor", key="run_assistant"):
#         # --- ASSISTANT COOLDOWN CHECK ---
#         if time.time() - st.session_state.last_assistant_call < 5:
#             st.sidebar.error("Please wait 5 seconds before asking the Assistant another question.")
#             return
#         # ------------------------------------
#         if user_query:
#             # Add user message to history
#             st.session_state.assistant_chat_history.append({"role": "user", "content": user_query})
#             # 1. Construct the message list for the LLM
#             messages = [{"role": "system", "content": SYSTEM_PROMPT}] 
#             messages.extend(st.session_state.assistant_chat_history[-4:]) # Last 4 messages for context
#             # Temporarily set current_tab to a non-lab value to suppress log messages in main panel
#             temp_current_tab = st.session_state.current_tab
#             st.session_state.current_tab = 'Assistant'
#             with st.spinner("Assistant is thinking..."):
#                 # 2. Call the LLM 
#                 assistant_result = llm_call_cerebras(
#                     messages=messages, 
#                     model=DEFAULT_MODEL, 
#                     max_tokens=256, 
#                     temperature=0.2 
#                 )
#             st.session_state.current_tab = temp_current_tab # Restore current tab
#             # --- UPDATE COOLDOWN TIMER ---
#             st.session_state.last_assistant_call = time.time()
#             # -----------------------------
#             if 'content' in assistant_result:
#                 response = assistant_result['content']
#             elif 'error' in assistant_result:
#                 response = f"**Assistant Error:** Sorry, I encountered a connection issue. Please check your API key or try again in a minute."
#             else:
#                 response = "I'm experiencing a service interruption. Please try again in a moment."
#             # Add assistant message to history and display
#             st.session_state.assistant_chat_history.append({"role": "assistant", "content": response})
#             st.rerun() 
#     # 4. Display Chat History
#     for message in st.session_state.assistant_chat_history:
#         if message['role'] == 'user':
#             st.sidebar.markdown(f'<div class="user-message">**You:** {message["content"]}</div>', unsafe_allow_html=True)
#         elif message['role'] == 'assistant':
#             st.sidebar.markdown(f'<div class="assistant-message">**Instructor:** {message["content"]}</div>', unsafe_allow_html=True)
#     # 5. Reset Button
#     st.sidebar.markdown("---")
#     if st.sidebar.button("Reset All Lab Progress ⚠️", type='secondary'):
#         # Clear all session state keys
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         # Initialize base state again
#         st.session_state.progress = {f'C{i}': '🔴' for i in range(1, 6)} 
#         st.session_state.journal = []
#         st.session_state.lab_results = {}
#         st.session_state.current_tab = 'Intro' # Reset to Intro
#         st.session_state.guidance = "Welcome! Select a lab tab (C1-C5) to begin your module." 
#         st.session_state.c1_step = 1 
#         st.session_state.c4_results = pd.DataFrame() 
#         st.session_state.onboarding_done = False
#         st.session_state.assistant_chat_history = [{"role": "assistant", "content": "👋 Hi there! I'm your AI Instructor for Collaboration Analytics. Ask me anything in simple terms about synthetic data, EDA, or what to do next!"}]
#         st.session_state.last_assistant_call = 0 
#         # Reset Authentication State
#         st.session_state.is_authenticated = True # <-- Reset to True
#         st.session_state.user_info = {"user_id": "DirectRunUser"} # <-- Reset user info
#         st.success("Session cleared. Please refresh the browser.")
#         st.rerun()

# # --- 6. MAIN APPLICATION ENTRY POINT ---
# def render_main_page():
#     """Renders the main Streamlit learning interface (secured content)."""
#     # 3. Onboarding Modal (Must be called early)
#     show_onboarding_modal()
#     # Final App Title (Enhanced)
#     st.markdown('<div class="title-header">Module 1: AI + Collaboration Analytics Foundations</div>', unsafe_allow_html=True)
#     st.markdown("---")
#     # Tab Titles - Added Getting Started Tab
#     tab_titles = [
#         "🧭 Getting Started", "C1: First Dataset 🚀", "C2: Role Designer 🎭", 
#         "C3: ML Modeling 🤖", "C4: Interpretability ⭐", 
#         "C5: Intervention 🎯", "📘 Learning Journal"
#     ]
#     tabs = st.tabs(tab_titles)
#     # Content Rendering
#     with tabs[0]:
#         st.session_state.current_tab = 'Intro'
#         render_getting_started()
#     with tabs[1]:
#         st.session_state.current_tab = 'C1'
#         render_lab1()
#     with tabs[2]:
#         st.session_state.current_tab = 'C2'
#         render_lab2()
#     with tabs[3]:
#         st.session_state.current_tab = 'C3'
#         render_lab3()
#     with tabs[4]:
#         st.session_state.current_tab = 'C4'
#         render_lab4()
#     with tabs[5]:
#         st.session_state.current_tab = 'C5'
#         render_lab5()
#     with tabs[6]:
#         st.session_state.current_tab = 'Journal'
#         render_learning_journal()

# # --- REMOVED AUTHENTICATION WRAPPER ---
# # The authentication_wrapper function has been removed.
# # The main page is now rendered directly.
# if __name__ == '__main__':
#     render_ai_assistant_sidebar()
#     # Directly render the main page instead of going through the auth wrapper
#     render_main_page()














import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. INITIAL SETUP ---
st.set_page_config(
    page_title="Module 1: AI for Student Teamwork",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIMPLE AESTHETIC TWEAKS ---
CUSTOM_CSS = """
<style>
    .title-header {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #1f2937;
    }
    .progress-tracker {
        background: #f9fafb;
        padding: 0.75rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.4rem 1.1rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State ---
if 'progress' not in st.session_state:
    st.session_state.progress = {f'C{i}': '🔴' for i in range(1, 6)}
    st.session_state.journal = []
    st.session_state.current_tab = 'Intro'
    st.session_state.guidance = "Welcome!"
    st.session_state.onboarding_done = False
    st.session_state.assistant_history = [
        {
            "role": "assistant",
            "content": "👋 Hi! I’m your AI Instructor. I’m here to help you understand teamwork — no prior experience needed. What would you like to know?"
        }
    ]

# --- AI Model Setup ---
AI_API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEY_NAME = "CEREBRAS_API_KEY"
DEFAULT_MODEL = "qwen-3-32b"

# --- Synthetic Data Generator (Hidden from Students) ---
def generate_educational_collab_dataset(n_groups=500):
    """Generates a rich but realistic synthetic dataset for educational use (not shown to students)."""
    np.random.seed(42)
    data = []
    for i in range(n_groups):
        group_id = f"G{i+1:03d}"
        # Core Features
        comm_freq = np.random.randint(1, 11)
        task_clarity = np.random.randint(1, 6)
        role_clarity = np.random.randint(1, 6)
        conflict_freq = np.random.randint(0, 5)
        equal_participation = np.random.rand()
        trust_score = np.random.randint(1, 6)

        # Additional Realistic Features
        meeting_frequency = np.random.randint(0, 4)  # meetings per week
        deadline_pressure = np.random.randint(1, 6)  # 1=low, 5=high
        prior_relationship = np.random.choice([0, 1], p=[0.6, 0.4])  # 0=no, 1=yes
        external_support = np.random.randint(1, 6)  # access to TA/mentor
        tech_comfort = np.random.randint(1, 6)  # comfort with collaboration tools
        goal_alignment = np.random.rand()  # how aligned are group goals?
        feedback_quality = np.random.randint(1, 6)  # quality of peer feedback
        time_zone_overlap = np.random.randint(1, 6)  # for remote teams

        # Success Logic (hidden from students)
        success_score = (
            0.2 * (comm_freq / 10) +
            0.15 * (task_clarity / 5) +
            0.15 * (role_clarity / 5) +
            0.1 * (1 - conflict_freq / 5) +
            0.2 * equal_participation +
            0.1 * (trust_score / 5) +
            0.05 * (meeting_frequency / 4) +
            0.05 * (1 - deadline_pressure / 5) +
            0.05 * prior_relationship +
            0.05 * (external_support / 5) +
            0.05 * (tech_comfort / 5) +
            0.1 * goal_alignment +
            0.05 * (feedback_quality / 5) +
            0.05 * (time_zone_overlap / 5)
        )
        success = 1 if success_score > 0.6 else 0

        data.append([
            group_id, comm_freq, task_clarity, role_clarity, conflict_freq,
            equal_participation, trust_score, meeting_frequency, deadline_pressure,
            prior_relationship, external_support, tech_comfort, goal_alignment,
            feedback_quality, time_zone_overlap, success
        ])

    df = pd.DataFrame(data, columns=[
        'Group_ID', 'Communication_Frequency', 'Task_Clarity', 'Role_Clarity',
        'Conflict_Frequency', 'Equal_Participation', 'Trust_Score', 'Meeting_Frequency',
        'Deadline_Pressure', 'Prior_Relationship', 'External_Support', 'Tech_Comfort',
        'Goal_Alignment', 'Feedback_Quality', 'Time_Zone_Overlap', 'Successful_Collaboration'
    ])
    return df

# --- Utility Functions ---
def update_progress(key, status):
    st.session_state.progress[key] = status

def update_guidance(message):
    st.session_state.guidance = message

def save_to_journal(title, prompt, result, metrics=None):
    st.session_state.journal.append({
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'lab': st.session_state.current_tab,
        'title': title,
        'prompt': prompt,
        'result': result,
        'metrics': metrics or {}
    })

# --- LLM CALL (Precise Answer Only, No Internal Reasoning) ---
def llm_call_cerebras(messages, model=DEFAULT_MODEL, max_tokens=300, temperature=0.3):
    try:
        api_key = st.secrets[API_KEY_NAME]
    except KeyError:
        st.error(f"⚠️ **API Key Missing!** Please configure the **{API_KEY_NAME}** in your `secrets.toml` file.")
        return {"error": f"API Key Missing: {API_KEY_NAME}"}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system_instruction = {
        "role": "system",
        "content": (
            "You are a warm, student-friendly AI instructor. "
            "You MUST return only the final answer in clear, simple language. "
            "Do NOT include internal reasoning, analysis steps, or chain-of-thought. "
            "Do NOT mention that you are hiding your reasoning. "
            "Keep answers short, human, and non-technical."
        )
    }

    final_messages = [system_instruction] + messages

    payload = {
        "model": model,
        "messages": final_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(AI_API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

# --- ONBOARDING ---
def show_onboarding_modal():
    if not st.session_state.get("onboarding_done", False):
        st.session_state["onboarding_done"] = True
        with st.popover("✨ Welcome to the Collaboration Lab!✨", use_container_width=True):
            st.markdown("""
                ### 🌟 Your Mission:
                You’re here to learn how to help students work better together — using AI as a tool for insight.

                **Here’s how:**
                1. **C1: Understand Teamwork** — See real examples and ask the AI why teams succeed or fail.
                2. **C2: Speak Like a Teacher** — Tell the AI to explain teamwork in simple words.
                3. **C3: Spot the Problem** — Let the AI find the biggest issue in a struggling group.
                4. **C4: Find the Fix** — Learn which teacher action is most helpful.
                5. **C5: Take Action** — Write your own simple plan for a struggling group.

                Click any tab to begin!
            """)

# --- GETTING STARTED ---
def render_getting_started():
    st.session_state.current_tab = "Intro"
    st.markdown('<div class="title-header">🧭 Your Journey to Better Teamwork</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.header("💡 What This Module Is About")
    st.info("""
        Many student groups struggle — someone does all the work, no one talks, or people feel unheard.  
        This module uses AI to help you **see these problems clearly** — so you can help students before they fail.

        **You’re here to learn how to support students — not to become a data scientist.**
    """)

    # C1
    st.markdown("---")
    st.subheader("🧠 C1: Understanding Teamwork Scenarios 🚀")
    st.markdown("""
        - **Goal:** Learn what makes student teams succeed or fail — by seeing real examples.
        - **You’ll do:** Look at 3 simple group stories → Ask the AI: *“Why did this group struggle?”*
        - **You’ll learn:** That communication, fairness, and trust matter more than grades or deadlines.
    """)

    # C2
    st.markdown("---")
    st.subheader("🗣️ C2: Speak Like a Teacher 🎭")
    st.markdown("""
        - **Goal:** Make the AI explain teamwork in simple, kind, clear language.
        - **You’ll do:** Ask the AI: *“Tell a student why their group failed”* — and see how it changes.
        - **You’ll learn:** How to turn insights into helpful advice.
    """)

    # C3
    st.markdown("---")
    st.subheader("🔍 C3: Spot the Hidden Problem 🤖")
    st.markdown("""
        - **Goal:** See how AI spots the biggest reason teams fail — without you needing to guess.
        - **You’ll do:** Ask: *“What’s the #1 thing hurting this group?”*
        - **You’ll learn:** That one factor (like unequal work) often explains everything.
    """)

    # C4
    st.markdown("---")
    st.subheader("💡 C4: Find the Best Fix ⭐")
    st.markdown("""
        - **Goal:** Choose the best way to help a struggling group.
        - **You’ll do:** Pick from 3 teacher actions: *“Assign roles”*, *“Hold a check-in”*, *“Teach feedback skills.”*
        - **You’ll learn:** What intervention actually changes outcomes.
    """)

    # C5
    st.markdown("---")
    st.subheader("✅ C5: Take Action 🎯")
    st.markdown("""
        - **Goal:** Decide what you, as a teacher, would do next.
        - **You’ll do:** Write one simple thing you’d tell a struggling group.
        - **You’ll leave with:** A clear, human, actionable plan — not a model.
    """)

    st.markdown("---")
    st.success("Ready? Click **C1: Understanding Teamwork 🚀** above to begin.")

# --- C1: UNDERSTANDING TEAMWORK (NON-TECHNICAL) ---
def render_lab1():
    st.session_state.current_tab = "C1"
    st.header("C1: Understanding Teamwork Scenarios 🚀")
    st.markdown("##### **Goal:** Learn why some student teams succeed — and others fail — by seeing real examples.")

    # Generate dataset internally (not shown)
    if 'c1_result' not in st.session_state or st.session_state.c1_result is None:
        with st.spinner("Preparing realistic collaboration scenarios for you..."):
            df = generate_educational_collab_dataset(n_groups=500)
            st.session_state.c1_result = df

    with st.expander("📝 What You’ll Do", expanded=True):
        st.markdown("""
        👉 Look at 3 short stories about student groups.  
        👉 Ask the AI Instructor: *“Why did Group G001 struggle?”*  
        👉 Reflect: *“Have I seen this in my own group?”*

        **No technical jargon. Just people.**
        """)

    # Simple human-readable group stories only (no raw data)
    st.subheader("📖 Three Group Stories")
    example_data = pd.DataFrame({
        'Group': ['G001', 'G002', 'G003'],
        'What Happened': [
            "One student did all the work. Others didn’t reply to messages. They argued in meetings. Trust was low.",
            "Everyone talked often. Everyone helped. They checked in weekly. Everyone felt heard.",
            "They talked sometimes. One person led. Others helped, but didn’t speak up. Trust was okay."
        ],
        'Outcome': ['Struggled', 'Succeeded', 'Succeeded']
    })
    st.dataframe(example_data, use_container_width=True, hide_index=True)

    st.info("""
    **G001 Struggled** because:  
    ➤ One person carried the group.  
    ➤ Others didn’t speak up.  
    ➤ No one felt safe to say they were overwhelmed.

    **G002 Succeeded** because:  
    ➤ Everyone talked.  
    ➤ Everyone helped.  
    ➤ They trusted each other.

    **G003 Succeeded** because:  
    ➤ Even though one person led, others still contributed.  
    ➤ They didn’t argue.  
    ➤ People felt respected.
    """)

    # --- ASK AI INSTRUCTOR ---
    st.subheader("💬 Ask the AI Instructor: Why Did This Happen?")
    st.markdown("**Try asking questions like:**")
    st.markdown("""
    - *Why did Group G001 fail?*
    - *What’s the biggest reason teams struggle?*
    - *How can a teacher help a group like G001?*
    - *Is it better to have equal work or good communication?*
    - *Can a team succeed if no one talks?*
    """)

    question = st.text_input(
        "Type your question here — no right or wrong answers:",
        placeholder="What do you want to know about teamwork?",
        key="c1_question"
    )

    if st.button("Ask AI Instructor", type="primary", key="btn_c1_ask_ai"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        with st.spinner("AI Instructor is thinking..."):
            prompt = f"""
Answer ONLY with the final answer. Do not show your reasoning.

You are a compassionate teacher who helps students work better together.
Explain in simple, kind, everyday language why Group G001 struggled and Groups G002 and G003 succeeded.
Use these stories:
{example_data.to_string(index=False)}

Now answer this student's question: "{question}"
Keep it under 150 words. No jargon. No numbers. Just real human reasons.
"""
            result = llm_call_cerebras([{"role": "user", "content": prompt}])

        if 'content' in result:
            st.success(result['content'])
            save_to_journal("Teamwork Question", question, result)
            update_progress('C1', '🟢')
            update_guidance("✅ C1 Complete! You now see what really affects teamwork. Move to C2.")
            st.success("✅ You’ve completed C1! Jump to **C2: Speak Like a Teacher 🎭**.")
        else:
            st.error("AI Instructor couldn’t respond. Try again in a moment.")

    # --- SIMPLE CHART: BEFORE & AFTER COLLABORATION HEALTH ---
    st.subheader("📊 How AI Insights Can Help Teams Improve (Example)")
    st.markdown("This simple chart imagines how teams might improve after using insights like the ones you’re exploring.")

    mock_data = pd.DataFrame({
        'Metric': ['Communication', 'Trust', 'Equal Participation'],
        'Before AI': [30, 40, 25],
        'After AI': [75, 85, 90]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before',
        x=mock_data['Metric'],
        y=mock_data['Before AI'],
        marker_color='#FBBF77'
    ))
    fig.add_trace(go.Bar(
        name='After',
        x=mock_data['Metric'],
        y=mock_data['After AI'],
        marker_color='#34D399'
    ))

    fig.update_layout(
        title="Team Collaboration Health: Before vs. After Using AI Insights (Illustrative)",
        yaxis_title="Score (0-100)",
        barmode='group',
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- REFLECTION ---
    st.subheader("🧠 Your Reflection")
    reflection = st.text_area(
        "Have you ever been in a group like G001 or G002? What did you learn from this?",
        height=100,
        key="c1_reflection"
    )
    if st.button("Save Reflection & Complete C1", key="btn_c1_save_reflection"):
        if reflection.strip():
            save_to_journal("C1 Reflection on Teamwork", "Reflection on group dynamics", {"reflection": reflection})
            update_progress('C1', '🟢')
            update_guidance("🎉 C1 Complete! Move to C2: Speak Like a Teacher.")
            st.success("✅ You’ve completed C1! Jump to **C2: Speak Like a Teacher 🎭**.")
        else:
            st.warning("Please write something before saving.")

# --- C2: SPEAK LIKE A TEACHER ---
def render_lab2():
    st.session_state.current_tab = "C2"
    st.header("C2: Speak Like a Teacher 🎭")
    st.markdown("##### **Goal:** Make the AI explain teamwork in words students understand — not data.")

    with st.expander("📝 What You’ll Do", expanded=True):
        st.markdown("""
        👉 Ask the AI: *“Explain teamwork like you’re talking to a 15-year-old.”*  
        👉 Try different tones: kind, urgent, calm.  
        👉 See how the same idea changes when you change the voice.

        **You’re not teaching AI — you’re learning how to teach students.**
        """)

    tone_options = [
        "Kind and Calm Teacher",
        "Urgent Coach",
        "Friendly Peer",
        "Strict Professor"
    ]
    selected_tone = st.selectbox("Choose the tone you want:", tone_options, key="c2_tone")

    question = st.text_input(
        "What teamwork idea do you want explained simply?",
        placeholder="Why is equal participation important?",
        key="c2_question"
    )

    if st.button("Explain Like a Teacher", type="primary", key="btn_c2_explain"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner(f"AI is explaining in a {selected_tone.lower()} tone..."):
            prompt = f"""
Answer ONLY with the final answer. Do not show your reasoning.

You are a {selected_tone}. Explain this idea to a student who’s never thought about teamwork before:
"{question}"

Use simple words. No jargon. No numbers. No charts.
Make it feel personal — like you’re talking to a friend.
Keep it under 120 words.
"""
            result = llm_call_cerebras([{"role": "user", "content": prompt}])

        if 'content' in result:
            st.success(result['content'])
            save_to_journal("Teacher Explanation", f"Tone: {selected_tone} | Question: {question}", result)
            update_progress('C2', '🟢')
            update_guidance("✅ C2 Complete! You now know how to turn insight into empathy. Move to C3.")
            st.success("✅ You’ve completed C2! Jump to **C3: Spot the Hidden Problem 🤖**.")
        else:
            st.error("AI couldn’t respond. Try again.")

# --- C3: SPOT THE HIDDEN PROBLEM ---
def render_lab3():
    st.session_state.current_tab = "C3"
    st.header("C3: Spot the Hidden Problem 🤖")
    st.markdown("##### **Goal:** Let AI tell you what’s *really* hurting a group — without you guessing.")

    with st.expander("📝 What You’ll Do", expanded=True):
        st.markdown("""
        👉 Ask the AI: *“What’s the #1 reason teams fail?”*  
        👉 See how AI spots the one thing that changes everything.  
        👉 Realize: It’s not about grades. It’s about people.

        **You’re not looking at data. You’re listening to wisdom.**
        """)

    question = st.text_input(
        "What do you want to know about why teams fail?",
        placeholder="What’s the biggest problem in student groups?",
        key="c3_question"
    )

    if st.button("Ask AI Instructor", type="primary", key="btn_c3_top_problem"):
        if not question.strip():
            st.warning("Please ask a question.")
            return

        with st.spinner("AI is finding the most important reason..."):
            prompt = f"""
Answer ONLY with the final answer. Do not show your reasoning.

Based on real student group experiences, what is the #1 reason teams fail?
Focus on human behavior — not tools, deadlines, or grades.
Answer in one clear sentence, then explain it in 2 short sentences.
Use kind, simple language. No jargon.
The student's question was: "{question}"
"""
            result = llm_call_cerebras([{"role": "user", "content": prompt}])

        if 'content' in result:
            st.success(result['content'])
            save_to_journal("Hidden Problem Revealed", question, result)
            update_progress('C3', '🟢')
            update_guidance("✅ C3 Complete! You now know what to fix first. Move to C4.")
            st.success("✅ You’ve completed C3! Jump to **C4: Find the Best Fix ⭐**.")
        else:
            st.error("AI couldn’t respond. Try again.")

    # Simple illustrative chart (no technical metrics)
    st.subheader("📊 Imagine: How Often Does This Problem Show Up?")
    st.markdown("This simple example shows that one big issue (like unequal work) appears in many struggling groups.")

    example_counts = pd.DataFrame({
        "Issue": ["Unequal work", "Poor communication", "Low trust"],
        "Struggling Groups (Example Count)": [80, 60, 50]
    })

    fig_issue = go.Figure()
    fig_issue.add_trace(go.Bar(
        x=example_counts["Issue"],
        y=example_counts["Struggling Groups (Example Count)"],
        marker_color="#F97373"
    ))
    fig_issue.update_layout(
        title="An Example of How Often Key Problems Appear in Struggling Groups",
        yaxis_title="Number of Groups (Illustrative)",
        template="plotly_white"
    )
    st.plotly_chart(fig_issue, use_container_width=True)

# --- C4: FIND THE BEST FIX ---
def render_lab4():
    st.session_state.current_tab = "C4"
    st.header("C4: Find the Best Fix ⭐")
    st.markdown("##### **Goal:** Choose the best way to help a struggling group — based on what matters most.")

    with st.expander("📝 What You’ll Do", expanded=True):
        st.markdown("""
        👉 Read a real student group story.  
        👉 Pick the best thing a teacher could do.  
        👉 See why one action changes everything.

        **You’re not choosing a model. You’re choosing care.**
        """)

    st.markdown("""
    > **Group Story:**  
    > “We had 4 people. One did 80% of the work. The others didn’t talk in meetings. No one felt comfortable saying they were overwhelmed. They got a C.”
    """)

    options = [
        "Assign clear roles at the start",
        "Hold a 10-minute check-in every week",
        "Teach students how to give feedback to each other",
        "Let them figure it out on their own"
    ]

    selected = st.radio("What’s the BEST thing a teacher could do?", options, key="c4_choice")

    if st.button("Why Is This the Best Choice?", type="primary", key="btn_c4_why_best"):
        with st.spinner("AI is explaining why..."):
            prompt = f"""
Answer ONLY with the final answer. Do not show your reasoning.

A student group failed because one person did all the work and others didn’t speak up.  
The teacher chose: "{selected}"  
Explain in 2 simple sentences why this is the best choice — and why the others are less helpful.
Use kind, human language. No jargon.
"""
            result = llm_call_cerebras([{"role": "user", "content": prompt}])

        if 'content' in result:
            st.success(result['content'])
            save_to_journal("Best Intervention", f"Choice: {selected}", result)
            update_progress('C4', '🟢')
            update_guidance("✅ C4 Complete! You now know how to act with purpose. Move to C5.")
            st.success("✅ You’ve completed C4! Jump to **C5: Take Action 🎯**.")
        else:
            st.error("AI couldn’t respond.")

# --- C5: TAKE ACTION ---
def render_lab5():
    st.session_state.current_tab = "C5"
    st.header("C5: Take Action 🎯")
    st.markdown("##### **Goal:** Write one thing you’d say to a struggling group — in your own words.")

    with st.expander("📝 What You’ll Do", expanded=True):
        st.markdown("""
        👉 Imagine a group like G001 from C1.  
        👉 Write one simple, powerful thing you’d say to them as a teacher.  
        👉 This is your takeaway. This is your impact.

        **This isn’t a lab. It’s your first step to change lives.**
        """)

    action = st.text_area(
        "What would you say to a group that’s struggling?",
        placeholder="Example: 'I noticed one person is doing most of the work. Let’s talk about how we can all share the load.'",
        height=150,
        key="c5_action"
    )

    if st.button("Save My Action Plan", type="primary", key="btn_c5_save_plan"):
        if not action.strip():
            st.warning("Please write something meaningful.")
            return

        save_to_journal("C5 Teacher Action Plan", "Student intervention plan", {"action": action})
        update_progress('C5', '🟢')
        update_guidance("🎉 CONGRATULATIONS! You’ve completed the module. You now know how to help students work better together — with care and insight.")
        st.balloons()
        st.success("✅ You’ve completed C5! You’re ready to make a difference.")
        st.info("🌟 You’ve finished the entire module. Thank you for caring about student teamwork!")

# --- LEARNING JOURNAL ---
def render_learning_journal():
    st.session_state.current_tab = "Journal"
    st.header("📘 Your Learning Journal")
    st.markdown("##### Review what you’ve learned — in your own words.")

    if st.session_state.journal:
        for entry in reversed(st.session_state.journal):
            with st.expander(f"**{entry['lab']}** — {entry['title']}"):
                st.markdown(f"*{entry['timestamp']}*")
                if entry['prompt']:
                    st.markdown("**You asked / wrote:**")
                    st.markdown(f"> {entry['prompt']}")
                if 'content' in entry['result']:
                    st.markdown("**AI said:**")
                    st.markdown(f"> {entry['result']['content']}")
                if 'reflection' in entry['metrics']:
                    st.markdown("**Your reflection:**")
                    st.markdown(f"> {entry['metrics']['reflection']}")
                if 'action' in entry['metrics']:
                    st.markdown("**Your plan:**")
                    st.markdown(f"> {entry['metrics']['action']}")
    else:
        st.info("Your journal is empty. Start with C1 to save your first insight.")

# --- AI ASSISTANT SIDEBAR ---
def render_ai_assistant_sidebar():
    st.sidebar.markdown('<div class="progress-tracker">', unsafe_allow_html=True)
    st.sidebar.markdown("#### 🎯 Your Progress")
    progress_percent = sum(1 for s in st.session_state.progress.values() if s == '🟢') * 20
    st.sidebar.progress(progress_percent, text=f"Module Complete: {progress_percent}%")
    for lab, status in st.session_state.progress.items():
        st.sidebar.caption(f"{lab}: {status}")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 💬 AI Instructor Chat")
    st.sidebar.caption("Ask anything — no question is too simple.")

    for msg in st.session_state.assistant_history:
        if msg['role'] == 'user':
            st.sidebar.markdown(
                f'<div style="background:#fff9e0; padding:10px; border-radius:8px; margin:5px 0;">'
                f"**You:** {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f'<div style="background:#e0f7fa; padding:10px; border-radius:8px; margin:5px 0;">'
                f"**Instructor:** {msg['content']}</div>",
                unsafe_allow_html=True
            )

    question = st.sidebar.text_input("Ask me anything:", key="sidebar_question")
    if st.sidebar.button("Ask Instructor", key="btn_sidebar_ask_instructor"):
        if question.strip():
            st.session_state.assistant_history.append({"role": "user", "content": question})
            with st.sidebar.spinner("Thinking..."):
                prompt = f"""
Answer ONLY with the final answer. Do not show your reasoning.

You are a kind, patient AI Instructor helping teachers understand student teamwork.
Answer this in 1–2 short, simple sentences. No jargon. No numbers. Only human care.
Question: {question}
"""
                result = llm_call_cerebras([{"role": "user", "content": prompt}])
                if 'content' in result:
                    st.session_state.assistant_history.append({"role": "assistant", "content": result['content']})
                else:
                    st.session_state.assistant_history.append(
                        {"role": "assistant", "content": "I'm having trouble connecting right now — but I’m here when you're ready."}
                    )
            st.rerun()

# --- MAIN ---
def render_main_page():
    show_onboarding_modal()
    st.markdown('<div class="title-header">Module 1: AI for Student Teamwork</div>', unsafe_allow_html=True)
    st.markdown(f"> {st.session_state.guidance}")
    st.markdown("---")

    tabs = st.tabs([
        "🧭 Getting Started",
        "C1: Understanding Teamwork 🚀",
        "C2: Speak Like a Teacher 🎭",
        "C3: Spot the Hidden Problem 🤖",
        "C4: Find the Best Fix ⭐",
        "C5: Take Action 🎯",
        "📘 Learning Journal"
    ])

    with tabs[0]:
        render_getting_started()
    with tabs[1]:
        render_lab1()
    with tabs[2]:
        render_lab2()
    with tabs[3]:
        render_lab3()
    with tabs[4]:
        render_lab4()
    with tabs[5]:
        render_lab5()
    with tabs[6]:
        render_learning_journal()

if __name__ == '__main__':
    render_ai_assistant_sidebar()
    render_main_page()



