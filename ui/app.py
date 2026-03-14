import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import io
import re
import PyPDF2
import docx

# Page Configuration
st.set_page_config(page_title="LegalMind", page_icon="⚖️", layout="wide")

# CUSTOM CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
    .app-title { font-weight: 700; font-size: 2.25rem; margin-bottom: 0.25rem; color: var(--text-color); }
    .app-subtitle { font-weight: 400; font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.8; color: var(--text-color); }
    .stButton>button {
        background-color: #3b82f6 !important; color: white !important; border-radius: 0.5rem !important;
        border: none !important; font-weight: 600 !important; padding: 0.75rem 1.5rem !important;
        width: 100%; transition: transform 0.2s ease;
    }
    .stButton>button:hover { background-color: #2563eb !important; transform: translateY(-2px); }
    .metric-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background-color: var(--secondary-background-color); padding: 1rem; border-radius: 0.75rem;
        border: 1px solid var(--border-color); box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-title { font-size: 0.9rem; font-weight: 600; color: var(--text-color); opacity: 0.8; margin-bottom: 0.25rem; text-transform: uppercase;}
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .risk-card {
        background-color: var(--secondary-background-color); padding: 1.5rem; border-radius: 0.75rem;
        border: 1px solid var(--border-color); margin-bottom: 1.5rem; transition: transform 0.2s ease;
    }
    .risk-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
    .risk-high { border-left: 6px solid #ef4444; }
    .risk-medium { border-left: 6px solid #f59e0b; }
    .risk-low { border-left: 6px solid #10b981; }
    .clause-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-color); display: flex; justify-content: space-between; align-items: center; }
    .clause-text { font-size: 1rem; line-height: 1.6; margin-bottom: 1rem; padding: 1rem; background-color: rgba(128, 128, 128, 0.05); border-radius: 0.5rem; color: var(--text-color); }
    .warning-box { font-size: 0.9rem; padding: 0.75rem; border-radius: 0.5rem; background-color: var(--background-color); border: 1px dashed var(--border-color); color: var(--text-color); }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/analyze"

# HELPER FUNCTIONS 
def extract_text_from_file(uploaded_file):
    #Hàm bóc tách text từ các định dạng file khác nhau
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'txt':
            return uploaded_file.getvalue().decode("utf-8")
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""
    return ""

def highlight_risky_keywords(text, risk_level):
    #Hàm highlight bôi màu từ khóa nguy hiểm 
    if risk_level == "LOW":
        return text 
        
    # Danh sách từ khóa từ risk_logic.py
    high_keywords = [
        "immediate termination", "without cause", "at will", "without notice",
        "unlimited duration", "perpetual", "no exceptions", "all information",
        "unlimited liability", "all damages", "gross negligence", "sole discretion",
        "non-assignable", "without consent"
    ]
    
    medium_keywords = [
        "30 days notice", "notice period", "breach", "default",
        "5 years", "proprietary information", "trade secrets",
        "reasonable costs", "third party claims", "defend and hold harmless",
        "prior written consent", "affiliate", "successor"
    ]
    
    highlighted_text = text
    
    # Tô màu Đỏ cho các từ khóa High Risk
    for kw in high_keywords:
        pattern = re.compile(r'\b(' + kw + r')\b', re.IGNORECASE)
        mark_style_high = 'background-color: #fecaca; color: #991b1b; padding: 2px 4px; border-radius: 4px; font-weight: 600;'
        highlighted_text = pattern.sub(f'<mark style="{mark_style_high}">\\1</mark>', highlighted_text)

    # Tô màu Vàng cho các từ khóa Medium Risk
    for kw in medium_keywords:
        pattern = re.compile(r'\b(' + kw + r')\b', re.IGNORECASE)
        mark_style_medium = 'background-color: #fde68a; color: #92400e; padding: 2px 4px; border-radius: 4px; font-weight: 600;'
        highlighted_text = pattern.sub(f'<mark style="{mark_style_medium}">\\1</mark>', highlighted_text)
        
    return highlighted_text

st.markdown('<div class="app-title">⚖️ Automated Contract Review & Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Automated risk assessment utilizing fine-tuned Legal-BERT model</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1.8]) 

with col1:
    st.markdown("### 📄 Input Document")
    input_method = st.radio("Select input method", ("Upload File (.pdf, .docx, .txt)", "Paste Text"), label_visibility="collapsed")
    
    contract_text = ""
    if input_method == "Paste Text":
        contract_text = st.text_area("Contract Clauses:", height=300, 
                                     placeholder="Receiving Party shall keep all Confidential Information strictly confidential...",
                                     label_visibility="collapsed")
    else:
        # Hỗ trợ thêm PDF và DOCX
        uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            contract_text = extract_text_from_file(uploaded_file)
            st.success(f"File **{uploaded_file.name}** loaded successfully!")
            with st.expander("Preview Extracted Text"):
                st.text(contract_text[:1000] + ("..." if len(contract_text) > 1000 else "")) # Chỉ preview 1000 ký tự đầu cho đỡ lag
                
    analyze_btn = st.button("Analyze Risk")

with col2:
    if analyze_btn:
        if not contract_text.strip():
            st.warning("⚠️ Please provide contract text before analysis.")
        else:
            with st.spinner("AI is parsing and analyzing the document..."):
                try:
                    payload = {"contract_text": contract_text}
                    response = requests.post(API_URL, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        total_clauses = data.get("total_clauses", 0)
                        
                        high_risk_count = sum(1 for c in results if c.get("risk_level", "").upper() == "HIGH")
                        medium_risk_count = sum(1 for c in results if c.get("risk_level", "").upper() == "MEDIUM")
                        low_risk_count = sum(1 for c in results if c.get("risk_level", "").upper() == "LOW")

                        st.markdown("### 📈 Executive Summary")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: st.markdown(f'<div class="metric-container"><div class="metric-title">Total Clauses</div><div class="metric-value" style="color: #3b82f6;">{total_clauses}</div></div>', unsafe_allow_html=True)
                        with m2: st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #ef4444;"><div class="metric-title">High Risk</div><div class="metric-value" style="color: #ef4444;">{high_risk_count}</div></div>', unsafe_allow_html=True)
                        with m3: st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #f59e0b;"><div class="metric-title">Medium Risk</div><div class="metric-value" style="color: #f59e0b;">{medium_risk_count}</div></div>', unsafe_allow_html=True)
                        with m4: st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #10b981;"><div class="metric-title">Low Risk</div><div class="metric-value" style="color: #10b981;">{low_risk_count}</div></div>', unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        df = pd.DataFrame(results)

                        if not df.empty:
                            with st.expander("📊 View Label Distribution Chart", expanded=False):
                                label_counts = df['label'].value_counts().reset_index()
                                label_counts.columns = ['Label', 'Count']
                                fig = px.pie(label_counts, names='Label', values='Count', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                                fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fig, use_container_width=True)

                        st.markdown("---")
                        st.markdown("### 📋 Detailed Clause Analysis")
                        
                        for i, clause in enumerate(results):
                            risk_raw = clause.get("risk_level", "LOW")
                            risk = risk_raw.upper()
                            label = clause.get("label", "Unknown")
                            conf = clause.get("confidence", 0.0)
                            
                            if risk == "HIGH":
                                risk_class = "risk-high"
                                risk_color = "#ef4444"
                                icon = "🔴"
                            elif risk == "MEDIUM":
                                risk_class = "risk-medium"
                                risk_color = "#f59e0b"
                                icon = "🟡"
                            else:
                                risk_class = "risk-low"
                                risk_color = "#10b981"
                                icon = "🟢"
                            
                            # HÀM HIGHLIGHT
                            display_text = highlight_risky_keywords(clause['clause_text'], risk)
                            
                            card_html = f"""
                            <div class="risk-card {risk_class}">
                                <div class="clause-header">
                                    <span>{icon} Clause {i+1} | <strong>{label}</strong></span>
                                    <span style="font-size: 0.85rem; opacity: 0.7;">Conf: {conf:.2f}</span>
                                </div>
                                <div class="clause-text">
                                    "{display_text}"
                                </div>
                                <div class="warning-box">
                                    <strong style="color: {risk_color};">Risk Level: {risk}</strong><br>
                                    <span style="margin-top: 5px; display: inline-block;">💡 <strong>Recommendation:</strong> {clause['warning_message']}</span>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("Export Report")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download Results as CSV", data=csv, file_name='contract_risk_report.csv', mime='text/csv', type="primary")
                            
                    else:
                        st.error(f"Server Error: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Connection failed. Backend server required (`uvicorn main:app --reload`).")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("Upload a PDF/Word file or paste contract text, then click **Analyze Risk** to begin.")