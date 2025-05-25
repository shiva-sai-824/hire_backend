#!/bin/bash

echo "Creating project structure for Hiring Copilot AI..."

# Create directories
mkdir -p core
mkdir -p models
mkdir -p services
mkdir -p api
mkdir -p utils

# --- .env ---
echo "Creating .env file..."
cat << 'EOF' > .env
SUPABASE_URL="YOUR_SUPABASE_URL"
SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
SUPABASE_SERVICE_ROLE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
HF_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"

# Optional: For Hugging Face NER model
HF_NER_MODEL_ID="dslim/bert-base-NER"
# Or for a more capable model if you have access / it exists for your task:
# HF_RESUME_PARSER_MODEL_ID="your-fine-tuned-resume-parser"

# Default LLM for Groq
GROQ_DEFAULT_MODEL="llama3-8b-8192" # Or mixtral-8x7b-32768

# For core/security.py API Key
INTERNAL_API_KEY="your_strong_internal_api_key_for_fastapi_if_needed"
EOF

# --- core/config.py ---
echo "Creating core/config.py..."
cat << 'EOF' > core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Hiring Copilot AI"
    PROJECT_VERSION: str = "0.1.0"

    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_DEFAULT_MODEL: str = os.getenv("GROQ_DEFAULT_MODEL", "llama3-8b-8192")

    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN")
    HF_NER_MODEL_ID: str = os.getenv("HF_NER_MODEL_ID", "dslim/bert-base-NER")
    # HF_RESUME_PARSER_MODEL_ID: str = os.getenv("HF_RESUME_PARSER_MODEL_ID")

    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "default_fallback_key_if_not_in_env")


settings = Settings()

if not all([settings.SUPABASE_URL, settings.SUPABASE_KEY, settings.SUPABASE_SERVICE_ROLE_KEY, settings.GROQ_API_KEY, settings.HF_API_TOKEN]):
    # Allow INTERNAL_API_KEY to be optional if not strictly used everywhere
    # if not settings.INTERNAL_API_KEY:
    #     print("Warning: INTERNAL_API_KEY is not set in .env")
    raise ValueError("One or more critical environment variables (Supabase, Groq, HF tokens) are not set. Check your .env file.")
EOF

# --- services/supabase_client.py ---
echo "Creating services/supabase_client.py..."
cat << 'EOF' > services/supabase_client.py
from supabase import create_client, Client
from core.config import settings
import logging

logger = logging.getLogger(__name__)

try:
    supabase_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    logger.info("Supabase client initialized successfully using service role key.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase_client = None # type: ignore

def get_supabase_client() -> Client:
    if supabase_client is None:
        raise RuntimeError("Supabase client is not initialized. Check configuration and logs.")
    return supabase_client

# Example: Create your tables in Supabase SQL Editor
"""
-- Candidates Table
CREATE TABLE candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    email TEXT UNIQUE,
    linkedin_url TEXT,
    github_url TEXT,
    raw_resume_text TEXT,
    extracted_skills JSONB,
    extracted_experience JSONB, -- e.g., [{"title": "SWE", "company": "Google", "duration": "2 years"}]
    extracted_education JSONB,
    location TEXT,
    open_to_contract BOOLEAN,
    current_score FLOAT, -- Can be specific to a job req
    screening_questions JSONB, -- Store AI generated questions
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job Requisitions (Optional, but useful for context)
CREATE TABLE job_requisitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    criteria JSONB, -- Store structured criteria parsed from NLP search
    recruiter_id UUID, -- REFERENCES auth.users(id), -- If you use Supabase Auth
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Outreach Logs
CREATE TABLE outreach_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates(id),
    job_requisition_id UUID REFERENCES job_requisitions(id),
    message_template TEXT,
    generated_message TEXT,
    status TEXT DEFAULT 'pending', -- e.g., pending, sent, failed
    sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Function to automatically update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_candidates_timestamp
BEFORE UPDATE ON candidates
FOR EACH ROW
EXECUTE PROCEDURE trigger_set_timestamp();

-- Example PostgreSQL function for skill distribution
CREATE OR REPLACE FUNCTION get_skill_distribution()
RETURNS TABLE(skill TEXT, count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT s.skill_name, COUNT(*) AS occurrences
    FROM candidates c, jsonb_array_elements_text(c.extracted_skills) AS s(skill_name)
    GROUP BY s.skill_name
    ORDER BY occurrences DESC
    LIMIT 10; -- Top 10 skills
END;
$$ LANGUAGE plpgsql;
"""
EOF

# --- services/groq_service.py ---
echo "Creating services/groq_service.py..."
cat << 'EOF' > services/groq_service.py
from groq import Groq
from core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

class GroqService:
    def __init__(self, api_key: str = settings.GROQ_API_KEY, model: str = settings.GROQ_DEFAULT_MODEL):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"GroqService initialized with model: {self.model}")

    def generate_completion(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", max_tokens=1024, temperature=0.7, json_mode=False) -> str | None:
        try:
            request_params = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode: # Some models support a response_format for JSON
                 request_params["response_format"] = {"type": "json_object"}


            chat_completion = self.client.chat.completions.create(**request_params)
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API request failed: {e}")
            return None

groq_service = GroqService()

# --- Specific use case functions ---

def parse_search_query_to_criteria(natural_language_query: str) -> str | None:
    system_prompt = """
    You are an expert recruitment assistant. Your task is to parse a natural language job search query
    and convert it into a structured JSON object representing the search criteria.
    The JSON should include keys like 'skills' (list of strings), 'experience_level' (e.g., 'senior', 'junior', 'mid-level'),
    'location' (string), 'role_type' (e.g., 'full-time', 'contract', 'internship'),
    'technologies' (list of strings, e.g., 'LangChain', 'RAG', 'Python'), 'role_focus' (string, e.g., "Gen-AI engineer").
    If a field is not mentioned, omit it from the JSON.
    Prioritize extracting specific technologies mentioned.
    For example, for "Find senior Gen-AI engineers with LangChain + RAG experience in Europe, open to contract work",
    the output should be something like:
    {
      "experience_level": "senior",
      "role_focus": "Gen-AI engineer",
      "technologies": ["LangChain", "RAG"],
      "location": "Europe",
      "role_type": "contract"
    }
    Only output the JSON object. Ensure the output is valid JSON.
    """
    prompt = f"Parse the following search query: \"{natural_language_query}\""
    return groq_service.generate_completion(prompt, system_prompt=system_prompt, json_mode=True, temperature=0.2)


def generate_screening_questions(job_description: str, candidate_skills: list[str], num_questions: int = 3) -> str | None:
    system_prompt = f"""
    You are an AI assistant that generates insightful screening questions for job candidates.
    Based on the job description and the candidate's key skills, generate {num_questions} relevant questions.
    The questions should help assess the candidate's depth of knowledge and practical experience.
    Format the output as a JSON list of strings.
    Example: ["Question 1?", "Question 2?", "Question 3?"]
    Only output the JSON list. Ensure the output is valid JSON.
    """
    prompt = f"""
    Job Description:
    ---
    {job_description}
    ---
    Candidate's Key Skills: {', '.join(candidate_skills)}
    ---
    Generate {num_questions} screening questions.
    """
    return groq_service.generate_completion(prompt, system_prompt=system_prompt, json_mode=True, temperature=0.5)

def generate_personalized_outreach(job_title: str, company_name: str, candidate_name: str, candidate_key_skills: list[str], job_highlights: str) -> str | None:
    system_prompt = """
    You are an expert talent acquisition specialist crafting compelling and personalized outreach messages.
    Keep the tone professional, enthusiastic, and concise. Highlight why the candidate is a good fit
    and what's exciting about the role/company.
    The output should be the email body text only.
    Do not include any preamble like "Here's the draft:" or "Subject: ...". Just the body.
    """
    prompt = f"""
    Draft an outreach email to {candidate_name} for the {job_title} role at {company_name}.
    Candidate's key skills: {', '.join(candidate_key_skills)}.
    Key highlights of the job/company: {job_highlights}.

    Hi {candidate_name},

    I came across your profile and was very impressed with your experience in [mention 1-2 specific skills from candidate_key_skills that match the job].

    At {company_name}, we're currently looking for a {job_title} to join our innovative team. This role focuses on [mention 1-2 job_highlights or connect to candidate skills]. We believe your background in [mention another skill/experience] could be a great asset.

    Would you be open to a brief chat next week to explore this opportunity further?

    Best regards,
    [Recruiter Name/Hiring Team]
    {company_name}
    """
    return groq_service.generate_completion(prompt, system_prompt=system_prompt, temperature=0.7, max_tokens=500)
EOF

# --- services/hf_service.py ---
echo "Creating services/hf_service.py..."
cat << 'EOF' > services/hf_service.py
from huggingface_hub import InferenceClient
from core.config import settings
import logging

# from services.groq_service import groq_service # Uncomment if using Groq fallback for skills
# import json # Uncomment if using Groq fallback for skills

logger = logging.getLogger(__name__)

class HuggingFaceService:
    def __init__(self, api_token: str = settings.HF_API_TOKEN):
        if not api_token:
            raise ValueError("HF_API_TOKEN is not set.")
        try:
            self.client = InferenceClient(token=api_token)
            logger.info("HuggingFace InferenceClient initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace InferenceClient: {e}")
            self.client = None


    def extract_entities_ner(self, text: str, model_id: str = settings.HF_NER_MODEL_ID) -> list | None:
        """
        Uses a Token Classification model (NER) to extract entities.
        """
        if not self.client:
            logger.error("HuggingFace client not initialized. Cannot perform NER.")
            return None
        if not model_id:
            logger.warning("HF_NER_MODEL_ID not specified, NER extraction might not work as expected.")
            return None
        try:
            # The output format depends heavily on the model.
            # dslim/bert-base-NER output:
            # [{'entity_group': 'PER', 'score': 0.99, 'word': 'John Doe', 'start': 0, 'end': 8}]
            response = self.client.token_classification(text, model=model_id)
            return response
        except Exception as e:
            logger.error(f"Hugging Face API request for NER failed (model: {model_id}): {e}")
            return None

hf_service = HuggingFaceService()

def extract_skills_from_text_hf(text: str) -> list[str]:
    """
    This is a simplified skill extractor using a generic NER model.
    A production system would need a model fine-tuned for resume skill extraction
    or more sophisticated post-processing of NER results.
    For dslim/bert-base-NER, 'MISC' or 'ORG' (if tech companies) might sometimes catch skills.
    This needs careful tuning and model selection.
    """
    if not hf_service or not hf_service.client:
        logger.warning("HF service not available for skill extraction.")
        return []
        
    # Limit text to avoid overly long inputs for NER models if they have strict limits
    # Common practice for some models is ~512 tokens, but this depends on the model.
    # Let's assume text input might be long, so we'll truncate it for this example.
    # A better approach would be to chunk the text if it's too long.
    max_text_length = 1000 # characters, adjust as needed
    truncated_text = text[:max_text_length] if len(text) > max_text_length else text


    entities = hf_service.extract_entities_ner(truncated_text)
    if not entities:
        return []

    skills = set()
    logger.debug(f"HF NER Entities for skill extraction: {entities}")

    # This logic is HIGHLY dependent on your chosen NER model's entity labels.
    # For dslim/bert-base-NER, it doesn't have a specific 'SKILL' entity.
    # We might look for 'MISC' or 'ORG' (for technologies/frameworks often part of org names)
    # This is a very crude way and needs a specialized model for good results.
    potential_skill_entities = ['MISC'] # Add 'ORG' if you see relevant tech in it
                                       # Or if you fine-tune a model, use your 'SKILL' label.

    for entity in entities:
        entity_group = entity.get('entity_group')
        word = entity.get('word')

        if entity_group and word:
            if entity_group in potential_skill_entities:
                # Basic filtering:
                # - Avoid single characters or very short words (unless they are known acronyms like 'C#')
                # - Avoid common words or titles that might be misclassified.
                # - Normalize: lowercase, strip whitespace.
                # This needs a proper deny-list or more intelligent filtering.
                normalized_word = word.strip().lower()
                if len(normalized_word) > 1 and normalized_word not in ['inc', 'llc', 'ltd', 'corp', 'gmbh', 'mr', 'ms', 'dr']:
                    skills.add(normalized_word)
            # Example: if 'ORG' sometimes catches tech like 'TensorFlow' (as an org)
            # elif entity_group == 'ORG':
            #     normalized_word = word.strip().lower()
            #     if normalized_word in ['python', 'java', 'tensorflow', 'pytorch', 'langchain', 'react']: # Example known tech
            #         skills.add(normalized_word)


    # Fallback or complementary: Use Groq to identify skills if HF NER is not specialized.
    # This can be slower and more expensive.
    # if not skills: # Or always supplement
    #     logger.info("Attempting skill extraction with Groq as fallback/supplement.")
    #     groq_prompt = f"Extract a list of all technical skills, programming languages, tools, and frameworks from the following text. Return as a JSON list of unique strings. Text: {text[:2000]}"
    #     groq_system_prompt = "You are an expert in identifying technical skills from text. Only output a JSON list of strings (e.g., [\"python\", \"java\", \"machine learning\"]). Ensure the output is valid JSON."
    #     extracted_skills_groq_str = groq_service.generate_completion(groq_prompt, system_prompt=groq_system_prompt, json_mode=True, temperature=0.3)
    #     try:
    #         if extracted_skills_groq_str:
    #             parsed_skills_groq = json.loads(extracted_skills_groq_str)
    #             if isinstance(parsed_skills_groq, list):
    #                  skills.update([str(skill).lower().strip() for skill in parsed_skills_groq if isinstance(skill, str) and skill.strip()])
    #     except json.JSONDecodeError:
    #         logger.warning(f"Failed to parse skills from Groq response: {extracted_skills_groq_str}")

    logger.info(f"Extracted skills using HF NER (and potentially Groq): {list(skills)}")
    return list(skills)
EOF

# --- services/resume_parser.py ---
echo "Creating services/resume_parser.py..."
cat << 'EOF' > services/resume_parser.py
import PyPDF2
import docx # python-docx
from io import BytesIO
import logging
from services.hf_service import extract_skills_from_text_hf
from services.groq_service import groq_service # if you want Groq to parse structure
import json

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or "" # Add "or ''" to handle None
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PyPDF2 PdfReadError: {e} - file might be corrupted or encrypted.")
        return "" # Return empty string or raise custom error
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
    return text

def extract_text_from_docx(file_bytes: BytesIO) -> str:
    try:
        doc = docx.Document(file_bytes)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def parse_resume_content(file_name: str, file_content: bytes) -> dict:
    raw_text = ""
    file_like_object = BytesIO(file_content)

    if file_name.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_like_object)
    elif file_name.lower().endswith((".docx")): # Handle .doc too if possible, though python-docx is for .docx
        raw_text = extract_text_from_docx(file_like_object)
    elif file_name.lower().endswith(".txt"):
        try:
            raw_text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                raw_text = file_content.decode('latin-1') # Fallback
            except Exception as e:
                logger.error(f"Could not decode text for file {file_name}: {e}")
                raw_text = "Error: Could not decode file content."
    else:
        logger.warning(f"Unsupported file type for direct text extraction: {file_name}. Will attempt generic decode.")
        try:
            raw_text = file_content.decode('utf-8', errors='ignore')
        except Exception:
             raw_text = "Error: Could not decode file content for unknown type."


    if not raw_text.strip():
        logger.warning(f"No text extracted from {file_name}")
        return {"raw_text": "", "skills": [], "experience": [], "education": []}

    # --- Skill Extraction ---
    # Primarily using Hugging Face NER, possibly supplemented by Groq (see hf_service.py)
    extracted_skills = extract_skills_from_text_hf(raw_text)


    # --- Structural Parsing (Experience, Education) using Groq ---
    # This is complex. For robust parsing, fine-tuned models or very specific prompts are needed.
    # We'll use a Groq prompt that asks for JSON output.
    # Limit raw_text sent to LLM to avoid excessive token usage/cost.
    text_for_llm_parsing = raw_text[:3500] # Adjust as needed

    structured_data_prompt = f"""
    Analyze the following resume text and extract structured information about work experience and education.
    For work experience, identify job title, company name, and approximate duration (e.g., "2 years", "Jan 2020 - Present").
    For education, identify degree, institution, and graduation year (if available).
    Return the result as a JSON object with two keys: "experience" (a list of objects) and "education" (a list of objects).
    If information for a section is not found or ambiguous, return an empty list for that section.
    Prioritize accuracy. Do not invent information.

    Resume Text:
    ---
    {text_for_llm_parsing}
    ---
    Expected JSON format example:
    {{
      "experience": [
        {{"title": "Software Engineer", "company": "Tech Solutions Inc.", "duration": "Jan 2020 - Present"}},
        {{"title": "Intern", "company": "Startup X", "duration": "Jun 2019 - Aug 2019"}}
      ],
      "education": [
        {{"degree": "B.S. Computer Science", "institution": "State University", "graduation_year": "2019"}}
      ]
    }}
    Only output the JSON object. Ensure the output is valid JSON.
    """
    logger.info("Attempting structured data parsing with Groq...")
    structured_data_str = groq_service.generate_completion(
        structured_data_prompt,
        system_prompt="You are an expert resume parser. Output only valid JSON that strictly adheres to the requested format.",
        max_tokens=2048, # May need more for complex resumes
        temperature=0.2, # Lower temperature for more factual, less creative parsing
        json_mode=True
    )

    parsed_experience = []
    parsed_education = []
    if structured_data_str:
        try:
            # Basic cleanup: LLMs sometimes wrap JSON in ```json ... ```
            if structured_data_str.strip().startswith("```json"):
                structured_data_str = structured_data_str.strip()[7:]
                if structured_data_str.strip().endswith("```"):
                    structured_data_str = structured_data_str.strip()[:-3]
            
            data = json.loads(structured_data_str)
            parsed_experience = data.get("experience", [])
            parsed_education = data.get("education", [])
            if not isinstance(parsed_experience, list): parsed_experience = []
            if not isinstance(parsed_education, list): parsed_education = []
            logger.info(f"Successfully parsed structured data from Groq. Experience items: {len(parsed_experience)}, Education items: {len(parsed_education)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured resume data from Groq JSON: {e}. Response: {structured_data_str[:500]}...")
        except Exception as e: # Catch any other error during processing
            logger.error(f"An unexpected error occurred during Groq structured data parsing: {e}. Response: {structured_data_str[:500]}...")
    else:
        logger.warning("No structured data response from Groq for resume parsing.")


    return {
        "raw_text": raw_text,
        "skills": list(set(extracted_skills)), # Ensure unique skills
        "experience": parsed_experience,
        "education": parsed_education
    }
EOF

# --- services/outreach_service.py ---
echo "Creating services/outreach_service.py..."
cat << 'EOF' > services/outreach_service.py
from services.groq_service import generate_personalized_outreach as groq_outreach
from services.supabase_client import get_supabase_client
import logging
import json

logger = logging.getLogger(__name__)

class OutreachService:
    def __init__(self):
        self.supabase = get_supabase_client()

    async def create_and_log_outreach(
        self,
        candidate_id: str,
        candidate_name: str,
        candidate_email: str, # Added for potential direct sending
        candidate_skills: list[str],
        job_title: str,
        company_name: str,
        job_highlights: str,
        job_requisition_id: str | None = None,
        recruiter_name: str = "The Hiring Team" # Default recruiter name
    ) -> tuple[str | None, str | None]: # (message_id, generated_message_text)

        # Augment job_highlights with a call to action or company mission if desired
        # full_job_context = f"Job Highlights: {job_highlights}. We are excited about [Company Mission/Project Impact]."

        generated_message_body = groq_outreach(
            job_title=job_title,
            company_name=company_name,
            candidate_name=candidate_name,
            candidate_key_skills=candidate_skills[:3], # Use top 3-5 skills
            job_highlights=job_highlights
        )

        if not generated_message_body:
            logger.error(f"Failed to generate outreach message body for candidate {candidate_id}")
            return None, None
        
        # Construct full message (e.g. if Groq only provides body)
        # You might want Groq to generate the subject too, or have a template
        # For now, let's assume generated_message_body is the full email text.
        final_message = generated_message_body # Adjust if Groq's output needs more wrapping

        try:
            log_entry = {
                "candidate_id": candidate_id,
                "job_requisition_id": job_requisition_id,
                # "message_template": "groq_personalized_v1", # Optional: version your templates
                "generated_message": final_message,
                "status": "generated" # Initial status
            }
            response = self.supabase.table("outreach_logs").insert(log_entry, returning="minimal").execute() # Changed to 'minimal'

            # After insert, if you need the ID, you'd typically get it from the response.
            # Supabase Python client's insert often doesn't return the full new row by default unless specified with returning="representation"
            # For this example, we'll assume the operation was successful if no error.
            # A robust way is to fetch the latest log for this candidate/job if ID is needed immediately and not returned.
            # For simplicity, we'll skip returning the log ID here, or assume it can be queried later.
            # If `returning="representation"` was used:
            # message_id = response.data[0]['id'] if response.data else None

            message_id = "logged_but_id_not_retrieved_in_this_step" # Placeholder
            logger.info(f"Outreach message generated and logged for candidate {candidate_id}.")
            return message_id, final_message
        except Exception as e:
            logger.error(f"Failed to log outreach message for candidate {candidate_id}: {e}")
            return None, final_message # Return message even if logging fails, so it can be reviewed

    # Placeholder for actually sending the email
    # async def send_email_via_service(self, to_email: str, subject: str, body: str, outreach_log_id: str | None):
    #     # Integration with an email service like SendGrid, AWS SES, Resend, etc.
    #     logger.info(f"SIMULATING: Email send to {to_email} for outreach log {outreach_log_id or 'N/A'}")
    #     # mail_sent_successfully = email_service.send(to_email, subject, body)
    #     mail_sent_successfully = True # Simulate success
    #
    #     if mail_sent_successfully and outreach_log_id and outreach_log_id != "logged_but_id_not_retrieved_in_this_step":
    #         try:
    #             self.supabase.table("outreach_logs").update({
    #                 "status": "sent",
    #                 "sent_at": "now()"
    #             }).eq("id", outreach_log_id).execute()
    #             logger.info(f"Outreach log {outreach_log_id} status updated to 'sent'.")
    #         except Exception as e:
    #             logger.error(f"Failed to update outreach log {outreach_log_id} status: {e}")
    #     elif not mail_sent_successfully and outreach_log_id and outreach_log_id != "logged_but_id_not_retrieved_in_this_step":
    #          self.supabase.table("outreach_logs").update({"status": "failed_to_send"}).eq("id", outreach_log_id).execute()
    #          logger.warning(f"Failed to send email for outreach log {outreach_log_id}, status updated.")
    #
    #     return mail_sent_successfully

outreach_service = OutreachService()
EOF

# --- models/request_response.py ---
echo "Creating models/request_response.py..."
cat << 'EOF' > models/request_response.py
from pydantic import BaseModel, EmailStr, Field, HttpUrl
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

class NLPSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, example="Find senior Gen-AI engineers with LangChain + RAG experience in Europe, open to contract work")

class ParsedCriteria(BaseModel):
    skills: Optional[List[str]] = Field(default_factory=list)
    experience_level: Optional[str] = None
    location: Optional[str] = None
    role_type: Optional[str] = None # e.g., 'contract', 'full-time'
    technologies: Optional[List[str]] = Field(default_factory=list)
    role_focus: Optional[str] = None # e.g., "Gen-AI engineer"
    # Add other fields Groq might extract

class CandidateProfileBase(BaseModel):
    name: Optional[str] = Field(None, example="Jane Doe")
    email: Optional[EmailStr] = Field(None, example="jane.doe@example.com")
    linkedin_url: Optional[HttpUrl] = Field(None, example="https://linkedin.com/in/janedoe")
    github_url: Optional[HttpUrl] = Field(None, example="https://github.com/janedoe")
    location: Optional[str] = Field(None, example="Berlin, Germany")
    open_to_contract: Optional[bool] = Field(None, example=True)
    notes: Optional[str] = Field(None, example="Met at conference X.")

class CandidateProfileCreate(CandidateProfileBase):
    email: EmailStr # Make email mandatory for creation for upsert logic

class CandidateProfileResponse(CandidateProfileBase):
    id: uuid.UUID
    raw_resume_text: Optional[str] = Field(None, repr=False) # Don't show full text in normal list views
    extracted_skills: Optional[List[str]] = Field(default_factory=list)
    extracted_experience: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    extracted_education: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    current_score: Optional[float] = None # Score relevant to a specific search/job
    screening_questions: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CandidateRankRequest(BaseModel):
    job_criteria: ParsedCriteria # Or a more detailed job description / job_id

class ScreeningQuestionsResponse(BaseModel):
    candidate_id: uuid.UUID
    questions: List[str]

class OutreachRequest(BaseModel):
    candidate_id: uuid.UUID
    job_title: str = Field(..., example="Senior Gen-AI Engineer")
    company_name: str = Field(..., example="Innovatech AI Solutions")
    job_highlights: str = Field(..., min_length=10, example="Work on cutting-edge RAG systems, collaborate with a world-class team.")
    job_requisition_id: Optional[uuid.UUID] = None
    recruiter_name: Optional[str] = Field("The Hiring Team", example="Alice Wonderland")

class OutreachResponse(BaseModel):
    outreach_log_id: Optional[str] # Could be UUID if your DB returns it
    candidate_id: uuid.UUID
    generated_message_preview: str # Preview of the message
    status: str # e.g., "generated_pending_send", "sent", "failed"

class DashboardInsight(BaseModel):
    label: str
    value: int

class TalentPoolInsightsResponse(BaseModel):
    total_candidates: int
    skill_distribution: List[DashboardInsight] # e.g., [{"label": "python", "value": 50}]
    location_distribution: List[DashboardInsight] # e.g., [{"label": "Europe", "value": 20}]
    # Add more insights as needed

# --- Auth Models (basic, Supabase handles more) ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    # Supabase might also return refresh_token, expires_in, etc.

class SupabaseUserResponse(BaseModel): # For /me endpoint
    id: uuid.UUID
    email: Optional[EmailStr] = None
    # Add other fields from Supabase user object you might need
    # e.g., aud: str, role: str, app_metadata: dict, user_metadata: dict
EOF

# --- models/candidate.py ---
# This file content was merged into models/request_response.py for Pydantic models
# but if you prefer a separate DB model definition (e.g. for an ORM), it would go here.
# For this setup, we'll rely on Pydantic models in request_response.py and direct Supabase interaction.
# So, this file can be minimal or removed if not using an ORM layer like SQLAlchemy.
echo "Creating models/candidate.py (can be minimal if Pydantic models are in request_response.py)..."
cat << 'EOF' > models/candidate.py
from pydantic import BaseModel, EmailStr, Field, HttpUrl
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

# This can be used for strict database model representation if needed,
# separate from API request/response models.
# Often, API models (like CandidateProfileResponse) inherit or compose these.

class CandidateDB(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: Optional[str] = None
    email: EmailStr # Email is key for upserting
    linkedin_url: Optional[HttpUrl] = None
    github_url: Optional[HttpUrl] = None
    raw_resume_text: Optional[str] = None
    extracted_skills: List[str] = Field(default_factory=list)
    extracted_experience: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_education: List[Dict[str, Any]] = Field(default_factory=list)
    location: Optional[str] = None
    open_to_contract: Optional[bool] = None
    current_score: Optional[float] = None # This might be better in a linking table if score is per-job
    screening_questions: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True # Pydantic V2 for ORM mode / from_orm
        # orm_mode = True # Pydantic V1
EOF

# --- core/security.py ---
echo "Creating core/security.py..."
cat << 'EOF' > core/security.py
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
# from jose import jwt, JWTError # If validating Supabase JWTs directly
# from datetime import datetime, timedelta
from core.config import settings
from services.supabase_client import get_supabase_client
from models.request_response import SupabaseUserResponse # Use the Pydantic model
import logging

logger = logging.getLogger(__name__)

# If you were to validate Supabase JWTs passed from frontend:
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login") # Your token URL

# Supabase typically uses its own SDK for auth on the frontend.
# For backend-to-backend or admin operations, service_role_key is used.
# If frontend calls this FastAPI backend with a user's Supabase JWT,
# you would need to validate it here. This involves:
# 1. Getting Supabase's JWKS (JSON Web Key Set) URI.
# 2. Fetching the public keys.
# 3. Using python-jose or PyJWT to decode and verify the token.
# This is more complex than simple API key auth.

# For internal services or simplifying, an API key can be used.
API_KEY_NAME = "X-API-KEY" # Standard practice
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header_auth)):
    """
    Dependency to verify a static API key.
    This is a simple auth method, good for internal services.
    """
    if not api_key:
        logger.warning("Missing API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Missing API Key"
        )
    if api_key != settings.INTERNAL_API_KEY:
        logger.warning("Invalid API Key received.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Invalid API Key"
        )
    return api_key


# Example of how you might get current Supabase user if a JWT is passed AND validated
# This function would replace `verify_api_key` for user-context endpoints
# async def get_current_active_supabase_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         # This is where you'd put Supabase JWT validation logic
#         # For example using supabase-py's auth.get_user(jwt_token)
#         supabase = get_supabase_client()
#         user_response = supabase.auth.get_user(token) # This validates the token with Supabase
#         if user_response.user:
#             # You might want to check if user is active, email verified etc.
#             return SupabaseUserResponse(id=user_response.user.id, email=user_response.user.email)
#         else:
#             logger.error(f"Token validation failed or no user found for token. Error: {user_response.error}")
#             raise credentials_exception
#     except Exception as e: # Includes errors from supabase.auth.get_user if token is invalid/expired
#         logger.error(f"JWT validation/user retrieval error: {e}")
#         raise credentials_exception

# For now, most endpoints will use service-level access via Supabase service_role_key
# or the simple X-API-KEY. If you need user-specific RLS applied via FastAPI endpoints,
# then proper Supabase JWT validation (like get_current_active_supabase_user) is essential.
# We'll use `verify_api_key` as the default for now for simplicity, assuming this backend
# might be called by a trusted frontend/service.
PROTECTED = Depends(verify_api_key)
EOF

# --- api/auth.py ---
echo "Creating api/auth.py..."
cat << 'EOF' > api/auth.py
from fastapi import APIRouter, HTTPException, Depends, status
from services.supabase_client import get_supabase_client
from models.request_response import UserCreate, UserLogin, Token, SupabaseUserResponse
# from core.security import get_current_active_supabase_user # If using JWT validation for /me
from core.security import PROTECTED # Using API Key for now
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/signup", response_model=SupabaseUserResponse, status_code=status.HTTP_201_CREATED)
async def signup_user(user_in: UserCreate):
    """
    User self-signup. Supabase handles email confirmation if enabled.
    """
    supabase = get_supabase_client()
    try:
        # Note: Supabase signup with service_role_key bypasses RLS for the signup itself,
        # but the user created will be subject to RLS policies.
        # Email confirmation flow is typically handled by Supabase.
        response = supabase.auth.sign_up({
            "email": user_in.email,
            "password": user_in.password,
            # You can add options like "data" for user_metadata here
            # "options": { "data": { "full_name": "Test User" } }
        })
        if response.user:
            logger.info(f"User signed up successfully: {response.user.email}")
            return SupabaseUserResponse(id=response.user.id, email=response.user.email)
        elif response.error:
            logger.error(f"Supabase signup error: {response.error.message}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response.error.message)
        else:
            logger.error("Unknown error during Supabase signup: No user and no error object.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unknown error occurred during signup.")
    except Exception as e:
        logger.error(f"Exception during user signup: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin):
    """
    User login, returns a Supabase session token (JWT).
    """
    supabase = get_supabase_client()
    try:
        response = supabase.auth.sign_in_with_password({
            "email": form_data.email,
            "password": form_data.password
        })
        if response.session and response.session.access_token:
            logger.info(f"User logged in: {response.user.email if response.user else 'Unknown - check Supabase response'}")
            return Token(
                access_token=response.session.access_token,
                token_type="bearer"
                # You can add other session details if needed:
                # refresh_token=response.session.refresh_token,
                # expires_in=response.session.expires_in
            )
        elif response.error:
            logger.error(f"Supabase login error: {response.error.message}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=response.error.message or "Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            logger.error("Unknown error during Supabase login: No session and no error object.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login failed due to an unknown error.")

    except Exception as e:
        logger.error(f"Exception during user login: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# This /me endpoint would ideally use Depends(get_current_active_supabase_user)
# if you expect the frontend to pass a Supabase JWT for user-specific info.
# For now, it's protected by the general API key.
@router.get("/me", response_model=SupabaseUserResponse)
async def read_users_me(
    # current_user: SupabaseUserResponse = Depends(get_current_active_supabase_user) # Ideal for user context
    api_key: str = PROTECTED # Or use this simple protection for now
):
    """
    Get current user details. This endpoint's behavior depends on the auth method.
    If using `get_current_active_supabase_user`, it would return the user associated with the JWT.
    If using `PROTECTED` (API Key), this endpoint doesn't have user context from a JWT.
    A practical implementation would require passing the JWT obtained from /login.
    """
    # This is a placeholder if not using JWT-based user context.
    # If `get_current_active_supabase_user` was used, `current_user` would be the actual user.
    # For now, we'll simulate a generic response or error if no user context.
    # raise HTTPException(status_code=400, detail="This /me endpoint requires JWT authentication. For API key usage, user context is not available here.")
    # Or, if you want to demonstrate it with a hardcoded user (NOT FOR PRODUCTION):
    # return SupabaseUserResponse(id="00000000-0000-0000-0000-000000000000", email="service@example.com")
    
    # For a real /me with JWT:
    # return current_user
    
    # Fallback if this endpoint is called with API key and no user context to return
    logger.warning("/me endpoint called with API key auth. No specific user context to return beyond auth success.")
    return SupabaseUserResponse(id="00000000-0000-0000-0000-000000000000", email="api_key_user@service.local") # Placeholder
EOF

# --- api/candidates.py ---
echo "Creating api/candidates.py..."
cat << 'EOF' > api/candidates.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, status, Query
from services.supabase_client import get_supabase_client
from services.resume_parser import parse_resume_content
from services.groq_service import generate_screening_questions
from services.outreach_service import outreach_service
from models.request_response import (
    CandidateProfileResponse, CandidateProfileCreate, ScreeningQuestionsResponse,
    OutreachRequest, OutreachResponse
)
# from models.candidate import CandidateDB # Using Pydantic models from request_response directly for DB interaction
from typing import List, Optional
import json
import logging
import uuid
from pydantic import EmailStr, HttpUrl
from core.security import PROTECTED # Use API Key protection

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload_resume", response_model=CandidateProfileResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_process_resume(
    email: EmailStr = Form(...),
    name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    open_to_contract: Optional[bool] = Form(None),
    linkedin_url: Optional[HttpUrl] = Form(None), # Pydantic will validate URL
    github_url: Optional[HttpUrl] = Form(None),  # Pydantic will validate URL
    notes: Optional[str] = Form(None),
    resume_file: UploadFile = File(...),
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()
    
    # Check if candidate with this email already exists to decide on create vs update logic for some fields
    # This is implicitly handled by upsert, but good to be aware of.
    
    logger.info(f"Processing resume for email: {email}, filename: {resume_file.filename}")
    file_content = await resume_file.read()
    if not file_content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Resume file is empty.")

    parsed_data = parse_resume_content(resume_file.filename, file_content)
    if not parsed_data.get("raw_text"): # Check if any text was extracted
        logger.warning(f"No text could be extracted from resume: {resume_file.filename} for email {email}")
        # Decide if you want to save a candidate record even if parsing fails partially or completely
        # For now, we'll proceed but log a warning. Some fields might be empty.


    candidate_payload = {
        "name": name,
        "email": email, # Key for upsert
        "linkedin_url": str(linkedin_url) if linkedin_url else None,
        "github_url": str(github_url) if github_url else None,
        "raw_resume_text": parsed_data["raw_text"],
        "extracted_skills": parsed_data["skills"],
        "extracted_experience": parsed_data["experience"],
        "extracted_education": parsed_data["education"],
        "location": location,
        "open_to_contract": open_to_contract,
        "notes": notes
    }
    # Filter out None values for fields that are optional and not provided
    candidate_payload_cleaned = {k: v for k, v in candidate_payload.items() if v is not None or k == "email"}


    try:
        # Upsert: inserts if email doesn't exist, updates if it does.
        # `on_conflict` specifies the constraint to check (email must be unique).
        # `ignore_duplicates=False` (default) means it will perform an update on conflict.
        response = supabase.table("candidates").upsert(candidate_payload_cleaned, on_conflict="email", returning="representation").execute()

        if response.data:
            created_or_updated_candidate = response.data[0]
            logger.info(f"Candidate {created_or_updated_candidate['email']} data saved/updated successfully. ID: {created_or_updated_candidate['id']}")
            return CandidateProfileResponse(**created_or_updated_candidate)
        else:
            # This case should ideally not happen with upsert if no error, but good to log
            logger.error(f"Failed to save/update candidate {email}. Supabase response: {response.error or 'No data returned'}")
            detail_msg = f"Failed to save candidate data: {response.error.message if response.error else 'Unknown error'}"
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg)

    except Exception as e:
        logger.error(f"Error saving/updating candidate {email}: {e}", exc_info=True)
        # Check for specific Supabase/Postgres errors if possible
        if "violates unique constraint" in str(e) and "email" in str(e): # Should be handled by upsert but as fallback
             raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Candidate with email {email} already exists and upsert failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing resume: {str(e)}")


@router.get("/{candidate_id}", response_model=CandidateProfileResponse)
async def get_candidate_profile(candidate_id: uuid.UUID, api_key: str = PROTECTED):
    supabase = get_supabase_client()
    try:
        response = supabase.table("candidates").select("*").eq("id", str(candidate_id)).single().execute()
        if response.data:
            return CandidateProfileResponse(**response.data)
        # .single() raises an error if not found or multiple found, caught by generic Exception
        # However, explicitly checking can be clearer. If response.data is None and no error, something is odd.
        logger.warning(f"Candidate {candidate_id} not found, or no data in response.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    except Exception as e: # Catches PostgrestAPIError (e.g. for not found with .single()) or other errors
        logger.error(f"Error fetching candidate {candidate_id}: {e}", exc_info=True)
        if "PGRST116" in str(e): # PGRST116: "The result contains 0 rows"
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching candidate: {str(e)}")


@router.post("/{candidate_id}/screening_questions", response_model=ScreeningQuestionsResponse)
async def generate_candidate_screening_questions_endpoint(
    candidate_id: uuid.UUID,
    job_description: str = Form(..., min_length=20, example="We are looking for a Python developer with FastAPI and RAG experience."),
    num_questions: int = Form(3, ge=1, le=5),
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()
    try:
        # Fetch candidate skills
        cand_response = supabase.table("candidates").select("id, extracted_skills").eq("id", str(candidate_id)).single().execute()
        if not cand_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        
        candidate_data = cand_response.data
        candidate_skills = candidate_data.get("extracted_skills")
        if not isinstance(candidate_skills, list) or not candidate_skills:
            logger.warning(f"Candidate {candidate_id} has no extracted skills. Generating generic questions.")
            candidate_skills = ["general technical ability"] # Fallback

        questions_json_str = generate_screening_questions(job_description, candidate_skills, num_questions)
        if not questions_json_str:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate screening questions from LLM.")

        try:
            questions_list = json.loads(questions_json_str)
            if not isinstance(questions_list, list) or not all(isinstance(q, str) for q in questions_list):
                raise ValueError("LLM did not return a valid JSON list of strings for questions.")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse screening questions from LLM response: '{questions_json_str}'. Error: {e}")
            # Fallback: try to use the string as a single question if it looks like one, or error out
            if len(questions_json_str) > 10 and '?' in questions_json_str: # very basic check
                 questions_list = [questions_json_str]
            else:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM returned malformed content for questions: {e}")

        # Store questions with candidate (optional, but good for record)
        update_response = supabase.table("candidates").update({"screening_questions": questions_list}).eq("id", str(candidate_id)).execute()
        if update_response.error:
             logger.error(f"Failed to save screening questions for candidate {candidate_id}: {update_response.error.message}")

        logger.info(f"Screening questions generated and saved for candidate {candidate_id}")
        return ScreeningQuestionsResponse(candidate_id=candidate_id, questions=questions_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating screening questions for {candidate_id}: {e}", exc_info=True)
        if "PGRST116" in str(e): # Not found from .single()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/outreach", response_model=OutreachResponse)
async def send_personalized_outreach_message(
    request: OutreachRequest,
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()
    try:
        # Fetch candidate details
        cand_response = supabase.table("candidates").select("id, name, email, extracted_skills").eq("id", str(request.candidate_id)).single().execute()
        if not cand_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")

        candidate = cand_response.data
        candidate_name = candidate.get("name", "Prospective Candidate") # Fallback name
        candidate_email = candidate.get("email")
        if not candidate_email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Candidate email is missing, cannot send outreach.")
        
        candidate_skills = candidate.get("extracted_skills", [])
        if not isinstance(candidate_skills, list): candidate_skills = []


        log_id, message_text = await outreach_service.create_and_log_outreach(
            candidate_id=str(request.candidate_id),
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_skills=candidate_skills,
            job_title=request.job_title,
            company_name=request.company_name,
            job_highlights=request.job_highlights,
            job_requisition_id=str(request.job_requisition_id) if request.job_requisition_id else None,
            recruiter_name=request.recruiter_name or "The Hiring Team"
        )

        if not message_text:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate outreach message content.")

        # Actual email sending would go here using a proper email service
        # For now, we simulate it and log.
        # subject = f"Opportunity: {request.job_title} at {request.company_name}"
        # email_sent_successfully = await outreach_service.send_email_via_service(candidate_email, subject, message_text, log_id)
        email_sent_successfully = True # SIMULATE SUCCESS
        current_status = "generated_pending_send"
        if email_sent_successfully:
            logger.info(f"SIMULATED: Email outreach to {candidate_email} for candidate {request.candidate_id} was successful.")
            current_status = "simulated_sent"
            # Update log status if actual sending happens here and is synchronous
            if log_id and log_id != "logged_but_id_not_retrieved_in_this_step":
                supabase.table("outreach_logs").update({"status": "sent", "sent_at": "now()"}).eq("id", log_id).execute()

        else:
            logger.warning(f"SIMULATED: Email outreach to {candidate_email} for candidate {request.candidate_id} FAILED.")
            current_status = "generation_complete_send_failed"
             # Update log status
            if log_id and log_id != "logged_but_id_not_retrieved_in_this_step":
                supabase.table("outreach_logs").update({"status": "failed_to_send"}).eq("id", log_id).execute()


        logger.info(f"Outreach message prepared for candidate {request.candidate_id}")
        return OutreachResponse(
            outreach_log_id=log_id,
            candidate_id=request.candidate_id,
            generated_message_preview=message_text[:250] + "..." if message_text and len(message_text) > 250 else message_text, # Preview
            status=current_status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in outreach process for candidate {request.candidate_id}: {e}", exc_info=True)
        if "PGRST116" in str(e): # Not found from .single()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found for outreach.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# --- Get all candidates (with pagination) ---
@router.get("", response_model=List[CandidateProfileResponse])
async def list_candidates(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()
    try:
        response = supabase.table("candidates").select("*").order("created_at", desc=True).range(skip, skip + limit - 1).execute()
        if response.data:
            return [CandidateProfileResponse(**c) for c in response.data]
        return []
    except Exception as e:
        logger.error(f"Error listing candidates: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not retrieve candidates: {e}")
EOF

# --- api/search.py ---
echo "Creating api/search.py..."
cat << 'EOF' > api/search.py
from fastapi import APIRouter, HTTPException, Depends, Query
from services.supabase_client import get_supabase_client
from services.groq_service import parse_search_query_to_criteria
from models.request_response import NLPSearchRequest, CandidateProfileResponse, ParsedCriteria
from typing import List, Optional
import json
import logging
from core.security import PROTECTED # Use API Key protection

router = APIRouter()
logger = logging.getLogger(__name__)

def map_criteria_to_supabase_filters(query_builder, criteria: ParsedCriteria):
    """
    Dynamically builds Supabase query filters based on parsed criteria.
    Uses PostgREST syntax for filtering.
    """
    filters = []

    if criteria.skills:
        # For JSONB array of strings: field @> '["skill1", "skill2"]' (contains all)
        # or field ?| array['skill1', 'skill2'] (contains any - using overlaps is easier)
        # We assume skills are stored as a text array or JSONB array of strings.
        # Using 'overlaps' (&& operator in Postgres) for "contains any of these"
        skills_lower = [s.strip().lower() for s in criteria.skills if s.strip()]
        if skills_lower:
            # query_builder = query_builder.overlaps("extracted_skills", skills_lower)
            # More general: search in raw text if skill extraction is not perfect or skills are nuanced
            # This can be slow on large datasets without proper indexing (e.g., FTS)
            for skill in skills_lower:
                 query_builder = query_builder.ilike("raw_resume_text", f"%{skill}%") # Or use FTS
            # If you are sure extracted_skills is accurate and want to use it:
            # query_builder = query_builder.overlaps("extracted_skills", skills_lower) # requires extracted_skills to be array type

    if criteria.technologies:
        tech_lower = [t.strip().lower() for t in criteria.technologies if t.strip()]
        if tech_lower:
            # Similar to skills, search in raw_resume_text or a dedicated 'extracted_technologies' field.
            # For now, combining with skills logic or treating similarly:
            for tech in tech_lower:
                query_builder = query_builder.ilike("raw_resume_text", f"%{tech}%")
            # Or if extracted_skills contains technologies:
            # query_builder = query_builder.overlaps("extracted_skills", tech_lower)


    if criteria.location:
        # query_builder = query_builder.ilike("location", f"%{criteria.location.strip()}%")
        # Broader search in resume text for location mentions
        query_builder = query_builder.ilike("raw_resume_text", f"%{criteria.location.strip()}%")


    if criteria.experience_level:
        # This is heuristic. "senior", "lead", "junior" etc. in resume text.
        # Better if experience is structured and classified.
        query_builder = query_builder.ilike("raw_resume_text", f"%{criteria.experience_level.strip()}%")

    if criteria.role_type:
        role_type_lower = criteria.role_type.lower()
        if "contract" in role_type_lower:
            query_builder = query_builder.eq("open_to_contract", True)
        # Add other role_type mappings if you have a specific field, e.g.
        # elif "full-time" in role_type_lower:
        #     query_builder = query_builder.eq("employment_type", "full-time") # Assuming such a field

    if criteria.role_focus:
        # Search for main role description keywords in resume text
        query_builder = query_builder.ilike("raw_resume_text", f"%{criteria.role_focus.strip()}%")

    return query_builder


@router.post("/people", response_model=List[CandidateProfileResponse])
async def natural_language_talent_search(
    request: NLPSearchRequest,
    limit: int = Query(20, ge=1, le=100),
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()

    # 1. Parse natural language query with Groq
    logger.info(f"Received NLP search query: \"{request.query}\"")
    criteria_json_str = parse_search_query_to_criteria(request.query)
    if not criteria_json_str:
        logger.error("LLM failed to parse search query. No criteria returned.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to parse search query using LLM.")

    try:
        # LLMs sometimes add ```json ... ```, try to strip it
        if criteria_json_str.strip().startswith("```json"):
            criteria_json_str = criteria_json_str.strip()[7:]
            if criteria_json_str.strip().endswith("```"):
                criteria_json_str = criteria_json_str.strip()[:-3]

        parsed_criteria_json = json.loads(criteria_json_str)
        search_criteria = ParsedCriteria(**parsed_criteria_json)
        logger.info(f"Parsed search criteria: {search_criteria.model_dump_json(exclude_none=True)}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse LLM response for criteria: '{criteria_json_str}'. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM returned malformed JSON for search criteria: {e}")

    # 2. Build Supabase query based on parsed criteria
    try:
        query = supabase.table("candidates").select("*")
        query = map_criteria_to_supabase_filters(query, search_criteria)

        # Execute query
        response = query.limit(limit).execute()

        if response.data:
            candidates = [CandidateProfileResponse(**candidate_data) for candidate_data in response.data]
            logger.info(f"Found {len(candidates)} candidates matching criteria.")
            return candidates
        elif response.error:
            logger.error(f"Supabase query error: {response.error.message}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database query error: {response.error.message}")
        else: # No data and no error
            logger.info("No candidates found matching criteria.")
            return []

    except Exception as e:
        logger.error(f"Error during talent search database query: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database query failed: {str(e)}")


# --- Candidate Ranking Logic ---
def score_candidate(candidate: CandidateProfileResponse, criteria: ParsedCriteria) -> float:
    score = 0.0
    if not candidate or not criteria: # Should not happen if called correctly
        return 0.0

    # Max score components
    MAX_SKILL_SCORE = 40.0
    MAX_TECH_SCORE = 30.0
    MAX_LOCATION_SCORE = 15.0
    MAX_ROLE_TYPE_SCORE = 15.0
    # MAX_EXP_LEVEL_SCORE = 10.0 # Hard to score well without structured data

    # Skill match scoring
    if criteria.skills and candidate.extracted_skills:
        # Using lowercase for case-insensitive matching
        crit_skills_lower = {s.lower().strip() for s in criteria.skills if s.strip()}
        cand_skills_lower = {s.lower().strip() for s in candidate.extracted_skills if s.strip()}
        matched_skills = crit_skills_lower.intersection(cand_skills_lower)
        if crit_skills_lower: # Avoid division by zero
            score += (len(matched_skills) / len(crit_skills_lower)) * MAX_SKILL_SCORE
    
    # Technology match scoring (can overlap with skills if not distinct)
    if criteria.technologies and candidate.extracted_skills: # Assuming tech might be in extracted_skills
        crit_tech_lower = {t.lower().strip() for t in criteria.technologies if t.strip()}
        # Assuming candidate.extracted_skills might contain these technologies
        cand_skills_as_tech_lower = {s.lower().strip() for s in candidate.extracted_skills if s.strip()}
        matched_tech = crit_tech_lower.intersection(cand_skills_as_tech_lower)
        if crit_tech_lower:
            score += (len(matched_tech) / len(crit_tech_lower)) * MAX_TECH_SCORE

    # Location match (simple substring match for now)
    if criteria.location and candidate.location:
        if criteria.location.lower().strip() in candidate.location.lower():
            score += MAX_LOCATION_SCORE
    elif criteria.location and candidate.raw_resume_text: # Fallback to resume text
        if criteria.location.lower().strip() in candidate.raw_resume_text.lower():
            score += MAX_LOCATION_SCORE * 0.5 # Lower score for text match

    # Role type match
    if criteria.role_type and "contract" in criteria.role_type.lower():
        if candidate.open_to_contract:
            score += MAX_ROLE_TYPE_SCORE
    # Add other role types if applicable and if you have a field for it

    # Experience level (very heuristic, based on keywords in raw text)
    # if criteria.experience_level and candidate.raw_resume_text:
    #     if criteria.experience_level.lower().strip() in candidate.raw_resume_text.lower():
    #         score += MAX_EXP_LEVEL_SCORE

    return round(min(score, 100.0), 2) # Cap score at 100


@router.post("/people_ranked", response_model=List[CandidateProfileResponse])
async def natural_language_talent_search_and_rank(
    request: NLPSearchRequest,
    limit: int = Query(50, ge=1, le=200), # Fetch more initially for better ranking pool
    top_n: int = Query(20, ge=1, le=50),   # Return top N after ranking
    api_key: str = PROTECTED
):
    supabase = get_supabase_client()
    logger.info(f"Received NLP ranked search query: \"{request.query}\"")
    criteria_json_str = parse_search_query_to_criteria(request.query)
    if not criteria_json_str:
        raise HTTPException(status_code=500, detail="LLM failed to parse search query for ranking.")
    
    try:
        if criteria_json_str.strip().startswith("```json"):
            criteria_json_str = criteria_json_str.strip()[7:]
            if criteria_json_str.strip().endswith("```"):
                criteria_json_str = criteria_json_str.strip()[:-3]
        parsed_criteria_json = json.loads(criteria_json_str)
        search_criteria = ParsedCriteria(**parsed_criteria_json)
        logger.info(f"Parsed search criteria for ranking: {search_criteria.model_dump_json(exclude_none=True)}")
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"LLM returned malformed JSON for ranked search criteria: {e}")

    # Initial filtering (same as /people endpoint, but potentially broader limit)
    try:
        query = supabase.table("candidates").select("*")
        # Apply some basic filters if possible to narrow down before fetching all
        # For simplicity, we fetch a larger set then filter/rank in Python.
        # A more optimized approach would do more filtering in DB.
        query = map_criteria_to_supabase_filters(query, search_criteria) # Apply filters
        
        response = query.limit(limit).execute()

        if response.error:
            logger.error(f"Supabase query error during ranking: {response.error.message}")
            raise HTTPException(status_code=500, detail=f"Database query error: {response.error.message}")
        
        if not response.data:
            logger.info("No candidates found for ranking based on initial filters.")
            return []

        candidates_found = [CandidateProfileResponse(**c) for c in response.data]
        logger.info(f"Fetched {len(candidates_found)} candidates for ranking.")
    except Exception as e:
        logger.error(f"Error fetching candidates for ranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database query failed during ranking: {str(e)}")

    # Score and rank
    for cand in candidates_found:
        cand.current_score = score_candidate(cand, search_criteria)
        logger.debug(f"Candidate {cand.id} score: {cand.current_score}")

    # Sort by score, descending
    ranked_candidates = sorted(candidates_found, key=lambda c: c.current_score or 0.0, reverse=True)
    
    logger.info(f"Ranked {len(ranked_candidates)} candidates. Returning top {top_n}.")
    return ranked_candidates[:top_n]
EOF

# --- api/dashboard.py ---
echo "Creating api/dashboard.py..."
cat << 'EOF' > api/dashboard.py
from fastapi import APIRouter, HTTPException, Depends
from services.supabase_client import get_supabase_client
from models.request_response import TalentPoolInsightsResponse, DashboardInsight
from typing import List
import logging
from core.security import PROTECTED

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/insights", response_model=TalentPoolInsightsResponse)
async def get_talent_pool_insights(api_key: str = PROTECTED):
    supabase = get_supabase_client()
    try:
        # Total candidates
        # The `count` parameter in select is for PostgREST versions that support it for exact counts.
        # For Supabase, often `head=True` with `count='exact'` in the client call works, or a direct rpc.
        # Let's try getting a count via RPC or a simpler select.
        count_response = supabase.table("candidates").select("id", count="exact").limit(0).execute()
        total_candidates = count_response.count if count_response.count is not None else 0
        logger.info(f"Total candidates from count: {total_candidates}")

        # Skill distribution - using the DB function
        # Make sure the 'get_skill_distribution' function exists in your Supabase SQL.
        skill_dist_data = []
        try:
            skill_response = supabase.rpc('get_skill_distribution', {}).execute()
            if skill_response.data:
                skill_dist_data = [DashboardInsight(label=row['skill'], value=row['count']) for row in skill_response.data]
            elif skill_response.error:
                logger.error(f"Error calling RPC get_skill_distribution: {skill_response.error.message}")
        except Exception as e:
            logger.error(f"Exception calling RPC get_skill_distribution: {e}")
            # Fallback or empty if RPC fails

        # Location distribution (example of direct aggregation if simple enough)
        # This groups by the 'location' field. For more complex location analysis, FTS or GIS might be needed.
        location_dist_data = []
        try:
            # Note: Supabase/PostgREST typically requires explicit column selection for group by.
            # A direct query might be: SELECT location, COUNT(id) as count FROM candidates GROUP BY location ORDER BY count DESC LIMIT 10;
            # This is hard to do directly with the Python client's basic group_by without RPC.
            # For simplicity, let's assume a limited set of distinct locations or use RPC.
            # Using RPC for consistency if you have a similar DB function for locations:
            # location_response = supabase.rpc('get_location_distribution', {}).execute()
            # if location_response.data:
            #    location_dist_data = [DashboardInsight(label=row['location'] or "Unknown", value=row['count']) for row in location_response.data]
            
            # If not using RPC, this query is more complex with the python client, often easier to fetch data and aggregate in python for small datasets
            # or create a dedicated view/function in postgres
            # For now, a placeholder as direct aggregation on text fields with group by can be tricky with the client lib.
            # Fetch all locations and count in Python (not efficient for large datasets)
            all_locations_resp = supabase.table("candidates").select("location").execute()
            if all_locations_resp.data:
                location_counts = {}
                for item in all_locations_resp.data:
                    loc = item.get("location", "Unknown")
                    if loc: # Ensure loc is not None
                        location_counts[loc] = location_counts.get(loc, 0) + 1
                
                sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10] # Top 10
                location_dist_data = [DashboardInsight(label=loc, value=count) for loc, count in sorted_locations]

        except Exception as e:
            logger.error(f"Error generating location distribution: {e}")


        return TalentPoolInsightsResponse(
            total_candidates=total_candidates,
            skill_distribution=skill_dist_data,
            location_distribution=location_dist_data
        )
    except Exception as e:
        logger.error(f"Error fetching dashboard insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not retrieve dashboard insights: {str(e)}")

EOF

# --- api/__init__.py ---
echo "Creating api/__init__.py..."
cat << 'EOF' > api/__init__.py
from fastapi import APIRouter
from api import auth, candidates, search, dashboard

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(candidates.router, prefix="/candidates", tags=["Candidates & Resumes"])
api_router.include_router(search.router, prefix="/search", tags=["Talent Search (PeopleGPT)"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Talent Pool Insights"])
EOF

# --- utils/misc.py ---
echo "Creating utils/misc.py..."
cat << 'EOF' > utils/misc.py
# This file can hold miscellaneous utility functions that are shared across the application.
# For example, text cleaning, date formatting, etc.

import re

def clean_text_for_llm(text: str) -> str:
    """
    Basic text cleaning to remove excessive whitespace or problematic characters
    before sending to an LLM.
    """
    if not text:
        return ""
    # Replace multiple newlines/spaces with a single one
    text = re.sub(r'\s\s+', ' ', text)
    text = re.sub(r'\n\n+', '\n', text)
    return text.strip()

def truncate_text(text: str, max_length: int = 2000) -> str:
    """
    Truncates text to a maximum length, ensuring it doesn't cut mid-word if possible.
    """
    if not text or len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    # Try to avoid cutting mid-word
    last_space = truncated.rfind(' ')
    if last_space != -1:
        return truncated[:last_space] + "..."
    return truncated + "..."

# Add other utility functions as needed.
EOF


# --- main.py ---
echo "Creating main.py..."
cat << 'EOF' > main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api import api_router
import logging
import uvicorn
import time

# Configure logging
# Define a custom log format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# You can also configure specific loggers, e.g., uvicorn.access
# logging.getLogger("uvicorn.access").handlers = [] # To disable uvicorn access logs or customize
# logging.getLogger("uvicorn.error").propagate = True


logger = logging.getLogger(__name__) # Get root logger or specific app logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    openapi_url="/api/v1/openapi.json",
    description="AI Hiring Copilot Backend"
)

# Middleware for timing requests
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - Completed in {process_time:.4f}s - Status: {response.status_code}")
    return response

# CORS (Cross-Origin Resource Sharing)
# Adjust allow_origins for production
origins = [
    "http://localhost",       # Common for local dev
    "http://localhost:3000",  # React default
    "http://localhost:8080",  # Vue default
    "http://localhost:5173",  # Vite/React default
    # Add your production frontend URL here
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"], # Use specific origins in prod
    allow_credentials=True,
    allow_methods=["*"], # Or specify methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"], # Or specify headers
)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Application startup: {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    logger.info(f"Supabase URL configured: {settings.SUPABASE_URL[:20]}...") # Log part of URL for verification
    logger.info(f"Groq Model: {settings.GROQ_DEFAULT_MODEL}")
    logger.info(f"HF NER Model: {settings.HF_NER_MODEL_ID}")
    # Check Supabase client initialization
    try:
        from services.supabase_client import get_supabase_client
        client = get_supabase_client() # This will raise RuntimeError if not init
        logger.info("Supabase client connection test successful on startup.")
        # You could do a quick test query if needed, e.g., client.table('candidates').select('id').limit(1).execute()
    except Exception as e:
        logger.error(f"Supabase client initialization check FAILED on startup: {e}", exc_info=True)
        # Depending on severity, you might want to prevent app from starting or run in degraded mode.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown.")


@app.get("/", tags=["Root"])
async def read_root():
    """Welcome endpoint for the Hiring Copilot API."""
    return {"message": f"Welcome to {settings.PROJECT_NAME} API v{settings.PROJECT_VERSION}"}

# Include the main API router
app.include_router(api_router, prefix="/api/v1")

# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught: Status Code: {exc.status_code}, Detail: {exc.detail}, Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "path": str(request.url.path)},
        headers=exc.headers,
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # This catches any unhandled exceptions
    logger.error(f"Unhandled global exception: {exc}", exc_info=True) # exc_info=True logs traceback
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_type": type(exc).__name__}
    )


if __name__ == "__main__":
    # For development, uvicorn.run is convenient.
    # For production, consider Gunicorn as a process manager for Uvicorn workers.
    # Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    uvicorn.run(
        "main:app",
        host="0.0.0.0", # Listen on all available network interfaces
        port=8000,
        reload=True,     # Enable auto-reload for development
        log_level="info" # Uvicorn's own log level
    )
EOF

# --- requirements.txt ---
echo "Creating requirements.txt..."
cat << 'EOF' > requirements.txt
fastapi
uvicorn[standard]
supabase
python-dotenv
groq
huggingface_hub
python-multipart
pypdf2
python-docx
pydantic[email]
python-jose[cryptography] # For JWT handling if you implement Supabase JWT validation
# psycopg2-binary # If you need to connect to Supabase DB directly for complex queries not covered by supabase-py
# celery # For background tasks
# redis # For Celery broker/backend
# flower # For Celery monitoring
EOF

echo "Project structure and files created successfully!"
echo "Next steps:"
echo "1. Fill in your API keys and Supabase details in the .env file."
echo "2. Create the tables in your Supabase project using the SQL in services/supabase_client.py."
echo "3. Create a Python virtual environment: python -m venv venv"
echo "4. Activate it: source venv/bin/activate (or venv\\Scripts\\activate on Windows)"
echo "5. Install dependencies: pip install -r requirements.txt"
echo "6. Run the application: python main.py (or uvicorn main:app --reload)"