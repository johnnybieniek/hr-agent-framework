"""
This module defines prompt templates for the multi‑agent HR workflow.

"""

HR_PROMPT: str = """
You are the HR Agent. Read the input message and return ONLY a single JSON object for the HR database with this exact schema and order:

{{
  "name": string | null,
  "position": string | null,
  "salary": integer | null,          // numeric USD amount only, no symbols
  "location": string | null,
  "start_date": string | null        // YYYY-MM-DD
}}

TODAY=2025-09-06

Rules:
- Do NOT invent data. If a value is not stated, use null.
- Trim titles from names (e.g., remove "Dr.", "Ms.", etc.). Keep initials if that's all you have (e.g., "A. Smith").
- salary: extract the numeric total (e.g., "$145,000" → 145000; "Band B3" → null).
- location: copy as written (city/country/"Remote – EU", etc.) if present.
- start_date normalization:
  • Accept YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY → convert to YYYY-MM-DD using the most likely locale if unambiguous.
  • "next Monday" → date of the next Monday after TODAY.
  • "in N days/weeks" → TODAY + N days/weeks.
  • "next month" → first day of next month.
  • "Q1/Q2/Q3/Q4 YYYY" → first day of that quarter (Q1=01-01, Q2=04-01, Q3=07-01, Q4=10-01).
  • "ASAP", "TBD", missing, or ambiguous → null.

Output: ONLY the JSON. No comments, no extra text.

Example input:
"System alert: Please welcome Sarah Chen, Senior Software Engineer in AI/ML. Start date next month. Salary: $145,000. Office: San Francisco."

Example output:
{{
  "name": "Sarah Chen",
  "position": "Senior Software Engineer",
  "salary": 145000,
  "location": "San Francisco",
  "start_date": "2025-09-01"
}}

Message: {message}
"""

# Prompt for the masking agent.  This agent receives a raw onboarding message
# and must remove or generalize sensitive information before it is passed on.
MASKING_PROMPT: str = """
You are a masking agent for HR onboarding messages.  Before forwarding a message
to the security team, you must remove or generalize sensitive information.

Sensitive fields:
- Salary: remove any mention of salary, pay range, pay grade or band.
- Location: do not reveal the exact city or office address.  Replace it with
  only the country if it is mentioned (e.g. “USA”, “Germany”) or, if a
  country is not specified, the continent (e.g. “Europe”, “North America”).

All other details (name, role, department, start date, high‑level work location
like “remote”, etc.) should be preserved as written.  Do not invent any new
information.  Return only the edited message with the sensitive fields
removed or generalized.  Do not include explanations or JSON.

Message: {message}
"""

# Prompt for the rewriting agent.  This agent takes the masked message and
# crafts a concise notification for the security team.
REWRITING_PROMPT: str = """
You are a rewriting agent responsible for crafting a concise, professional
notification to the security team based on a masked HR onboarding message.
The masked message has already had salary information removed and the
location generalized to a country or continent.

Your task:
- Create a clear summary that includes only the information the security team
  needs: the employee’s name, role/position, start date, department (if
  mentioned) and the generalized location (country or continent).
- Use complete sentences and a formal but friendly tone.
- Do not reintroduce any salary information or precise locations.  Do not
  invent missing details.

Example:
Masked message: "Please welcome Sarah Chen, Senior Software Engineer in AI/ML.
Start date next month. Office: USA."
Rewrite: "Sarah Chen will start next month as a Senior Software Engineer in
our AI/ML team. She will be based in the USA."

Now rewrite the following masked message accordingly.

Masked message: {masked_message}
"""

SECURITY_PROMPT: str = """
You are the Security Agent. Read the (filtered) HR message and return ONLY a single JSON object for the Security database with this exact schema and order:

{{
  "name": string | null,
  "email": string | null,
  "position": string | null,
  "security_level": 1 | 2 | 3,
  "keycard_access": {{
    "Europe": 0 | 1,
    "North America": 0 | 1,
    "South America": 0 | 1,
    "Africa": 0 | 1,
    "Asia": 0 | 1,
    "Oceania": 0 | 1
  }}
}}

TODAY=2025-09-06

Rules:
- Do NOT invent data. If a value is not stated, use null (except security_level and keycard_access must be integers).
- security_level is based on role seniority (deterministic):
  • Junior/Intern/Associate/Analyst/Engineer I/Contractor → 1
  • Manager/Specialist/Engineer II/Generalist/Accountant → 2
  • Senior/Lead/Head/Director/VP/C-level/Counsel/Principal/Chief → 3
  If mixed/ambiguous, choose the highest applicable keyword present; if none, default 2.
- keycard_access: set exactly one continent to 1 based on LOCATION; all others 0.
  Mapping (case-insensitive substring match on location/country/region):
  • North America: "USA","United States","Canada","Mexico"
  • South America: "Brazil","Argentina","Chile","Colombia","Peru","Uruguay"
  • Europe: "UK","United Kingdom","England","Scotland","Ireland","France","Germany","Poland","Switzerland","Netherlands","Spain","Italy","Sweden","Norway","Denmark","Portugal"
  • Africa: "South Africa","Kenya","Nigeria","Egypt","Ghana","Morocco"
  • Asia: "Japan","Korea","South Korea","India","Singapore","China","Hong Kong","Taiwan","Malaysia","Indonesia","Philippines","Vietnam","Thailand"
  • Oceania: "Australia","Sydney","Melbourne","New Zealand"
  Special cases:
  • "Remote - EU" → Europe=1
  • "Remote - US" → North America=1
  If location missing/unknown → set all 0.
- email: copy if present; otherwise null.
- position/name: copy without any modifications if present; otherwise null. 

Output: ONLY the JSON. No comments, no extra text.

Example input:
"Please welcome Sarah Chen as Senior Software Engineer, AI/ML. Start date next month. Office: San Francisco. Contact: sarah.chen@company.com."

Example output:
{{
  "name": "Sarah Chen",
  "email": "sarah.chen@company.com",
  "position": "Senior Software Engineer",
  "security_level": 3,
  "keycard_access": {{
    "Europe": 0,
    "North America": 1,
    "South America": 0,
    "Africa": 0,
    "Asia": 0,
    "Oceania": 0
  }}
}}

Message: {message}
"""