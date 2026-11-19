import re

# 8 made-up company handbook passages
DOCS = [
    {
        "id": "HR-1",
        "title": "Vacation Policy",
        "text": "Full-time employees receive 24 days of paid vacation per calendar year, accrued monthly at 2 days per month. Vacation requests must be submitted at least 2 weeks in advance through the HR portal and require manager approval."
    },
    {
        "id": "HR-2",
        "title": "Sick Leave",
        "text": "Employees are entitled to 10 days of paid sick leave annually. If sick leave extends beyond 3 consecutive days, a doctor's note must be provided to HR upon return to work."
    },
    {
        "id": "HR-3",
        "title": "Remote Work Policy",
        "text": "Employees are permitted to work remotely up to 3 days per week with manager approval. All employees must remain available on Slack and email during core working hours from 10:00 AM to 4:00 PM."
    },
    {
        "id": "FIN-1",
        "title": "Expense Reimbursement",
        "text": "Business expenses up to $50 can be approved automatically with valid itemized receipts submitted within 30 days of purchase. Expenses over $50 require prior written manager approval."
    },
    {
        "id": "IT-1",
        "title": "Laptops & Hardware",
        "text": "New employees receive a company-issued laptop (Apple MacBook Pro or Lenovo ThinkPad) refreshed every 3 years. Installing personal software or using personal storage drives on company hardware is strictly prohibited."
    },
    {
        "id": "IT-2",
        "title": "Passwords & Security",
        "text": "All system passwords must be at least 14 characters long and include numbers, uppercase letters, and special symbols. Passwords expire and must be changed every 90 days. Multi-factor authentication (2FA) is mandatory."
    },
    {
        "id": "OPS-1",
        "title": "Office Hours & Facility Access",
        "text": "Standard corporate office hours are Monday through Friday from 8:00 AM to 7:00 PM. Access outside these hours requires security badge authorization, and security keycards must be worn visibly at all times."
    },
    {
        "id": "HR-4",
        "title": "Training & Development Budget",
        "text": "Every full-time team member has an annual training budget of $1,500 for approved educational courses, industry certifications, conferences, and books. Expense claims must be submitted within 14 days of completion."
    }
]

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def keywords(text: str) -> list[str]:
    """Lowercase words minus stopwords."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    cleaned = [w.strip("-_") for w in words]
    return [w for w in cleaned if w and w not in STOPWORDS]

def search_docs(query: str, k: int = 2) -> list[dict]:
    """
    Search docs using keyword overlap score.
    Keeps passages scoring >= 60% of the best score.
    Returns up to k documents.
    """
    q_words = set(keywords(query))
    if not q_words:
        return []

    scored_docs = []
    for doc in DOCS:
        doc_words = set(keywords(f"{doc['id']} {doc['title']} {doc['text']}"))
        score = len(q_words & doc_words)
        if score > 0:
            scored_docs.append((score, doc))

    if not scored_docs:
        return []

    # Sort descending by score
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    best_score = scored_docs[0][0]

    # Keep passages scoring >= 60% of best score
    threshold = 0.6 * best_score
    filtered_docs = [doc for score, doc in scored_docs if score >= threshold]

    return filtered_docs[:k]

def format_context(docs: list[dict]) -> str:
    """Format passages into '[ID] Title: text' lines."""
    return "\n\n".join(f"[{doc['id']}] {doc['title']}: {doc['text']}" for doc in docs)
