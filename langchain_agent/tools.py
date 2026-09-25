import asyncio
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings
from langchain_agent.rag import retrieve, build_context


# ---------------------------------------------------------------------------
# Shared singletons — initialized once at import time
# ---------------------------------------------------------------------------

embedder = SentenceTransformer(settings.embedding_model)
qdrant = QdrantClient(path=settings.qdrant_path)

try:
    _count = qdrant.get_collection(settings.collection_name).points_count
    print(f"[OK] Connected to '{settings.collection_name}' — {_count} points")
except Exception as exc:
    raise RuntimeError(
        f"Could not open Qdrant collection '{settings.collection_name}' "
        f"at '{settings.qdrant_path}'. Did you run `python ingest.py` first?"
    ) from exc


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_CUSTOMER_DB = {
    "9876543210": {
        "name": "Rajesh Kumar",
        "plan": "Lauki Premium 2GB",
        "balance_inr": 450.50,
        "balance_type": "postpaid",
        "next_bill_date": "2026-10-11",
        "data_usage_gb": 1.2,
        "data_limit_gb": 2.0,
        "voice_usage_min": 120,
        "voice_limit_min": 300,
        "sms_usage": 45,
        "sms_limit": 100,
    },
    "9123456789": {
        "name": "Priya Singh",
        "plan": "Lauki Basic 1GB",
        "balance_inr": 199.99,
        "balance_type": "prepaid",
        "validity_end": "2026-10-11",
        "data_usage_gb": 0.5,
        "data_limit_gb": 1.0,
        "voice_usage_min": 50,
        "voice_limit_min": 300,
        "sms_usage": 20,
        "sms_limit": 100,
    },
    "8765432109": {
        "name": "Amit Patel",
        "plan": "Lauki Elite 5GB",
        "balance_inr": 899.00,
        "balance_type": "postpaid",
        "next_bill_date": "2026-10-15",
        "data_usage_gb": 3.8,
        "data_limit_gb": 5.0,
        "voice_usage_min": 250,
        "voice_limit_min": 1000,
        "sms_usage": 150,
        "sms_limit": 500,
    },
}

MOCK_PLANS_DB = {
    "Lauki Basic 1GB": {
        "price": 199,
        "validity": "30 days",
        "data": "1 GB",
        "voice": "300 minutes",
        "sms": "100 SMS",
        "benefits": ["Rollover 500MB unused data", "Free incoming calls"],
    },
    "Lauki Premium 2GB": {
        "price": 449,
        "validity": "30 days",
        "data": "2 GB",
        "voice": "300 minutes",
        "sms": "100 SMS",
        "benefits": ["Free data rollover", "Priority customer support", "1 month free Netflix"],
    },
    "Lauki Elite 5GB": {
        "price": 899,
        "validity": "30 days",
        "data": "5 GB",
        "voice": "1000 minutes",
        "sms": "500 SMS",
        "benefits": ["Unlimited data rollover", "24/7 priority support", "Netflix + Amazon Prime"],
    },
    "Lauki Lite 500MB": {
        "price": 99,
        "validity": "7 days",
        "data": "500 MB",
        "voice": "100 minutes",
        "sms": "50 SMS",
        "benefits": ["Pay-as-you-go after limit", "No commitment"],
    },
}

MOCK_NETWORK_STATUS = {
    "hyderabad": {"4g": "Excellent", "5g": "Available", "coverage": "99.5%"},
    "bangalore": {"4g": "Excellent", "5g": "Available", "coverage": "99.8%"},
    "delhi": {"4g": "Good", "5g": "Limited", "coverage": "98.2%"},
    "mumbai": {"4g": "Excellent", "5g": "Available", "coverage": "99.7%"},
    "pune": {"4g": "Good", "5g": "Available", "coverage": "97.8%"},
    "default": {"4g": "Good", "5g": "Coming soon", "coverage": "95%"},
}


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


@tool
def search_plans(query: str) -> str:
    """Search Lauki Phones' plan database and documentation for information about
    data plans, pricing, validity, voice minutes, SMS limits, and benefits.
    Use this whenever the user asks about plans, pricing, or plan comparison."""
    chunks = retrieve(qdrant, embedder, query, top_k=5)
    if not chunks:
        return "No relevant information found in the documentation."
    return build_context(chunks)


@tool
def check_account_balance(phone_number: str) -> str:
    """Check the account balance, remaining data, voice minutes, and SMS for a customer.
    Requires a 10-digit phone number (e.g., 9876543210)."""

    phone_number = phone_number.strip()

    if not phone_number.isdigit() or len(phone_number) != 10:
        return (
            "Invalid phone number. Please provide a valid 10-digit Indian mobile number "
            "(e.g., 9876543210)."
        )

    customer = MOCK_CUSTOMER_DB.get(phone_number)
    if not customer:
        return f"No account found for {phone_number}. Please verify the number."

    if customer["balance_type"] == "postpaid":
        balance_info = f"Postpaid account. Outstanding bill: ₹{customer['balance_inr']:.2f}"
        period = f"Next bill date: {customer['next_bill_date']}"
    else:
        balance_info = f"Prepaid balance: ₹{customer['balance_inr']:.2f}"
        period = f"Validity until: {customer['validity_end']}"

    usage = (
        f"Current plan: {customer['plan']}. "
        f"Data: {customer['data_usage_gb']}GB of {customer['data_limit_gb']}GB used. "
        f"Voice: {customer['voice_usage_min']} of {customer['voice_limit_min']} minutes used. "
        f"SMS: {customer['sms_usage']} of {customer['sms_limit']} SMS used."
    )

    return f"{balance_info}. {period}. {usage}"


@tool
def check_network_status(region: str = "default") -> str:
    """Check 4G, 5G availability and network coverage in a specific region.
    Common regions: hyderabad, bangalore, delhi, mumbai, pune."""

    region = region.lower().strip()
    status = MOCK_NETWORK_STATUS.get(region, MOCK_NETWORK_STATUS["default"])

    return (
        f"Network status in {region}: 4G coverage is {status['4g']}, "
        f"5G is {status['5g']}, overall coverage is {status['coverage']}."
    )


@tool
def get_plan_recommendation(usage_type: str, data_gb: int = 0, voice_min: int = 0) -> str:
    """Recommend a Lauki Phones plan based on customer usage patterns.
    Usage types: 'light', 'medium', 'heavy', or specify exact data/voice needs."""

    usage_type = usage_type.lower().strip()

    if usage_type == "light":
        recommended = "Lauki Lite 500MB"
        reason = "Perfect for occasional browsing and calls"
    elif usage_type == "medium":
        recommended = "Lauki Basic 1GB"
        reason = "Ideal for social media, messaging, and moderate video watching"
    elif usage_type == "heavy":
        recommended = "Lauki Premium 2GB or Elite 5GB"
        reason = "Great for streaming, video calls, and downloading"
    else:
        if data_gb <= 1:
            recommended = "Lauki Basic 1GB"
        elif data_gb <= 2:
            recommended = "Lauki Premium 2GB"
        else:
            recommended = "Lauki Elite 5GB"
        reason = f"Based on your requirement of {data_gb}GB data"

    plan = MOCK_PLANS_DB.get(recommended)
    if plan:
        return (
            f"{reason}. Plan: {recommended}. "
            f"Price: ₹{plan['price']}/month, {plan['data']} data, "
            f"{plan['voice']} voice, {plan['sms']} SMS. "
            f"Benefits: {', '.join(plan['benefits'])}."
        )

    return f"Recommended plan: {recommended}. Please ask for more details."


@tool
def get_plan_details(plan_name: str) -> str:
    """Get detailed information about a specific Lauki Phones plan.
    Provide the plan name (e.g., 'Lauki Premium 2GB')."""

    plan_name = plan_name.strip()

    # Try exact match first, then case-insensitive
    plan = MOCK_PLANS_DB.get(plan_name)
    if not plan:
        for key in MOCK_PLANS_DB:
            if key.lower() == plan_name.lower():
                plan = MOCK_PLANS_DB[key]
                plan_name = key
                break

    if not plan:
        return (
            f"Plan '{plan_name}' not found. Available plans: "
            f"{', '.join(MOCK_PLANS_DB.keys())}."
        )

    benefits = ", ".join(plan["benefits"])
    return (
        f"{plan_name}: ₹{plan['price']} for {plan['validity']}. "
        f"Includes {plan['data']} data, {plan['voice']} voice, {plan['sms']} SMS. "
        f"Benefits: {benefits}."
    )


@tool
def escalate_to_support(reason: str) -> str:
    """Escalate to a human support representative for issues the agent cannot resolve —
    account disputes, technical issues, complaints, or special requests."""

    ticket_id = "LAUKI-" + str(abs(hash(reason)) % 100000).zfill(5)
    timestamp = datetime.now().isoformat()

    print(f"[ESCALATION] ID={ticket_id} time={timestamp} reason={reason!r}")

    return (
        f"I've created support ticket {ticket_id}. "
        f"A representative will contact you within 2 hours. "
        f"Your reference is {ticket_id}."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly and professional voice-based customer support agent for Lauki Phones.\n"
    "\n"
    "GUIDELINES:\n"
    "- Use search_plans to answer questions about data plans, pricing, validity, or comparisons.\n"
    "- Use check_account_balance when the customer provides their 10-digit phone number.\n"
    "- Use check_network_status for 4G/5G coverage questions.\n"
    "- Use get_plan_recommendation when the customer describes usage patterns.\n"
    "- Use get_plan_details for detailed plan information.\n"
    "- Use escalate_to_support for issues you cannot resolve.\n"
    "\n"
    "TONE:\n"
    "- Be warm, professional, and helpful.\n"
    "- Keep responses SHORT — this is a voice call, not text chat.\n"
    "- Use Indian currency (₹).\n"
    "\n"
    "CONSTRAINTS:\n"
    "- NEVER make up information about plans, pricing, or coverage.\n"
    "- NEVER request sensitive details beyond phone number.\n"
    "- ALWAYS verify phone numbers are 10 digits before lookup.\n"
    "- If you can't find information, offer to escalate."
    "- Always keep your responses conversational, don't respond with tables or diagrams."
)


# ---------------------------------------------------------------------------
# Build the LangChain agent graph
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_plans,
    check_account_balance,
    check_network_status,
    get_plan_recommendation,
    get_plan_details,
    escalate_to_support,
]

llm = ChatGroq(model=settings.groq_model, temperature=0.0)

lauki_agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
