"""
Auto-Review Responder - FastAPI Backend
Complete production-ready API for Chrome extension
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import json
from datetime import datetime, timedelta
from anthropic import Anthropic

app = FastAPI(title="Auto-Review Responder API")

# CORS - Allow all origins for now (Chrome extension + testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= DATA MODELS =============

class UserProfile(BaseModel):
    user_id: str
    business_name: str
    business_type: str
    tone: str = "professional"
    brand_voice: Optional[str] = None
    signature: Optional[str] = None
    subscription_tier: str = "free"
    usage_count: int = 0
    usage_reset_date: Optional[str] = None

class ReviewInput(BaseModel):
    review_text: str
    rating: int
    reviewer_name: Optional[str] = None
    platform: str = "google"
    context: Optional[str] = None

class ResponseOutput(BaseModel):
    responses: List[dict]
    usage_remaining: int

# ============= IN-MEMORY STORAGE =============
users_db = {}
api_keys_db = {}

# ============= SUBSCRIPTION LIMITS =============
TIER_LIMITS = {
    "free": {"monthly_limit": 10, "response_count": 3},
    "pro": {"monthly_limit": 500, "response_count": 3},
    "enterprise": {"monthly_limit": -1, "response_count": 5}
}

# ============= HELPER FUNCTIONS =============

def verify_api_key(x_api_key: str = Header(...)):
    """Verify user's API key"""
    if x_api_key not in api_keys_db:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_keys_db[x_api_key]

def check_usage_limit(user: UserProfile):
    """Check if user has exceeded monthly limit"""
    tier_limit = TIER_LIMITS[user.subscription_tier]["monthly_limit"]
    
    # Reset counter if new month
    if user.usage_reset_date:
        reset_date = datetime.fromisoformat(user.usage_reset_date)
        if datetime.now() > reset_date:
            user.usage_count = 0
            user.usage_reset_date = (datetime.now() + timedelta(days=30)).isoformat()
    else:
        user.usage_reset_date = (datetime.now() + timedelta(days=30)).isoformat()
    
    if tier_limit != -1 and user.usage_count >= tier_limit:
        raise HTTPException(
            status_code=429, 
            detail=f"Monthly limit reached. Upgrade to Pro for 500 responses/month"
        )
    
    return tier_limit - user.usage_count if tier_limit != -1 else 999

async def generate_ai_responses(review: ReviewInput, profile: UserProfile) -> List[dict]:
    """Generate AI responses using Claude API"""
    
    # Sentiment detection
    sentiment = "positive" if review.rating >= 4 else "negative" if review.rating <= 2 else "neutral"
    
    # Build context-aware prompt
    brand_context = f"\n\nBrand Voice: {profile.brand_voice}" if profile.brand_voice else ""
    signature = f"\n\nAlways end with: {profile.signature}" if profile.signature else ""
    
    response_count = TIER_LIMITS[profile.subscription_tier]["response_count"]
    
    prompt = f"""You are responding to a {sentiment} review for {profile.business_name}, a {profile.business_type}.

Review ({review.rating} stars) from {review.reviewer_name or 'customer'} on {review.platform}:
"{review.review_text}"
{brand_context}{signature}

Generate {response_count} different response options with a {profile.tone} tone:

1. Short & Sweet (1-2 sentences) - Quick, warm acknowledgment
2. Detailed & Personal (3-4 sentences) - Shows you read and care
3. Professional & Branded (2-3 sentences with subtle CTA)

IMPORTANT RULES:
- For negative reviews: Acknowledge issue, apologize sincerely, offer solution
- For positive reviews: Thank genuinely, reinforce specific points they mentioned
- Match the energy level of the review
- Never be defensive or robotic
- Include specific details from their review

Return ONLY valid JSON with no markdown formatting:
{{
  "responses": [
    {{"style": "Short & Sweet", "text": "response here"}},
    {{"style": "Detailed & Personal", "text": "response here"}},
    {{"style": "Professional & Branded", "text": "response here"}}
  ]
}}"""

    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Make API call to Claude
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract text from response
        response_text = message.content[0].text
        
        # Clean and parse JSON
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean_text)
        
        return parsed.get("responses", [])
        
    except json.JSONDecodeError as e:
        # Fallback responses if JSON parsing fails
        print(f"JSON parsing error: {e}")
        return [
            {
                "style": "Short & Sweet",
                "text": f"Thank you for your {review.rating}-star review! We appreciate your feedback. {profile.signature or ''}"
            },
            {
                "style": "Detailed & Personal",
                "text": f"Thank you for taking the time to share your experience at {profile.business_name}. We're {'thrilled' if sentiment == 'positive' else 'sorry to hear about your experience and would love to make it right'}. {profile.signature or ''}"
            },
            {
                "style": "Professional & Branded",
                "text": f"We appreciate your feedback. {'Your satisfaction is our priority and we hope to see you again soon!' if sentiment == 'positive' else 'We take all feedback seriously and would love the opportunity to improve. Please contact us directly.'} {profile.signature or ''}"
            }
        ]
    except Exception as e:
        print(f"AI generation error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

# ============= API ENDPOINTS =============

@app.post("/api/register")
async def register_user(profile: UserProfile):
    """Register new user and generate API key"""
    api_key = f"rr_{profile.user_id}_{os.urandom(16).hex()}"
    
    users_db[profile.user_id] = profile.dict()
    api_keys_db[api_key] = profile.user_id
    
    return {
        "api_key": api_key,
        "message": "Registration successful",
        "tier": profile.subscription_tier
    }

@app.get("/api/profile")
async def get_profile(user_id: str = Depends(verify_api_key)):
    """Get user profile and settings"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.put("/api/profile")
async def update_profile(updates: dict, user_id: str = Depends(verify_api_key)):
    """Update user profile settings"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    allowed_fields = ["business_name", "business_type", "tone", "brand_voice", "signature"]
    for field in allowed_fields:
        if field in updates:
            users_db[user_id][field] = updates[field]
    
    return {"message": "Profile updated", "profile": users_db[user_id]}

@app.post("/api/generate", response_model=ResponseOutput)
async def generate_responses(review: ReviewInput, user_id: str = Depends(verify_api_key)):
    """Generate AI responses for a review"""
    
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    profile_data = users_db[user_id]
    profile = UserProfile(**profile_data)
    
    # Check usage limits
    remaining = check_usage_limit(profile)
    
    # Generate responses
    responses = await generate_ai_responses(review, profile)
    
    # Update usage count
    profile.usage_count += 1
    users_db[user_id] = profile.dict()
    
    return ResponseOutput(
        responses=responses,
        usage_remaining=remaining - 1
    )

@app.get("/api/usage")
async def get_usage(user_id: str = Depends(verify_api_key)):
    """Get current usage stats"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    profile = UserProfile(**users_db[user_id])
    tier_limit = TIER_LIMITS[profile.subscription_tier]["monthly_limit"]
    
    return {
        "tier": profile.subscription_tier,
        "usage_count": profile.usage_count,
        "monthly_limit": tier_limit if tier_limit != -1 else "unlimited",
        "reset_date": profile.usage_reset_date
    }

@app.post("/api/upgrade")
async def upgrade_subscription(tier: str, user_id: str = Depends(verify_api_key)):
    """Upgrade user subscription"""
    
    if tier not in TIER_LIMITS:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    users_db[user_id]["subscription_tier"] = tier
    
    return {
        "message": f"Upgraded to {tier}",
        "new_limit": TIER_LIMITS[tier]["monthly_limit"]
    }

@app.get("/")
async def root():
    return {
        "service": "Auto-Review Responder API",
        "version": "1.0",
        "docs": "/docs",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)