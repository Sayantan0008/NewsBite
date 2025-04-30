from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import re

# Global variables to avoid reloading model
MODEL = None
TOKENIZER = None
CATEGORIES = [
    "Politics", "Tech", "Sports", "Health", 
    "Entertainment", "World", "Business", "Science"
]

# Define subcategories for each main category
SUBCATEGORIES = {
    "Politics": ["International Relations", "Elections", "Legislation", "Government", "Policy", "Terrorism"],
    "Business": ["Finance", "Economy", "Markets", "Startups", "Corporate", "Real Estate", "Cryptocurrency", "Insurance"],
    "Tech": ["AI", "Cybersecurity", "Software", "Hardware", "Startups", "Social Media", "Mobile", "Consumer Electronics"],
    "Health": ["Medical Research", "Healthcare Policy", "Wellness", "Disease", "Mental Health", "Fitness"],
    "Entertainment": ["Movies", "Music", "Celebrity", "TV Shows", "Video Games", "Gaming", "Arts"],
    "Sports": ["Football", "Basketball", "Tennis", "Olympics", "Motorsports", "Cricket", "Esports"],
    "Science": ["Space", "Environment", "Research", "Climate", "Biology", "Physics"],
    "World": ["Europe", "Asia", "Americas", "Africa", "Middle East", "Oceania"]
}

def load_model():
    """
    Load the BERT classification model and tokenizer
    """
    global MODEL, TOKENIZER
    
    if MODEL is None or TOKENIZER is None:
        print("Loading BERT categorization model...")
        model_name = "bert-base-uncased"
        num_labels = len(CATEGORIES)
        
        # Load tokenizer
        TOKENIZER = BertTokenizer.from_pretrained(model_name)
        
        # Load model
        # Note: In a real implementation, you would load a fine-tuned model
        # For demonstration, we're using the base model
        MODEL = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            MODEL = MODEL.to("cuda")
    
    return MODEL, TOKENIZER

def predict_subcategory(text, main_category):
    """
    Predict the subcategory of a news article based on its main category
    
    Args:
        text (str): The article text to categorize
        main_category (str): The main category of the article
        
    Returns:
        str: The predicted subcategory
    """
    # Skip empty texts or invalid categories
    if not text or main_category not in SUBCATEGORIES:
        return None
    
    # Simple keyword-based approach for subcategory prediction
    # In a real implementation, you would use a more sophisticated model
    subcategories = SUBCATEGORIES[main_category]
    scores = {subcategory: 0 for subcategory in subcategories}
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Define keywords for each subcategory
    keywords = {
        # Politics subcategories
        "International Relations": ["diplomat", "treaty", "foreign", "international", "relations", "embassy", "global"],
        "Elections": ["vote", "ballot", "election", "campaign", "candidate", "poll", "voter", "riding", "federal election", "polls", "newfoundland", "labrador"],
        "Legislation": ["law", "bill", "legislation", "congress", "parliament", "senate", "legal"],
        "Government": ["government", "administration", "cabinet", "minister", "president", "official", "agency"],
        "Policy": ["policy", "regulation", "reform", "initiative", "program", "strategy", "plan"],
        "Terrorism": ["terrorist", "terrorism", "attack", "bomb", "bombing", "explosion", "hostage", "extremist", "militant", "gunman", "shooter", "massacre", "suicide bomb", "terror plot", "radicalized", "extremism", "jihad", "naxal", "naxalite", "conflict zone", "threat"],
        
        # Business subcategories
        "Finance": ["bank", "investment", "financial", "loan", "credit", "fund", "banking"],
        "Economy": ["economy", "economic", "gdp", "inflation", "recession", "growth", "fiscal"],
        "Markets": ["stock", "market", "trade", "investor", "share", "index", "exchange"],
        "Startups": ["startup", "entrepreneur", "venture", "funding", "seed", "incubator", "innovation"],
        "Corporate": ["corporate", "company", "ceo", "executive", "corporation", "board", "merger"],
        "Real Estate": ["property", "real estate", "housing", "mortgage", "commercial", "residential", "rent"],
        
        # Tech subcategories
        "AI": ["ai", "artificial intelligence", "machine learning", "neural", "algorithm", "deep learning", "model", "google ai", "chatgpt", "openai", "llm", "large language model", "generative ai", "bard", "gemini", "claude"],
        "Cybersecurity": ["security", "hack", "breach", "cyber", "encryption", "password", "threat"],
        "Software": ["software", "app", "application", "code", "developer", "program", "update"],
        "Hardware": ["hardware", "device", "chip", "processor", "computer", "server", "component"],
        "Social Media": ["social media", "facebook", "twitter", "instagram", "tiktok", "platform", "user"],
        "Mobile": ["smartphone", "mobile phone", "android", "ios", "iphone", "samsung", "galaxy", "pixel", "oneplus", "xiaomi", "huawei", "oppo", "vivo", "one ui", "miui", "oxygen os", "app store", "play store"],
        "Consumer Electronics": ["gadget", "device", "consumer tech", "wearable", "smartwatch", "headphone", "earbud", "speaker", "tv", "television", "laptop", "tablet", "camera", "drone"],
        
        # World subcategories
        "Middle East": ["middle east", "israel", "palestine", "syria", "iran", "iraq", "saudi", "yemen", "qatar", "lebanon", "turkey"],
        "Europe": ["europe", "eu", "european union", "uk", "britain", "france", "germany", "italy", "spain", "russia", "ukraine", "poland", "sweden", "norway", "finland", "denmark", "netherlands", "belgium", "switzerland", "austria", "greece", "portugal", "ireland", "scotland", "wales"],
        "Asia": ["asia", "china", "japan", "india", "pakistan", "north korea", "south korea", "taiwan", "philippines", "indonesia", "delhi", "mumbai", "bangalore", "kolkata", "chennai", "hyderabad", "bengaluru", "indian", "hindu", "modi", "bjp", "congress party", "lok sabha", "rajya sabha", "rupee", "thailand", "vietnam", "malaysia", "singapore", "bangladesh", "nepal", "sri lanka", "bhutan", "maldives"],
        "Americas": ["canada", "mexico", "brazil", "argentina", "colombia", "venezuela", "peru", "chile", "cuba", "united states", "usa", "us", "american", "washington", "new york", "california", "texas", "florida"],
        "Africa": ["africa", "egypt", "nigeria", "south africa", "kenya", "ethiopia", "somalia", "libya", "sudan", "ghana", "tanzania", "uganda", "morocco", "algeria", "tunisia", "cameroon", "senegal", "ivory coast", "zimbabwe", "zambia", "rwanda", "congo"],
        "Oceania": ["australia", "new zealand", "pacific", "papua", "fiji", "sydney", "melbourne", "brisbane", "perth", "auckland", "wellington", "samoa", "tonga", "vanuatu", "solomon islands"],
        
        # Sports subcategories with expanded keywords
        "Football": ["football", "nfl", "quarterback", "touchdown", "field goal", "super bowl", "offensive line", "defensive line", "wide receiver", "running back", "soccer", "fifa", "premier league", "goal", "striker", "midfielder", "defender", "goalkeeper", "penalty", "free kick", "corner", "offside", "world cup", "champions league"],
        "Basketball": ["basketball", "nba", "point guard", "shooting guard", "small forward", "power forward", "center", "dunk", "three-pointer", "free throw", "rebound", "assist", "block", "steal", "court", "backboard", "rim", "hoop", "draft", "playoffs", "finals", "all-star"],
        "Tennis": ["tennis", "grand slam", "wimbledon", "us open", "french open", "australian open", "atp", "wta", "racket", "serve", "ace", "forehand", "backhand", "volley", "deuce", "advantage", "break point", "set", "match point", "court", "singles", "doubles", "mixed doubles"],
        "Olympics": ["olympics", "olympic games", "summer olympics", "winter olympics", "gold medal", "silver medal", "bronze medal", "athlete", "ceremony", "torch", "relay", "opening ceremony", "closing ceremony", "host city", "international olympic committee", "ioc", "paralympics"],
        "Motorsports": ["formula one", "f1", "nascar", "indycar", "rally", "grand prix", "circuit", "race track", "driver", "team", "pit stop", "qualifying", "pole position", "lap", "championship", "podium", "constructor", "engine", "tire", "aerodynamics", "motorsport", "motogp", "superbike"],
        "Cricket": ["cricket", "test match", "one day international", "odi", "twenty20", "t20", "ipl", "indian premier league", "batsman", "bowler", "wicket", "run", "over", "innings", "pitch", "boundary", "six", "four", "lbw", "stumped", "caught", "world cup", "ashes", "bcci", "icc", "csk", "rcb", "mi", "kkr", "srh", "dc", "rr", "pbks", "gt", "lsg"],
        "Esports": ["esports", "competitive gaming", "tournament", "league of legends", "dota", "counter-strike", "overwatch", "fortnite", "pubg", "hearthstone", "starcraft", "pro gamer", "team", "streaming", "twitch", "prize pool", "championship", "world championship", "lan", "online tournament", "gaming house", "roster"],
        
        # Entertainment subcategories with expanded keywords
        "Video Games": ["video game", "gaming", "gamer", "console", "playstation", "xbox", "nintendo", "switch", "pc gaming", "steam", "epic games", "game developer", "game studio", "release date", "gameplay", "dlc", "expansion", "multiplayer", "single-player", "rpg", "fps", "mmorpg", "battle royale", "open world", "indie game", "pokemon", "pokemon unite", "league of legends", "call of duty", "minecraft", "fortnite", "apex legends", "world of warcraft", "final fantasy", "zelda", "mario", "smash bros", "animal crossing", "halo", "destiny", "valorant", "among us", "roblox", "genshin impact", "overwatch", "hearthstone", "rainbow six", "assassin's creed", "god of war", "cyberpunk", "elden ring", "diablo", "starcraft", "warcraft", "moba", "battle pass", "season pass", "esports"],
        "Gaming": ["gaming", "game", "player", "console", "controller", "level", "character", "quest", "achievement", "high score", "leaderboard", "virtual reality", "augmented reality", "mobile gaming", "casual gaming", "hardcore gaming", "game design", "game engine", "game mechanics", "loot box", "microtransaction", "early access", "beta", "pokemon", "nintendo", "playstation", "xbox", "steam", "epic games", "ubisoft", "ea", "activision", "blizzard", "riot games", "valve", "bethesda", "rockstar", "square enix", "capcom", "konami", "sega", "namco", "bandai", "sony", "microsoft"],
        
        # Add keywords for other subcategories as needed
    }
    
    # Score each subcategory based on keyword matches with context awareness
    for subcategory in subcategories:
        if subcategory in keywords:
            for keyword in keywords[subcategory]:
                # Check for whole word matches to improve accuracy
                start_pos = 0
                while True:
                    pos = text_lower.find(keyword, start_pos)
                    if pos == -1:
                        break
                    
                    # Check if it's a whole word match
                    is_whole_word = False
                    if pos == 0 or not text_lower[pos-1].isalnum():
                        if pos + len(keyword) >= len(text_lower) or not text_lower[pos + len(keyword)].isalnum():
                            is_whole_word = True
                    
                    # Add to score with appropriate weight
                    if is_whole_word:
                        # Give more weight to longer, more specific keywords
                        specificity = min(2.0, max(1.0, len(keyword) / 5))
                        scores[subcategory] += specificity
                    
                    start_pos = pos + len(keyword)
    
    # Find the subcategory with the highest score
    max_score = 0
    best_subcategory = subcategories[0]  # Default to first subcategory
    
    for subcategory, score in scores.items():
        if score > max_score:
            max_score = score
            best_subcategory = subcategory
    
    return best_subcategory

# Suppress warnings
import warnings
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress other warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def predict_category(text):
    """
    Predict the category of a news article using a hybrid approach combining
    keyword-based analysis and ML model predictions
    
    Args:
        text (str): The article text to categorize
        
    Returns:
        str: The predicted category
    """
    # Skip empty texts
    if not text or text.lower() == "nan":
        return CATEGORIES[0]  # Default to first category
    
    # Preprocess text for better analysis
    text_lower = text.lower()
    
    # Define comprehensive category-specific keywords with expanded vocabulary
    category_keywords = {
        "Politics": [
            "government", "president", "election", "congress", "senate", "parliament", "vote", "campaign", 
            "democrat", "republican", "legislation", "policy", "minister", "diplomat", "treaty", "bill", 
            "law", "political", "politician", "administration", "cabinet", "official", "referendum", "ballot",
            "governor", "mayor", "constitutional", "judiciary", "supreme court", "democracy", "authoritarian",
            "dictatorship", "monarchy", "sovereignty", "impeachment", "scandal", "corruption", "lobbying",
            "protest", "rally", "demonstration", "activism", "conservative", "liberal", "progressive",
            "left-wing", "right-wing", "centrist", "bipartisan", "filibuster", "geopolitical", "statecraft",
            "court", "judge", "ruling", "lawsuit", "legal", "justice", "abortion", "planned parenthood", "pro-life", "pro-choice"
        ],
        "Tech": [
            "technology", "software", "hardware", "app", "digital", "computer", "internet", "startup", 
            "innovation", "ai", "artificial intelligence", "machine learning", "algorithm", "cyber", "code", 
            "programming", "developer", "tech", "gadget", "device", "smartphone", "robot", "automation", 
            "data", "cloud", "virtual", "augmented reality", "blockchain", "google ai", "chatgpt", "openai",
            "neural network", "big data", "iot", "internet of things", "5g", "broadband", "wifi", "wireless",
            "quantum computing", "vr", "virtual reality", "ar", "metaverse", "saas", "api", "open source",
            "encryption", "cybersecurity", "malware", "ransomware", "phishing", "hacker", "tech giant",
            "silicon valley", "autonomous", "self-driving", "drone", "wearable", "biometric", "fintech",
            "samsung", "galaxy", "one ui", "android", "ios", "iphone", "pixel", "mobile phone", "smartphone"
        ],
        "Sports": [
            "game", "team", "player", "coach", "tournament", "championship", "match", "league", 
            "score", "win", "lose", "football", "soccer", "basketball", "baseball", "tennis", "golf", 
            "olympics", "athlete", "sport", "racing", "cricket", "rugby", "hockey", "stadium", "fitness",
            "nfl", "nba", "mlb", "nhl", "premier league", "la liga", "serie a", "bundesliga", "world cup",
            "grand slam", "wimbledon", "masters", "formula one", "f1", "nascar", "boxing", "ufc", "mma",
            "wrestling", "gymnastics", "swimming", "track", "field", "marathon", "triathlon", "cycling",
            "draft", "transfer", "contract", "roster", "playoff", "championship", "medal", "record",
            "injury", "comeback", "retirement", "hall of fame", "mvp", "all-star", "champion"
        ],
        "Health": [
            "medical", "doctor", "hospital", "patient", "treatment", "disease", "drug", "vaccine", 
            "research", "study", "health", "healthcare", "medicine", "therapy", "wellness", "diet", 
            "nutrition", "mental health", "psychology", "surgery", "diagnosis", "pandemic", "virus", 
            "cancer", "diabetes", "obesity", "fitness", "exercise", "prevention", "cure",
            "pharmaceutical", "clinical", "trial", "fda", "who", "world health organization", "cdc",
            "epidemic", "outbreak", "symptom", "syndrome", "chronic", "acute", "emergency", "icu",
            "specialist", "physician", "nurse", "caregiver", "therapy", "rehabilitation", "telemedicine",
            "health insurance", "medicare", "medicaid", "affordable care act", "obamacare", "public health",
            "immunity", "antibody", "antigen", "pathogen", "bacteria", "infection", "prescription",
            "workout", "gym", "training", "strength", "cardio", "aerobic", "anaerobic", "muscle", "weight lifting",
            "yoga", "pilates", "running", "jogging", "cycling", "swimming", "sports training", "personal trainer",
            "abortion", "planned parenthood", "reproductive health", "women's health", "pregnancy", "birth", 
            "fetus", "embryo", "womb", "forceps", "medical procedure", "body parts", "tissue", "organ", "donation",
            "transplant", "medical ethics", "medical research", "clinical", "physician", "surgeon", "gynecology", "obstetrics"
        ],
        "Entertainment": [
            "movie", "film", "actor", "actress", "director", "hollywood", "celebrity", "star", 
            "music", "song", "album", "concert", "festival", "tv", "television", "show", "series", 
            "streaming", "netflix", "award", "performance", "theater", "stage", "comedy", "drama", 
            "game", "gaming", "arts", "entertainment", "box office", "release", "premiere",
            "blockbuster", "sequel", "prequel", "franchise", "studio", "producer", "screenplay", "script",
            "cast", "crew", "soundtrack", "genre", "oscar", "emmy", "grammy", "golden globe", "billboard",
            "chart", "hit", "tour", "band", "musician", "singer", "rapper", "dj", "pop", "rock", "hip hop",
            "jazz", "classical", "broadway", "disney", "hbo", "amazon prime", "hulu", "youtube",
            "viral", "trending", "influencer", "podcast", "bestseller", "author", "novel", "literature"
        ],
        "World": [
            "international", "global", "foreign", "country", "nation", "world", "overseas", "abroad", 
            "diplomatic", "embassy", "united nations", "eu", "european union", "asia", "europe", "africa", 
            "middle east", "americas", "oceania", "border", "immigration", "refugee", "crisis", "conflict", 
            "war", "peace", "summit", "trade", "sanction", "treaty", "agreement", "cooperation",
            "nato", "g7", "g20", "brics", "asean", "commonwealth", "wto", "imf", "world bank",
            "humanitarian", "aid", "development", "poverty", "famine", "disaster", "earthquake", "tsunami",
            "hurricane", "typhoon", "climate change", "global warming", "pollution", "sustainability",
            "human rights", "democracy", "autocracy", "coup", "revolution", "protest", "civil war",
            "genocide", "ethnic cleansing", "terrorism", "nuclear", "missile", "military", "defense"
        ],
        "Business": [
            "company", "corporation", "business", "market", "stock", "investor", "investment", "finance", 
            "economy", "economic", "trade", "industry", "sector", "profit", "revenue", "sales", "growth", 
            "startup", "entrepreneur", "ceo", "executive", "merger", "acquisition", "ipo", "banking", 
            "financial", "commerce", "retail", "consumer", "product", "service", "real estate", "property",
            "dow jones", "nasdaq", "s&p 500", "nyse", "wall street", "hedge fund", "private equity",
            "venture capital", "angel investor", "shareholder", "stakeholder", "dividend", "portfolio",
            "asset", "liability", "debt", "credit", "loan", "mortgage", "interest rate", "inflation",
            "deflation", "recession", "depression", "boom", "bust", "bull market", "bear market",
            "cryptocurrency", "bitcoin", "ethereum", "blockchain", "nft", "supply chain", "logistics",
            "manufacturing", "outsourcing", "globalization", "tariff", "subsidy", "regulation", "deregulation",
            "insurance", "insurer", "policy", "premium", "claim", "coverage", "risk", "underwriting", 
            "diamond price", "jewelry", "precious metals", "gold", "silver", "platinum"
        ],
        "Science": [
            "research", "scientist", "study", "discovery", "experiment", "laboratory", "theory", "physics", 
            "chemistry", "biology", "astronomy", "space", "planet", "star", "galaxy", "nasa", "telescope", 
            "particle", "atom", "molecule", "dna", "gene", "climate", "environment", "species", "evolution", 
            "fossil", "dinosaur", "technology", "innovation", "breakthrough", "quantum", "engineering",
            "astrophysics", "cosmology", "black hole", "supernova", "exoplanet", "mars", "rover", "probe",
            "spacex", "blue origin", "rocket", "satellite", "iss", "international space station",
            "genetics", "genomics", "crispr", "stem cell", "cloning", "biodiversity", "ecosystem",
            "conservation", "extinction", "renewable", "solar", "wind", "geothermal", "nuclear",
            "nanotechnology", "biotechnology", "neuroscience", "cognitive", "ai", "artificial intelligence",
            "peer-reviewed", "journal", "hypothesis", "thesis", "dissertation", "academia", "university"
        ]
    }
    
    # Calculate context-aware scores for each category
    category_scores = {category: 0 for category in CATEGORIES}
    
    # Score based on keyword presence, frequency, and context
    for category, keywords in category_keywords.items():
        # Base score from keyword matches with improved context awareness
        keyword_matches = 0
        keyword_importance = {}
        keyword_frequency = {}
        
        # Assign importance to each keyword based on specificity
        # Longer, more specific terms get higher weights
        for keyword in keywords:
            # Weight by keyword length (longer keywords are more specific)
            specificity = min(1.8, max(0.9, len(keyword) / 4))
            
            # Special handling for ambiguous terms
            # Increase specificity for tech terms to prevent false positives
            if category == "Tech" and keyword in ["ai", "model", "virtual", "cloud", "data"]:
                specificity *= 0.8  # Reduce weight for common ambiguous tech terms
            
            # Increase specificity for sports terms to better differentiate
            if category == "Sports" and keyword in ["basketball", "football", "soccer", "tennis"]:
                specificity *= 1.2  # Increase weight for major sports
                
            # Adjust weights for entertainment vs tech content
            if category == "Entertainment" and keyword in ["game", "gaming", "video"]:
                specificity *= 1.1  # Slightly increase entertainment gaming terms
            
            keyword_importance[keyword] = specificity
            
            # Count frequency of each keyword with improved whole-word matching
            count = 0
            start_pos = 0
            while True:
                pos = text_lower.find(keyword, start_pos)
                if pos == -1:
                    break
                # Check if it's a whole word match with improved boundary detection
                is_whole_word = False
                if pos == 0 or not text_lower[pos-1].isalnum():
                    if pos + len(keyword) >= len(text_lower) or not text_lower[pos + len(keyword)].isalnum():
                        is_whole_word = True
                
                # Add to count with appropriate weight
                if is_whole_word:
                    count += 1.5  # Higher weight for whole word matches
                else:
                    count += 0.5  # Lower weight for partial matches
                
                start_pos = pos + len(keyword)
            
            keyword_frequency[keyword] = count
            
            # Add to keyword matches with frequency and specificity factored in
            if count > 0:
                # Apply diminishing returns for repeated keywords
                frequency_factor = min(3.0, 1.0 + (0.5 * math.log(count + 1)))
                keyword_matches += frequency_factor * specificity
        
        # Weight by keyword density (matches relative to text length)
        text_length_factor = min(1.0, len(text_lower) / 5000)  # Normalize for very long texts
        density_score = keyword_matches / (len(keywords) * text_length_factor)
        
        # Apply category-specific adjustments to improve disambiguation
        category_adjustment = 1.0
        
        # Adjust scores for commonly confused categories
        if category == "Tech" and any(term in text_lower for term in ["celebrity", "movie star", "actor", "actress", "film", "hollywood"]):
            # Reduce Tech score if entertainment terms are present
            category_adjustment = 0.85
        
        if category == "Entertainment" and any(term in text_lower for term in ["algorithm", "programming", "software development", "code", "engineer", "technical"]):
            # Reduce Entertainment score if technical terms are present
            category_adjustment = 0.85
            
        # Improve Sports vs Health disambiguation
        if category == "Health" and any(term in text_lower for term in ["team", "player", "championship", "tournament", "league", "coach"]):
            # Reduce Health score if sports-specific terms are present
            category_adjustment = 0.9
            
        if category == "Sports" and any(term in text_lower for term in ["treatment", "therapy", "diagnosis", "patient", "doctor", "hospital", "medical condition"]):
            # Reduce Sports score if medical terms are present
            category_adjustment = 0.9
            
        # Apply the adjustment
        density_score *= category_adjustment
        
        # Enhanced positional weighting with multiple sections and semantic analysis
        # Title is most important, followed by first paragraph, then rest of text
        title_end = text_lower.find(".", 0, 200)  # Approximate title end
        if title_end == -1:
            title_end = min(200, len(text_lower))
        
        # Find first paragraph end (approximated by double newline or several sentences)
        first_para_end = text_lower.find("\n\n", 0, 500)
        if first_para_end == -1:
            # If no double newline, look for several sentences
            period_count = 0
            for i, char in enumerate(text_lower[:500]):
                if char == '.' and i > title_end:
                    period_count += 1
                    if period_count >= 3:  # After 3 sentences
                        first_para_end = i
                        break
        
        if first_para_end == -1 or first_para_end < title_end:
            first_para_end = min(500, len(text_lower))
        
        # Extract sections
        title_text = text_lower[:title_end]
        first_para = text_lower[title_end:first_para_end]
        rest_text = text_lower[first_para_end:]
        
        # Score each section with different weights and improved semantic analysis
        # For title, check for exact matches and partial matches with context
        title_matches = 0
        for keyword in keywords:
            if keyword in title_text:
                # Higher weight for keywords that appear at the beginning of the title
                position_factor = 1.0
                if title_text.find(keyword) < len(title_text) / 3:
                    position_factor = 1.5  # Beginning of title
                
                # Higher weight for keywords that make up a significant portion of the title
                length_ratio = len(keyword) / max(1, len(title_text))
                size_factor = 1.0 + min(1.0, length_ratio * 5)
                
                # Combine factors with keyword importance
                title_matches += keyword_importance.get(keyword, 1.0) * position_factor * size_factor
        
        # For first paragraph, analyze keyword density and proximity
        first_para_matches = 0
        for keyword in keywords:
            if keyword in first_para:
                # Count occurrences in first paragraph
                count = 0
                start_pos = 0
                while True:
                    pos = first_para.find(keyword, start_pos)
                    if pos == -1:
                        break
                    count += 1
                    start_pos = pos + 1
                
                # Apply diminishing returns for repeated keywords
                frequency_factor = min(2.0, 1.0 + (0.3 * math.log(count + 1)))
                
                # Higher weight for keywords that appear early in the paragraph
                position_factor = 1.0
                first_pos = first_para.find(keyword)
                if first_pos < len(first_para) / 4:
                    position_factor = 1.3  # Beginning of paragraph
                
                first_para_matches += keyword_importance.get(keyword, 1.0) * frequency_factor * position_factor * 0.7
        
        # For rest of text, check for keyword clusters (topics)
        rest_matches = 0
        if len(rest_text) > 0:
            # Find clusters of category keywords (indicates focused discussion)
            keyword_positions = []
            for keyword in keywords:
                pos = 0
                while True:
                    pos = rest_text.find(keyword, pos)
                    if pos == -1:
                        break
                    keyword_positions.append((pos, keyword))
                    pos += len(keyword)
            
            # Sort positions
            keyword_positions.sort()
            
            # Look for clusters (keywords close to each other)
            cluster_bonus = 0
            if len(keyword_positions) >= 2:
                for i in range(len(keyword_positions) - 1):
                    pos1, kw1 = keyword_positions[i]
                    pos2, kw2 = keyword_positions[i + 1]
                    # If keywords are within 100 chars, consider them related
                    if pos2 - pos1 < 100:
                        cluster_bonus += 0.5 * (keyword_importance.get(kw1, 1.0) + keyword_importance.get(kw2, 1.0))
            
            # Add cluster bonus to rest matches
            rest_matches = cluster_bonus * 0.5
        
        # Title keywords count triple, first paragraph keywords count double, rest has lower weight
        positional_score = (title_matches * 3.5) + (first_para_matches * 2.0) + (rest_matches * 0.8)
        
        # Calculate final score for this category with improved weighting
        category_scores[category] = (density_score * 1.5) + positional_score
    
    # Enhanced special handling for potentially ambiguous categories
    
    # Handle articles that mention both basketball and football
    if "basketball" in text_lower and "football" in text_lower:
        # Check which sport is more prominent by counting occurrences and context
        basketball_terms = ["basketball", "nba", "court", "dunk", "three-pointer", "point guard", "shooting guard"]
        football_terms = ["football", "nfl", "quarterback", "touchdown", "field goal", "soccer", "goal"]
        
        basketball_count = sum(text_lower.count(term) for term in basketball_terms)
        football_count = sum(text_lower.count(term) for term in football_terms)
        
        # Adjust Sports score based on which sport is more prominent
        if basketball_count > football_count * 1.5:
            # Article is more about basketball
            category_scores["Sports"] *= 1.1
        elif football_count > basketball_count * 1.5:
            # Article is more about football
            category_scores["Sports"] *= 1.1
    
    # Handle celebrity tech news vs pure tech content
    if "tech" in text_lower and any(term in text_lower for term in ["celebrity", "star", "famous"]):
        # Check if it's primarily about tech or about celebrities
        tech_focus_terms = ["technology", "innovation", "product", "device", "software", "hardware", "feature"]
        celebrity_focus_terms = ["celebrity", "star", "famous", "popular", "influencer", "social media star"]
        
        tech_focus_count = sum(text_lower.count(term) for term in tech_focus_terms)
        celebrity_focus_count = sum(text_lower.count(term) for term in celebrity_focus_terms)
        
        # Adjust scores based on primary focus
        if tech_focus_count > celebrity_focus_count * 1.5:
            # Article is more about tech than celebrities
            category_scores["Tech"] *= 1.15
            category_scores["Entertainment"] *= 0.9
        elif celebrity_focus_count > tech_focus_count * 1.5:
            # Article is more about celebrities than tech
            category_scores["Entertainment"] *= 1.15
            category_scores["Tech"] *= 0.9
    
    # Check for terrorism-related content with improved context analysis
    terrorism_keywords = [
        "terrorist", "terrorism", "attack", "bomb", "bombing", "explosion", "kill", "murder", "violence", 
        "hostage", "extremist", "militant", "gunman", "shooter", "massacre",
        "suicide bomb", "terror plot", "radicalized", "extremism", "jihad",
        "isis", "al-qaeda", "taliban", "boko haram", "terrorist group", "terror attack",
        "terrorist organization", "terrorist cell", "terrorist threat", "terrorist suspect"
    ]
    
    # Advanced semantic analysis for terrorism content
    terrorism_score = 0
    terrorism_context_words = [
        "killed", "injured", "casualties", "victims", "claimed responsibility", 
        "security", "threat", "police", "military", "intelligence", "authorities",
        "investigation", "suspect", "perpetrator", "motive", "radical", "violent",
        "detonated", "exploded", "shot", "attacked", "incident", "emergency"
    ]
    
    # First check for terrorism keywords
    terrorism_keyword_matches = []
    for keyword in terrorism_keywords:
        # Find all occurrences of the keyword
        start_pos = 0
        while True:
            pos = text_lower.find(keyword, start_pos)
            if pos == -1:
                break
            terrorism_keyword_matches.append((pos, keyword))
            start_pos = pos + len(keyword)
    
    # If we found terrorism keywords, analyze their context
    if terrorism_keyword_matches:
        # Sort by position
        terrorism_keyword_matches.sort()
        
        for pos, keyword in terrorism_keyword_matches:
            # Base score for presence
            base_score = 2
            
            # Check if terrorism terms appear in title (higher importance)
            if keyword in title_text:
                base_score += 3
            
            # Extract context around the keyword (50 chars before and after)
            start_context = max(0, pos - 50)
            end_context = min(len(text_lower), pos + len(keyword) + 50)
            context = text_lower[start_context:end_context]
            
            # Check for context words that strengthen terrorism association
            context_score = sum(1 for word in terrorism_context_words if word in context)
            base_score += min(3, context_score * 0.5)
            
            # Check for combinations of terrorism terms (indicates stronger relevance)
            for other_pos, other_keyword in terrorism_keyword_matches:
                if other_keyword != keyword:
                    # If keywords are within 100 characters of each other, they're likely related
                    if abs(other_pos - pos) < 100:
                        base_score += 1.5
            
            # Check for sentence structure indicating terrorism event
            # Look for patterns like "[group] attacked", "bombing in [location]", etc.
            attack_patterns = [
                r"attack(ed|s)? (in|on|at)", r"bomb(ed|ing)? (in|at)", 
                r"killed [0-9]+ people", r"claimed responsibility", 
                r"terrorist(s)? (attacked|killed|bombed)"
            ]
            
            for pattern in attack_patterns:
                if re.search(pattern, context):
                    base_score += 2
                    break
            
            terrorism_score += base_score
    
    if terrorism_score >= 6:  # Increased threshold for more accuracy
        # Determine if it's international (World) or domestic (Politics)
        world_indicators = [
            "foreign", "international", "overseas", "global", "abroad", "country",
            "border", "nation", "foreign country", "foreign national", "foreign government"
        ]
        
        # Check for specific country mentions that would indicate World category
        foreign_countries = [
            "afghanistan", "iraq", "syria", "yemen", "pakistan", "iran", "saudi", 
            "israel", "palestine", "egypt", "libya", "somalia", "nigeria", "france", 
            "germany", "uk", "britain", "russia", "china", "japan", "india", "australia"
        ]
        
        world_score = sum(1.5 for keyword in world_indicators if keyword in text_lower)
        world_score += sum(2 for country in foreign_countries if country in text_lower)
        
        # Check for domestic indicators
        domestic_indicators = [
            "domestic", "homeland", "national security", "fbi", "cia", "nsa", 
            "department of homeland security", "dhs", "local", "state", "federal"
        ]
        domestic_score = sum(1.5 for keyword in domestic_indicators if keyword in text_lower)
        
        if world_score >= domestic_score and world_score >= 3:
            category_scores["World"] += terrorism_score * 1.2
        else:
            category_scores["Politics"] += terrorism_score
    
    # Check for tech-business overlap with improved detection
    if category_scores["Tech"] > 0 and category_scores["Business"] > 0:
        tech_business_keywords = [
            "startup", "tech company", "funding", "venture capital", "ipo", "acquisition",
            "tech investment", "tech industry", "tech sector", "tech market", "tech stock",
            "tech giant", "silicon valley", "tech entrepreneur", "tech ceo", "tech founder",
            "tech unicorn", "tech valuation", "tech merger", "tech acquisition"
        ]
        
        tech_business_score = 0
        for keyword in tech_business_keywords:
            if keyword in text_lower:
                # Base score
                score = 1.5
                # Higher score for title mentions
                if keyword in title_text:
                    score += 1.5
                tech_business_score += score
        
        # If strong tech-business overlap, boost both categories with weighted distribution
        if tech_business_score >= 3:
            # Calculate relative strength of each category
            tech_strength = category_scores["Tech"] / (category_scores["Tech"] + category_scores["Business"])
            business_strength = 1 - tech_strength
            
            # Distribute boost proportionally to current category strength
            category_scores["Tech"] += tech_business_score * (0.4 + (tech_strength * 0.2))
            category_scores["Business"] += tech_business_score * (0.4 + (business_strength * 0.2))
    
    # Check for health-science overlap
    if category_scores["Health"] > 0 and category_scores["Science"] > 0:
        health_science_keywords = [
            "medical research", "clinical trial", "scientific study", "medical breakthrough",
            "medical discovery", "health research", "medical science", "biomedical",
            "pharmaceutical research", "drug development", "medical technology", "biotech"
        ]
        
        health_science_score = sum(1.5 for keyword in health_science_keywords if keyword in text_lower)
        
        if health_science_score >= 3:
            # Check for research focus (Science) vs medical application (Health)
            research_indicators = ["research", "study", "laboratory", "experiment", "scientist", "journal"]
            medical_indicators = ["treatment", "patient", "doctor", "hospital", "clinic", "therapy"]
            
            research_score = sum(1 for keyword in research_indicators if keyword in text_lower)
            medical_score = sum(1 for keyword in medical_indicators if keyword in text_lower)
            
            if research_score > medical_score:
                category_scores["Science"] += health_science_score * 0.7
                category_scores["Health"] += health_science_score * 0.3
            else:
                category_scores["Health"] += health_science_score * 0.7
                category_scores["Science"] += health_science_score * 0.3
    
    # Check for sports-entertainment overlap (e.g., esports, celebrity athletes)
    if category_scores["Sports"] > 0 and category_scores["Entertainment"] > 0:
        sports_entertainment_keywords = [
            "esports", "gaming tournament", "sports celebrity", "athlete endorsement",
            "sports movie", "sports documentary", "sports drama", "sports star",
            "famous athlete", "sports entertainment", "wrestling", "sports media"
        ]
        
        sports_ent_score = sum(1.5 for keyword in sports_entertainment_keywords if keyword in text_lower)
        
        if sports_ent_score >= 2:
            # Check context to determine primary category
            competition_indicators = ["match", "game", "tournament", "championship", "league", "score", "win"]
            entertainment_indicators = ["show", "celebrity", "fame", "movie", "tv", "media", "popular"]
            
            competition_score = sum(1 for keyword in competition_indicators if keyword in text_lower)
            entertainment_score = sum(1 for keyword in entertainment_indicators if keyword in text_lower)
            
            if competition_score > entertainment_score:
                category_scores["Sports"] += sports_ent_score * 0.8
                category_scores["Entertainment"] += sports_ent_score * 0.2
            else:
                category_scores["Entertainment"] += sports_ent_score * 0.8
                category_scores["Sports"] += sports_ent_score * 0.2
    
    # Apply cross-category analysis for articles that might span multiple topics
    # Identify potential cross-category articles by checking for strong signals in multiple categories
    sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    top_categories = sorted_scores[:3]  # Get top 3 categories
    
    # If the top 2 categories have close scores, analyze their relationship
    if len(top_categories) >= 2 and top_categories[0][1] > 0 and top_categories[1][1] > 0:
        top_category, top_score = top_categories[0]
        second_category, second_score = top_categories[1]
        
        # If scores are within 30% of each other, they might be related
        if second_score > top_score * 0.7:
            # Define pairs of categories that commonly overlap
            related_pairs = [
                ("Tech", "Business"),
                ("Health", "Science"),
                ("Politics", "World"),
                ("Sports", "Entertainment"),
                ("Business", "World"),
                ("Science", "Tech")
            ]
            
            # Check if our top categories form a related pair
            is_related_pair = False
            for cat1, cat2 in related_pairs:
                if (top_category == cat1 and second_category == cat2) or \
                   (top_category == cat2 and second_category == cat1):
                    is_related_pair = True
                    break
            
            if is_related_pair:
                # Look for specific cross-category keywords that would confirm the relationship
                cross_category_keywords = {
                    ("Tech", "Business"): ["tech company", "startup", "tech industry", "tech investment", "tech stock"],
                    ("Health", "Science"): ["medical research", "clinical trial", "scientific study", "medical breakthrough"],
                    ("Politics", "World"): ["foreign policy", "international relations", "diplomatic", "global politics"],
                    ("Sports", "Entertainment"): ["sports celebrity", "athlete endorsement", "sports media"],
                    ("Business", "World"): ["global market", "international trade", "foreign investment", "global economy"],
                    ("Science", "Tech"): ["applied science", "technology research", "scientific innovation"]
                }
                
                # Check for keywords in both directions
                pair1 = (top_category, second_category)
                pair2 = (second_category, top_category)
                
                keywords_to_check = []
                if pair1 in cross_category_keywords:
                    keywords_to_check = cross_category_keywords[pair1]
                elif pair2 in cross_category_keywords:
                    keywords_to_check = cross_category_keywords[pair2]
                
                # Count matches
                cross_matches = sum(1 for keyword in keywords_to_check if keyword in text_lower)
                
                # If we find cross-category keywords, boost both categories
                if cross_matches > 0:
                    boost = cross_matches * 2
                    category_scores[top_category] += boost
                    category_scores[second_category] += boost * 0.7
    
    # Load ML model for additional prediction
    model, tokenizer = load_model()
    
    # Get ML model prediction
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        
        # Get top 3 predictions from model for more comprehensive analysis
        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
        top_ml_categories = [CATEGORIES[idx] for idx in top_indices[0].tolist()]
        top_ml_scores = top_probs[0].tolist()
    
    # Combine keyword-based scores with model predictions using adaptive weighting
    # If keyword analysis is very confident (high score), give it more weight
    # If ML model is very confident (high probability), give it more weight
    keyword_confidence = max(category_scores.values()) if category_scores else 0
    ml_confidence = top_ml_scores[0] if top_ml_scores else 0
    
    # Normalize confidence scores
    keyword_weight = 0.6  # Base weight for keyword analysis
    ml_weight = 0.4      # Base weight for ML model
    
    # Adjust weights based on relative confidence
    if keyword_confidence > 20 and ml_confidence < 0.5:
        # Keyword analysis is much more confident
        keyword_weight = 0.8
        ml_weight = 0.2
    elif ml_confidence > 0.8 and keyword_confidence < 10:
        # ML model is much more confident
        keyword_weight = 0.3
        ml_weight = 0.7
    elif keyword_confidence > 15 and ml_confidence > 0.7:
        # Both methods are confident but potentially disagree
        # Check if they agree on the top category
        keyword_top_category = max(category_scores.items(), key=lambda x: x[1])[0]
        if keyword_top_category == top_ml_categories[0]:
            # Strong agreement - increase confidence
            keyword_weight = 0.5
            ml_weight = 0.5
        else:
            # Disagreement - need more nuanced analysis
            # Check semantic relationships between categories
            related_categories = {
                "Politics": ["World", "Business"],
                "Business": ["Tech", "World", "Politics"],
                "Tech": ["Business", "Science"],
                "Health": ["Science"],
                "Entertainment": ["Sports"],
                "Sports": ["Entertainment"],
                "World": ["Politics", "Business"],
                "Science": ["Tech", "Health"]
            }
            
            # If categories are semantically related, blend their scores
            if top_ml_categories[0] in related_categories.get(keyword_top_category, []):
                # Categories are related - blend scores with slight preference for keyword analysis
                keyword_weight = 0.55
                ml_weight = 0.45
                
                # Also boost both related categories
                category_scores[keyword_top_category] *= 1.1
                category_scores[top_ml_categories[0]] *= 1.1
    
    # Apply the weighted combination
    for i, category in enumerate(top_ml_categories):
        # Add model confidence score (scaled) to keyword-based score
        category_scores[category] += top_ml_scores[i] * 10 * ml_weight  # Scale factor to make comparable
    
    # Handle ambiguous content with multiple strong category signals
    sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    # If top two categories have very close scores (within 10% of each other)
    if len(sorted_scores) >= 2 and sorted_scores[0][1] > 0 and sorted_scores[1][1] > 0:
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        
        if second_score > top_score * 0.9:
            # Content is highly ambiguous between two categories
            top_category = sorted_scores[0][0]
            second_category = sorted_scores[1][0]
            
            # Check for specific disambiguation patterns
            disambiguators = {
                # Patterns that strongly indicate one category over another
                ("Politics", "World"): {
                    "Politics": ["domestic", "national", "constitutional", "election", "senate", "congress", "parliament"],
                    "World": ["international", "global", "foreign", "diplomatic", "embassy", "overseas"]
                },
                ("Business", "Tech"): {
                    "Business": ["profit", "revenue", "market", "stock", "investor", "financial", "economy"],
                    "Tech": ["algorithm", "software", "hardware", "developer", "programming", "code", "app"]
                },
                ("Health", "Science"): {
                    "Health": ["patient", "treatment", "doctor", "hospital", "medical", "disease", "symptom"],
                    "Science": ["research", "study", "experiment", "theory", "scientist", "laboratory"]
                },
                ("Sports", "Entertainment"): {
                    "Sports": ["game", "match", "player", "team", "score", "championship", "tournament"],
                    "Entertainment": ["movie", "film", "actor", "actress", "celebrity", "music", "album"]
                }
            }
            
            # Check both orderings of the category pair
            pair1 = (top_category, second_category)
            pair2 = (second_category, top_category)
            
            # Get the appropriate disambiguator
            disambiguator = None
            if pair1 in disambiguators:
                disambiguator = disambiguators[pair1]
            elif pair2 in disambiguators:
                disambiguator = disambiguators[pair2]
                # Swap the categories to match the disambiguator keys
                top_category, second_category = second_category, top_category
            
            if disambiguator:
                # Count matches for each category's disambiguating terms
                top_matches = sum(1 for term in disambiguator[top_category] if term in text_lower)
                second_matches = sum(1 for term in disambiguator[second_category] if term in text_lower)
                
                # Apply disambiguation boost
                if top_matches > second_matches:
                    category_scores[top_category] += (top_matches - second_matches) * 2
                elif second_matches > top_matches:
                    category_scores[second_category] += (second_matches - top_matches) * 2
    
    # Apply contextual coherence analysis
    # Check if the content has a coherent theme or is fragmented across categories
    coherence_threshold = 0.6  # Threshold for determining coherence
    top_score = max(category_scores.values()) if category_scores else 0
    total_score = sum(category_scores.values())
    
    if total_score > 0:
        coherence_ratio = top_score / total_score
        
        # If content is highly coherent (dominated by one category), boost that category
        if coherence_ratio > coherence_threshold:
            top_category = max(category_scores.items(), key=lambda x: x[1])[0]
            category_scores[top_category] *= 1.2  # Boost the dominant category
    
    # Find category with highest combined score
    best_category = max(category_scores.items(), key=lambda x: x[1])[0]
    
    return best_category

def categorize_articles(articles):
    """
    Categorize a list of articles
    
    Args:
        articles (list): List of article dictionaries
        
    Returns:
        list: List of articles with added 'category' field
    """
    # Load model
    load_model()
    
    # Process each article
    categorized_articles = []
    for article in tqdm(articles, desc="Categorizing articles"):
        # Use title and content for categorization
        title = article.get("title", "")
        content = article.get("content", "")
        text = f"{title}. {content}"
        
        # Predict category
        category = predict_category(text)
        
        # Predict subcategory based on main category
        subcategory = predict_subcategory(text, category)
        
        # Add category and subcategory to article
        article_with_category = article.copy()
        article_with_category["category"] = category
        if subcategory:
            article_with_category["subcategory"] = subcategory
        categorized_articles.append(article_with_category)
    
    return categorized_articles

def fine_tune_model(dataset_path):
    """
    Fine-tune the BERT model on a news dataset
    
    Args:
        dataset_path (str): Path to the dataset file
        
    Note: This is a placeholder for the actual fine-tuning code
    In a real implementation, you would use the Trainer API as shown in the example
    """
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    
    # Load dataset (AG News or custom dataset)
    # dataset = load_dataset("ag_news")
    dataset = load_dataset("csv", data_files=dataset_path)
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=len(CATEGORIES)
    )
    
    # Preprocess function
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")
    
    # Encode dataset
    encoded_dataset = dataset.map(preprocess, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=3
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained("./model/bert_news_classifier")
    tokenizer.save_pretrained("./model/bert_news_classifier")
    
    print("Model fine-tuning complete!")

if __name__ == "__main__":
    # Test the categorizer
    test_text = "Apple announces new iPhone with improved camera and longer battery life"
    category = predict_category(test_text)
    print(f"Predicted category: {category}")