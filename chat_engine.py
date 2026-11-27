"""
chat_engine.py  -  the "brain" of Radar's conversation.

What it does
  * small talk (hi / how are you / thanks / bye / who are you ...)
  * understands messy English: spelling mistakes, shouty letters, short forms,
    a few Hinglish words, and simple negation ("no fever", "I don't have cough")
  * lets the user say they are not satisfied and add more detail, then re-runs
    the prediction and tells them what changed
  * basic safety messages (emergency symptoms, self-harm)

It does NOT import Streamlit, so it can be tested on its own.
Only Python's built-in libraries are used (difflib does the spell tolerance),
so requirements.txt does not change.
"""
import random
import re
import difflib
from datetime import datetime, timezone, timedelta

# --------------------------------------------------------------------------
# 1. Static data
# --------------------------------------------------------------------------
CITIES = {"Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
          "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36), "Hyderabad": (17.39, 78.49),
          "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57), "Jaipur": (26.91, 75.79),
          "Lucknow": (26.85, 80.95), "Kochi": (9.93, 76.27), "Mangaluru": (12.91, 74.86)}

ALIASES = {"bangalore": "Bengaluru", "bengalooru": "Bengaluru", "bangaluru": "Bengaluru",
           "blr": "Bengaluru", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
           "cochin": "Kochi", "mangalore": "Mangaluru", "poona": "Pune", "hyd": "Hyderabad",
           "ahmadabad": "Ahmedabad"}

NEGATORS = {"no", "not", "never", "without", "none", "nothing", "neither", "nor", "nahi", "nahin"}

GREET = {"hi", "hii", "hello", "hey", "hola", "namaste", "namaskar", "yo", "hiya", "howdy",
         "heya", "hallo", "sup", "wassup", "greetings", "hlo", "helo", "hai"}

# words that must never be "spell-corrected" into a symptom
COMMON = set("""
have having with since feel feeling from been very much little bit some also when then than that this
what which while about again after before today yesterday days day week weeks night nights morning
evening last past started start getting get got really quite just more most lot lots many severe
there their they them were where would could should will shall might must into onto over under
never always often still even only other another same such each every both either neither
because though although however please thank thanks hello okay fine good great better worse
having having feels felt like need want help know think maybe sure yes yeah nope
name city live living staying today tonight""".split())

# not allowed as a person's name
NOT_NAMES = set("""
ok okay fine cool hmm yes yeah yep yup no nope nah skip thanks thank bye help hello hi hey sick ill unwell
tired good bad well better worse great sad happy scared worried confused anxious nervous dizzy hungry
sleepy cold hot weak having feeling suffering getting not very really just here back new student patient
user human who what how why when where radar bot robot name
""".split())

# typed phrase -> dataset column. (Only used if the column exists in your dataset.)
SYNONYMS = {
    # fever
    "fever": "high_fever", "feverish": "high_fever", "temperature": "high_fever",
    "high temperature": "high_fever", "bukhar": "high_fever", "bukhaar": "high_fever",
    "slight fever": "mild_fever", "low fever": "mild_fever", "low grade fever": "mild_fever",
    # tiredness
    "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue", "exhaustion": "fatigue",
    "no energy": "fatigue", "low energy": "fatigue", "worn out": "fatigue", "weak": "fatigue",
    "weakness": "fatigue", "kamzori": "fatigue", "thakan": "fatigue", "sleepy all day": "lethargy",
    "lethargic": "lethargy", "sluggish": "lethargy",
    # stomach
    "throwing up": "vomiting", "threw up": "vomiting", "throw up": "vomiting", "puking": "vomiting",
    "vomit": "vomiting", "vomits": "vomiting", "ulti": "vomiting",
    "feel like vomiting": "nausea", "feeling like vomiting": "nausea", "feel like throwing up": "nausea",
    "nauseous": "nausea", "nauseated": "nausea", "queasy": "nausea",
    "loose motion": "diarrhoea", "loose motions": "diarrhoea", "loose stools": "diarrhoea",
    "watery stool": "diarrhoea", "dast": "diarrhoea", "diarrhea": "diarrhoea", "motions": "diarrhoea",
    "stomach ache": "stomach_pain", "stomach hurts": "stomach_pain", "stomach cramps": "stomach_pain",
    "tummy ache": "stomach_pain", "tummy pain": "stomach_pain", "tummy hurts": "stomach_pain",
    "pet dard": "stomach_pain", "belly ache": "belly_pain", "stomach pain": "stomach_pain",
    "constipated": "constipation", "heartburn": "acidity", "acid reflux": "acidity", "sour belching": "acidity",
    "gas": "passage_of_gases", "gassy": "passage_of_gases", "bloating": "distention_of_abdomen",
    "bloated": "distention_of_abdomen", "upset stomach": "indigestion",
    "no appetite": "loss_of_appetite", "not hungry": "loss_of_appetite", "lost appetite": "loss_of_appetite",
    "poor appetite": "loss_of_appetite", "do not feel like eating": "loss_of_appetite",
    "do not want to eat": "loss_of_appetite",
    # throat / chest / nose
    "sore throat": "throat_irritation", "throat pain": "throat_irritation", "throat hurts": "throat_irritation",
    "scratchy throat": "throat_irritation", "throat infection": "throat_irritation",
    "sneezing": "continuous_sneezing", "sneeze": "continuous_sneezing", "sneezes": "continuous_sneezing",
    "coughing": "cough", "khansi": "cough", "khasi": "cough", "dry cough": "cough", "wet cough": "cough",
    "cold": "runny_nose", "jukam": "runny_nose", "running nose": "runny_nose", "runny nose": "runny_nose",
    "blocked nose": "congestion", "stuffy nose": "congestion", "nose block": "congestion",
    "nasal congestion": "congestion", "stuffed nose": "congestion",
    "phlegm": "phlegm", "mucus": "phlegm", "sputum": "phlegm", "balgam": "phlegm",
    "coughing blood": "blood_in_sputum", "coughing up blood": "blood_in_sputum", "blood in cough": "blood_in_sputum",
    "chest tightness": "chest_pain", "chest discomfort": "chest_pain", "pain in chest": "chest_pain",
    "chest hurts": "chest_pain", "chest pains": "chest_pain",
    "short of breath": "breathlessness", "shortness of breath": "breathlessness",
    "cannot breathe": "breathlessness", "cannot breath": "breathlessness", "trouble breathing": "breathlessness",
    "difficulty breathing": "breathlessness", "breathing problem": "breathlessness",
    "breathing difficulty": "breathlessness", "hard to breathe": "breathlessness", "wheezing": "breathlessness",
    "heart racing": "fast_heart_rate", "fast heartbeat": "fast_heart_rate", "racing heart": "fast_heart_rate",
    "heart beating fast": "fast_heart_rate",
    # pain
    "body ache": "muscle_pain", "body aches": "muscle_pain", "body pain": "muscle_pain", "body pains": "muscle_pain",
    "aching body": "muscle_pain", "badan dard": "muscle_pain", "muscle ache": "muscle_pain",
    "muscle aches": "muscle_pain",
    "headache": "headache", "head ache": "headache", "head pain": "headache", "head hurts": "headache",
    "sir dard": "headache", "sar dard": "headache", "migraine": "headache",
    "joints pain": "joint_pain", "joint ache": "joint_pain", "joints hurt": "joint_pain",
    "pain in joints": "joint_pain", "joint pains": "joint_pain",
    "knee hurts": "knee_pain", "backache": "back_pain", "back ache": "back_pain", "lower back pain": "back_pain",
    "pain behind eyes": "pain_behind_the_eyes", "pain behind the eyes": "pain_behind_the_eyes",
    "eye pain": "pain_behind_the_eyes", "pain behind my eyes": "pain_behind_the_eyes",
    "swollen joints": "swelling_joints", "joints swollen": "swelling_joints", "joint swelling": "swelling_joints",
    "stiff joints": "movement_stiffness", "morning stiffness": "movement_stiffness",
    # head / senses
    "dizzy": "dizziness", "giddy": "dizziness", "lightheaded": "dizziness", "light headed": "dizziness",
    "vertigo": "dizziness", "chakkar": "dizziness", "feeling faint": "dizziness",
    "blurry vision": "blurred_and_distorted_vision", "blurred vision": "blurred_and_distorted_vision",
    "red eyes": "redness_of_eyes", "eye redness": "redness_of_eyes",
    "watery eyes": "watering_from_eyes", "eyes watering": "watering_from_eyes",
    "yellow eyes": "yellowing_of_eyes", "yellowing eyes": "yellowing_of_eyes", "eyes yellow": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin", "yellowish skin": "yellowish_skin", "skin yellow": "yellowish_skin",
    "cannot smell": "loss_of_smell", "no smell": "loss_of_smell", "lost smell": "loss_of_smell",
    "loss of smell": "loss_of_smell",
    # skin
    "rash": "skin_rash", "rashes": "skin_rash", "red rash": "skin_rash", "itchy": "itching", "itch": "itching",
    "scratching": "itching", "khujli": "itching", "pimples": "pus_filled_pimples", "acne": "pus_filled_pimples",
    "blisters": "blister",
    # body temperature / sweat / chills
    "chilly": "chills", "shivers": "shivering", "shaking": "shivering", "kapkapi": "shivering",
    "sweat": "sweating", "sweats": "sweating", "sweaty": "sweating", "night sweats": "sweating",
    # urine
    "burning urination": "burning_micturition", "burning while peeing": "burning_micturition",
    "burning when i pee": "burning_micturition", "pain while urinating": "burning_micturition",
    "burning pee": "burning_micturition", "painful urination": "burning_micturition",
    "frequent urination": "polyuria", "peeing a lot": "polyuria", "urinating often": "polyuria",
    # misc
    "losing weight": "weight_loss", "lost weight": "weight_loss", "gaining weight": "weight_gain",
    "put on weight": "weight_gain", "dehydrated": "dehydration", "very thirsty": "dehydration",
    "anxious": "anxiety", "nervous": "anxiety", "panic": "anxiety", "depressed": "depression",
    "irritable": "irritability", "irritated": "irritability", "restless": "restlessness",
    "swollen glands": "swelled_lymph_nodes", "swollen lymph nodes": "swelled_lymph_nodes",
    "glands swollen": "swelled_lymph_nodes", "swollen legs": "swollen_legs", "legs swollen": "swollen_legs",
    "swelling in legs": "swollen_legs", "unsteady": "unsteadiness", "slurred speech": "slurred_speech",
}

VERSION = 3            # bump when the Chat class changes (app.py uses it to drop stale sessions)
LLM_MAX_CALLS = 40     # AI calls allowed per visitor session

# fever levels, "I'm burning up" style statements, swelling
SYNONYMS.update({
    "feverish": "mild_fever",
    "light fever": "mild_fever", "little fever": "mild_fever", "bit of fever": "mild_fever",
    "slight temperature": "mild_fever", "mild temperature": "mild_fever", "halka bukhar": "mild_fever",
    "feeling warm": "mild_fever", "feel warm": "mild_fever", "slightly warm": "mild_fever",
    "little warm": "mild_fever", "bit warm": "mild_fever", "body feels warm": "mild_fever",
    "burning up": "high_fever", "burning hot": "high_fever", "body is burning": "high_fever",
    "body burning": "high_fever", "burning with fever": "high_fever", "feel hot": "high_fever",
    "feeling hot": "high_fever", "feels hot": "high_fever", "body is hot": "high_fever",
    "body hot": "high_fever", "hot body": "high_fever", "hot to touch": "high_fever",
    "running a fever": "high_fever", "running fever": "high_fever", "running a temperature": "high_fever",
    "running temperature": "high_fever", "have a temperature": "high_fever", "got a temperature": "high_fever",
    "very high fever": "high_fever", "tez bukhar": "high_fever", "tez bukhaar": "high_fever",
    "swollen feet": "swollen_legs", "swollen ankles": "swollen_legs", "feet swollen": "swollen_legs",
    "ankles swollen": "swollen_legs", "swelling in feet": "swollen_legs", "swelling in ankles": "swollen_legs",
    "swollen foot": "swollen_legs", "puffy feet": "swollen_legs",
})

MILD_Q = {"light", "slight", "slightly", "mild", "mildly", "low", "little", "minor", "small", "bit",
          "halka", "halki"}
HIGH_Q = {"high", "severe", "extreme", "extremely", "terrible", "raging", "strong", "tez", "bad", "badly",
          "heavy", "tough", "worse"}
FEVER_WORDS = {"fever", "temperature", "temp", "bukhar", "bukhaar", "feverish"}

# "pain in my <body part>": any body part works, not only memorised phrases.
# value = (candidate dataset columns in order of preference, is it an exact match for that body part?)
PAIN_WORDS = ["pain", "pains", "painful", "paining", "ache", "aches", "aching", "achy", "hurt", "hurts",
              "hurting", "sore", "soreness", "cramp", "cramps", "cramping", "throbbing", "stabbing", "tender"]
PAIN_TYPOS = {"payn", "pian", "paine", "paiin", "pein", "paen"}
_J = ("joint_pain",)
BODY_PARTS = {
    "head": (("headache",), True), "forehead": (("headache",), True), "temple": (("headache",), True),
    "temples": (("headache",), True), "neck": (("neck_pain",), True), "throat": (("throat_irritation",), True),
    "chest": (("chest_pain",), True), "stomach": (("stomach_pain",), True), "tummy": (("stomach_pain",), True),
    "abdomen": (("abdominal_pain",), True), "belly": (("belly_pain",), True), "back": (("back_pain",), True),
    "spine": (("back_pain",), False), "knee": (("knee_pain",), True), "knees": (("knee_pain",), True),
    "hip": (("hip_joint_pain",), True), "hips": (("hip_joint_pain",), True),
    "joint": (_J, True), "joints": (_J, True),
    "ankle": (_J, False), "ankles": (_J, False), "wrist": (_J, False), "wrists": (_J, False),
    "elbow": (_J, False), "elbows": (_J, False), "shoulder": (_J, False), "shoulders": (_J, False),
    "finger": (_J, False), "fingers": (_J, False), "toe": (_J, False), "toes": (_J, False),
    "foot": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "feet": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "heel": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "heels": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "sole": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "soles": (("painful_walking", "joint_pain", "muscle_pain"), False),
    "leg": (("muscle_pain", "joint_pain"), False), "legs": (("muscle_pain", "joint_pain"), False),
    "calf": (("muscle_pain",), False), "calves": (("muscle_pain",), False),
    "thigh": (("muscle_pain",), False), "thighs": (("muscle_pain",), False),
    "arm": (("muscle_pain",), False), "arms": (("muscle_pain",), False),
    "hand": (("muscle_pain",), False), "hands": (("muscle_pain",), False),
    "muscle": (("muscle_pain",), True), "muscles": (("muscle_pain",), True), "body": (("muscle_pain",), True),
    "eye": (("pain_behind_the_eyes",), False), "eyes": (("pain_behind_the_eyes",), False),
}


def is_pain_word(w):
    return w in PAIN_WORDS or w in PAIN_TYPOS or bool(close(w, PAIN_WORDS))


# --------------------------------------------------------------------------
# 2. Small-talk phrases
# --------------------------------------------------------------------------
INTENT_PHRASES = {
    "how_are_you": ["how are you", "how are you doing", "how is it going", "how are things",
                    "how do you do", "how have you been", "what is up", "whats up", "how you doing",
                    "how is everything", "hru"],
    "thanks": ["thanks", "thank you", "thanks a lot", "thank you so much", "thanks so much", "many thanks",
               "thankyou", "tysm", "ty", "thank you radar"],
    "bye": ["bye", "goodbye", "good bye", "see you", "see you later", "take care", "good night",
            "talk later", "cya", "gotta go"],
    "identity": ["who are you", "what are you", "what is your name", "your name", "what can you do",
                 "how do you work", "how does this work", "what is this", "are you a robot", "are you human",
                 "are you real", "are you a bot", "are you ai", "tell me about yourself"],
    "help": ["help", "help me", "i need help", "what should i do", "how to use this", "how do i use this"],
    "praise": ["good bot", "you are great", "you are awesome", "you are helpful", "you are amazing",
               "well done", "good job", "nice work", "love this", "this is cool", "you are good", "you are nice"],
    "vague_ill": ["not feeling well", "do not feel well", "feeling sick", "feel sick", "i am sick", "i am ill",
                  "feeling ill", "feel ill", "unwell", "not well", "feeling bad", "feel bad",
                  "not feeling good", "do not feel good", "under the weather", "not okay", "not fine",
                  "health problem", "health issue", "i have a problem", "something is wrong", "feeling off"],
    "ack": ["ok", "okay", "k", "fine", "cool", "alright", "great", "nice", "good", "understood", "got it",
            "makes sense", "i see", "sure", "hmm", "hmmm", "awesome", "perfect", "oh", "oh ok", "oh okay"],
}
SHORT_ONLY = {"ack": 3, "praise": 6, "help": 6}       # these only count in short messages

DISSATISFIED_WORDS = ["wrong", "incorrect", "inaccurate", "useless", "unsatisfied", "dissatisfied",
                      "unhappy", "disappointed", "doubtful"]
GOOD_WORDS = ["satisfied", "helpful", "helping", "helped", "right", "correct", "accurate", "convincing",
              "enough", "good", "clear", "sure"]
MORE_RE = re.compile(r"\b(more details?|more info|add more|add details?|something else|other symptoms?|"
                     r"more symptoms?|tell you more|explain more|forgot|missed|did not mention)\b")

DONE_RE = re.compile(r"\b(that(?:'s| is|s)?\s+(?:all|it)|done|nothing else|no more|nothing more|"
                     r"analy[sz]e|go ahead|proceed|check now|run it|submit)\b")
SELF_HARM_RE = re.compile(r"\b(suicid\w*|kill myself|end my life|want to die|wanna die|do not want to live|"
                          r"self harm|hurt myself|cut myself|end it all)\b")
EMERGENCY_RE = re.compile(r"\b(chest pain|cannot breathe|cannot breath|trouble breathing|difficulty breathing|"
                          r"hard to breathe|struggling to breathe|unconscious|passed out|fainted|seizure|"
                          r"convulsion|heavy bleeding|bleeding a lot|vomiting blood|coughing blood|"
                          r"blood in vomit|stroke|face drooping|slurred speech|paralys?zed)\b")
SEVERE_RE = re.compile(r"\b(severe|unbearable|worst|terrible|excruciating|very bad|extreme)\b")
DURATION_RE = re.compile(r"\b(\d+|a|one|two|three|four|five|six|seven|few)\s+(day|days|week|weeks|month|months)\b")
NAME_RE = re.compile(r"\b(?:my name is|name is|i am|call me|this is|it is|myself)\s+([a-z]{2,20})\b")


# --------------------------------------------------------------------------
# 3. Text helpers
# --------------------------------------------------------------------------
def normalize(text):
    """lower-case, expand short forms, squeeze repeated letters ("feverrrr")."""
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"\b(can't|cant|cannot|can not)\b", "cannot", t)
    t = re.sub(r"\b(don't|dont|doesn't|doesnt|didn't|didnt)\b", "do not", t)
    t = re.sub(r"\b(haven't|havent|hasn't|hasnt)\b", "have not", t)
    t = re.sub(r"\b(isn't|isnt|aren't|arent|wasn't|wasnt|won't|wont)\b", "is not", t)
    t = re.sub(r"\b(i'm|im)\b", "i am", t)
    t = re.sub(r"\bhw\b", "how", t)
    t = re.sub(r"\bgud\b", "good", t)
    t = re.sub(r"\b(wat|wht)\b", "what", t)
    t = re.sub(r"\br\b", "are", t)
    t = re.sub(r"\bu\b", "you", t)
    t = re.sub(r"\bur\b", "your", t)
    t = re.sub(r"\b(pls|plz)\b", "please", t)
    t = re.sub(r"\b(thx|thnx|tnx)\b", "thanks", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return t


def norm_phrase(feature):
    """'pain_behind_the_eyes' -> 'pain behind the eyes' (also cleans odd column names)."""
    s = feature.lower().replace("_", " ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_sim(a, b):
    """0..1 similarity of two words; swapped letters ('jonit' vs 'joint') count as a near match."""
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if len(a) >= 4 and a[0] == b[0] and sorted(a) == sorted(b):
        r = max(r, 0.88)
    return r


def close(word, vocab, cutoff=0.8):
    """Spell-tolerant match for one word. Needs >=5 letters and the same first letter
    (typos rarely change the first letter - this stops 'never' being read as 'fever')."""
    if word in vocab:
        return word
    if len(word) < 5:
        return None
    for m in difflib.get_close_matches(word, vocab, n=3, cutoff=cutoff):
        if m[0] == word[0]:
            return m
    for m in vocab:                       # swapped-letter typos such as 'vomitign'
        if m[0] == word[0] and len(m) == len(word) and word_sim(word, m) >= 0.88:
            return m
    return None


def exact_phrase(t, phrases):
    return any(re.search(r"\b" + re.escape(p) + r"\b", t) for p in phrases)


def best_fuzzy_intent(t, tokens, candidates):
    """Spelling-tolerant intent match for SHORT messages. Returns the single best intent or None."""
    t = t.strip()
    if len(tokens) > 6 or len(t) < 4:
        return None
    best, best_score = None, 0.0
    for intent in candidates:
        for p in INTENT_PHRASES[intent]:
            r = difflib.SequenceMatcher(None, t, p).ratio()
            if r >= 0.82 and r > best_score:
                best, best_score = intent, r
    return best


def join_and(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def ist_hour():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).hour


def dissatisfied(t, tokens):
    if any(close(tok, DISSATISFIED_WORDS) for tok in tokens):
        return True
    if any(tok in NEGATORS for tok in tokens) and any(close(tok, GOOD_WORDS) for tok in tokens):
        return True
    return bool(MORE_RE.search(t))


# --------------------------------------------------------------------------
# 4. The conversation
# --------------------------------------------------------------------------
class Chat:
    def __init__(self, features, predict_fn, report_fn=None, llm=None):
        self.version = VERSION
        self.llm = llm                    # optional llm_layer.LLM object
        self.llm_calls = 0
        self.features = list(features)
        self.predict_fn = predict_fn      # symptoms list -> top-3 result list
        self.report_fn = report_fn        # (city, disease) -> None  (logs one anonymous report)
        self.last = {}
        self.phrases, self.by_n = self._build_index()
        self.reset()

    # ---------- setup ----------
    def _build_index(self):
        ph = {}
        for f in self.features:
            ph.setdefault(norm_phrase(f), f)
        for p, f in SYNONYMS.items():
            if f in self.features:
                ph.setdefault(p, f)
        by_n = {}
        for p in ph:
            by_n.setdefault(len(p.split()), []).append(p)
        return ph, by_n

    def reset(self):
        self.msgs = []
        self.name = ""
        self.city = ""
        self.symptoms = []
        self.rounds = 0
        self.prev_top = None
        self.fails = 0
        self.duration = 0
        self.severe = False
        self.warned = False
        self.fever_q = False
        self.pain_q = False
        self.result_summary = None
        h = ist_hour()
        tod = "Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 17 else \
              "Good evening" if 17 <= h < 22 else "Hello"
        self.bot(self.say("open", [
            tod + "! 👋 I'm Radar, your friendly health sidekick. What should I call you?",
            "Hey there! I'm Radar. I help you make sense of symptoms and show you *why* I think what I think. "
            "What's your name?",
            "Hi! Radar here 🩺 Tell me your first name and we'll get started."]))

    # ---------- small utilities ----------
    @property
    def stage(self):
        if not self.name:
            return "name"
        if not self.city:
            return "city"
        return "result" if self.rounds else "symptoms"

    def n(self):
        return " " + self.name if self.name else ""

    def label(self, f):
        return norm_phrase(f)

    def bot(self, text):
        self.msgs.append(("assistant", text))

    def say(self, key, options, **kw):
        pool = [o for o in options if o != self.last.get(key)] or options
        choice = random.choice(pool)
        self.last[key] = choice
        return choice.format(n=self.n(), name=self.name, city=self.city, **kw)

    def nudge(self):
        s = self.stage
        if s == "name":
            return self.say("nudge_name", ["First things first: what should I call you?",
                                           "Tell me your first name and we'll get going.", "So, what's your name?"])
        if s == "city":
            return self.say("nudge_city", ["Which city are you in right now?",
                                           "Where are you based? (for example Bengaluru, Mumbai or Chennai)"])
        if s == "symptoms":
            return self.say("nudge_sym", ["Whenever you're ready, tell me what symptoms you're having.",
                                          "So, what's been bothering you?"])
        return self.say("nudge_res", ["Anything else you'd like me to look at? Add more symptoms, or type **restart**.",
                                      "If you remember anything else, tell me. Or type **restart** for a fresh check."])

    # ---------- understanding the user's text ----------
    def fuzzy_words(self, words):
        """Match a multi-word window word-by-word, tolerating typos inside each word."""
        best, best_score = None, 0.0
        for cand in self.by_n.get(len(words), []):
            score, ok = 0.0, True
            for a, b in zip(words, cand.split()):
                if a == b:
                    score += 1
                    continue
                if len(a) < 4 or len(b) < 4 or a[0] != b[0]:
                    ok = False
                    break
                r = word_sim(a, b)
                if r < 0.75:
                    ok = False
                    break
                score += r
            score /= len(words)
            if ok and score >= 0.78 and score > best_score:
                best, best_score = cand, score
        return best

    def lookup(self, window, n):
        if window in self.phrases:
            return self.phrases[window], None
        if n == 1:
            if len(window) < 5 or window in COMMON:
                return None
            m = close(window, self.by_n.get(1, []), 0.8)
        else:
            m = self.fuzzy_words(window.split())
        return (self.phrases[m], m) if m else None

    def body_hits(self, tokens, used):
        """'pain in my feet', 'my legs are aching', 'sore ankle' ... any body part + any pain word nearby."""
        pains = [i for i, w in enumerate(tokens) if not used[i] and is_pain_word(w)]
        if not pains:
            return []
        keys = list(BODY_PARTS)
        hits = []
        linked = {}                                        # body-part index -> the pain word it belongs to
        for bi, w in enumerate(tokens):
            if used[bi] or w in COMMON:
                continue
            key = w if w in BODY_PARTS else close(w, keys)
            if not key:
                continue
            near = [pi for pi in pains if pi != bi and abs(pi - bi) <= 4]
            if near:
                pi = min(near, key=lambda q: abs(q - bi))
            elif bi >= 2 and tokens[bi - 1] in ("and", "or", "also") and (bi - 2) in linked:
                pi = linked[bi - 2]                        # "pain in my ankle AND heel"
            else:
                continue
            if key == "back" and abs(pi - bi) != 1 and \
                    (bi == 0 or tokens[bi - 1] not in ("my", "the", "lower", "upper", "whole", "entire")):
                continue                                   # "I'm back and my head hurts" is not back pain
            cands, exact = BODY_PARTS[key]
            feat = next((f for f in cands if f in self.features), None)
            if not feat:
                continue
            linked[bi] = pi
            hits.append((min(pi, bi), max(pi, bi) + 1, feat, None, None if exact else {"approx": key}))
        return hits

    def match(self, tokens):
        used = [False] * len(tokens)
        found = []
        for n in (5, 4, 3, 2, 1):
            for i in range(len(tokens) - n + 1):
                if any(used[i:i + n]):
                    continue
                window = " ".join(tokens[i:i + n])
                hit = self.lookup(window, n)
                if hit:
                    feat, fuzzy = hit
                    found.append((i, i + n, feat, fuzzy))
                    for k in range(i, i + n):
                        used[k] = True
        hits = []
        for i, j, feat, fuzzy in found:
            extra = None
            if feat == "high_fever":                       # "light fever" -> mild, plain "fever" -> high (+ ask)
                words, before = tokens[i:j], tokens[max(0, i - 3):i]
                if any(w in FEVER_WORDS for w in words):
                    if any(w in MILD_Q for w in before) and not any(w in HIGH_Q for w in words + before) \
                            and "mild_fever" in self.features:
                        feat = "mild_fever"
                    elif all(w in FEVER_WORDS for w in words) and not any(w in HIGH_Q for w in before):
                        extra = {"generic_fever": True}
            hits.append((i, j, feat, fuzzy, extra))
        hits += self.body_hits(tokens, used)
        hits.sort(key=lambda h: h[0])
        return hits

    def temp_feature(self, t):
        """'temperature is 101', '102 degrees', 'fever of 38.5' -> mild_fever / high_fever."""
        for m in re.finditer(r"(\d{2,3}(?:\.\d+)?)\s*(?:(\u00b0|degrees?|deg|celsius|fahrenheit|f|c)\b)?", t):
            val, unit = float(m.group(1)), m.group(2)
            before = t[max(0, m.start() - 30):m.start()]
            if not (unit or re.search(r"(temp|temperature|fever|thermometer|reading|reads)", before)):
                continue
            if 94 <= val <= 110:
                c = (val - 32) * 5 / 9
            elif 34 <= val <= 43:
                c = val
            else:
                continue
            if c >= 38.5:
                return "high_fever"
            if c >= 37.6:
                return "mild_fever"
        return None

    def find_city(self, tokens):
        names = {c.lower(): c for c in CITIES}
        for tok in tokens:
            if tok in names:
                return names[tok]
            if tok in ALIASES:
                return ALIASES[tok]
        vocab = list(names) + list(ALIASES)
        for tok in tokens:
            m = close(tok, vocab, 0.8)
            if m:
                return names.get(m) or ALIASES[m]
        return None

    def is_greet_token(self, tok):
        return tok in GREET or (len(tok) >= 4 and bool(close(tok, list(GREET), 0.8)))

    def find_intents(self, t, tokens):
        found = []
        if re.match(r"^\s*good\s+(morning|afternoon|evening)", t) or any(self.is_greet_token(x) for x in tokens[:2]):
            found.append("greeting")
        pool = []
        for intent in ("how_are_you", "identity", "help", "vague_ill", "praise", "thanks", "ack", "bye"):
            limit = SHORT_ONLY.get(intent)
            if limit and len(tokens) > limit:
                continue
            pool.append(intent)
            if exact_phrase(t, INTENT_PHRASES[intent]):
                found.append(intent)
        if len(found) == 0 or found == ["greeting"] and len(tokens) <= 2:
            guess = best_fuzzy_intent(t, tokens, pool)
            if guess and guess not in found:
                found.append(guess)
        if "ack" in found and len(found) > 1:
            found.remove("ack")
        return found

    def extract_name(self, t, tokens, intents):
        m = NAME_RE.search(t)
        cand = m.group(1) if m else None
        if not cand and set(intents) <= {"greeting"}:
            rest = [x for x in tokens if not self.is_greet_token(x) and x not in ("there", "radar", "everyone")]
            if 1 <= len(rest) <= 2:
                cand = rest[0]
        if not cand or cand in NOT_NAMES or cand in COMMON:
            return None
        if self.lookup(cand, 1) or self.find_city([cand]):
            return None
        return cand.title()

    def parse(self, t):
        stage = self.stage
        clauses = re.split(r"[.,;:!?\n]+|\b(?:but|however|although|though|except)\b", t)
        present, negated, fixes, approx, all_tokens = [], [], [], [], []
        generic = False
        for cl in clauses:
            toks = re.findall(r"[a-z]+", cl)
            all_tokens += toks
            for i, j, feat, fuzzy, extra in self.match(toks):
                neg = any(x in NEGATORS for x in toks[max(0, i - 3):i])
                (negated if neg else present).append(feat)
                typed = " ".join(toks[i:j])
                if fuzzy and typed != fuzzy:
                    fixes.append((typed, feat))
                if extra and not neg:
                    if "approx" in extra:
                        approx.append((extra["approx"], feat))
                    if extra.get("generic_fever"):
                        generic = True
        tf = self.temp_feature(t)
        if tf and tf in self.features:
            present = [f for f in present if f not in ("high_fever", "mild_fever")] + [tf]
            generic = False
        present = list(dict.fromkeys(present))
        if "high_fever" in present and "mild_fever" in present:
            present.remove("mild_fever")
        if "high_fever" not in present:
            generic = False
        negated = [f for f in dict.fromkeys(negated) if f not in present]
        intents = self.find_intents(t, all_tokens)
        info = {"tokens": all_tokens, "present": present, "negated": negated, "fixes": fixes,
                "intents": intents, "city": None, "name": None, "done": False,
                "approx": approx, "generic_fever": generic}
        if not self.city:
            info["city"] = self.find_city(all_tokens)
        if stage == "name":
            info["name"] = self.extract_name(t, all_tokens, intents)
            if re.search(r"\bskip\b", t):
                info["name"] = "Friend"
        if stage in ("symptoms", "result"):
            bare_no = len(all_tokens) <= 2 and all_tokens[:1] and all_tokens[0] in ("no", "nope", "nah", "nothing") \
                and not negated
            info["done"] = bool((DONE_RE.search(t) or bare_no) and not present and not negated)
        return info

    def note_details(self, t):
        m = DURATION_RE.search(t)
        if m:
            words = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "few": 3}
            num = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
            unit = 1 if m.group(2).startswith("day") else 7 if m.group(2).startswith("week") else 30
            self.duration = max(self.duration, num * unit)
        if SEVERE_RE.search(t):
            self.severe = True

    # ---------- main entry point ----------
    def handle(self, text):
        text = text.strip()
        if not text:
            return
        self.msgs.append(("user", text))
        t = normalize(text)

        if re.search(r"\b(restart|start over|reset|new check)\b", t):
            self.reset()
            return
        if SELF_HARM_RE.search(t):
            self.bot("I'm really sorry you're going through something this heavy, and I'm glad you said it out "
                     "loud. 💙 You deserve support from a real person right now. If you're in immediate danger, "
                     "please call **112**. You can also talk to a trained counsellor for free, any time, on "
                     "**Tele-MANAS: 14416** (India, 24x7). If someone is near you, a family member or friend, "
                     "please tell them how you feel or ask them to stay with you. I'm a small student-built app "
                     "and can't give the help you deserve here, but I'm happy to keep chatting.")
            return

        stage = self.stage
        info = self.parse(t)
        self.note_details(t)
        if EMERGENCY_RE.search(t) and not self.warned:
            self.warned = True
            self.bot("🚨 **Quick important note:** what you've described (chest pain, trouble breathing, fainting, "
                     "heavy bleeding and similar) can be an emergency. Please don't wait on an app. Call **112** or "
                     "get to the nearest hospital now. I'll keep helping here, but real help comes first.")

        # --- optional AI conversation layer (falls back to the rules below on any problem) ---
        if self.llm is not None and self.llm_calls < LLM_MAX_CALLS:
            out = self._ask_llm(info)
            if out is not None and self._handle_llm(info, out):
                return
        return self._rules(t, info, stage)

    # ---------- AI layer helpers ----------
    def llm_status(self):
        if self.llm is None:
            return "rules"
        return "ai" if self.llm_calls < LLM_MAX_CALLS else "limit"

    def _ask_llm(self, info):
        self.llm_calls += 1
        history = [{"role": r, "content": c[:600]} for r, c in self.msgs[-14:]
                   if r in ("user", "assistant") and isinstance(c, str)]
        ctx = {"known_name": bool(self.name), "city": self.city or None,
               "symptom_keys_so_far": list(self.symptoms),
               "missing": [k for k, v in (("name", self.name), ("city", self.city), ("symptoms", self.symptoms)) if not v],
               "analysis_already_shown": bool(self.rounds), "last_result": self.result_summary,
               "rule_engine_saw": {"present": info["present"], "negated": info["negated"]}}
        try:
            return self.llm.turn(history, ctx)
        except Exception:
            return None

    def _valid_name(self, s):
        if not isinstance(s, str):
            return None
        m = re.match(r"\s*([A-Za-z][A-Za-z'-]{1,19})", s)
        if not m:
            return None
        cand = m.group(1).lower()
        if cand in NOT_NAMES or cand in COMMON or self.lookup(cand, 1) or self.find_city([cand]):
            return None
        return cand.title()

    def _valid_city(self, s):
        if not isinstance(s, str):
            return None
        toks = re.findall(r"[a-z]+", normalize(s))
        return self.find_city(toks) if toks else None

    def _handle_llm(self, info, out):
        """Apply a validated AI turn. Returns False if it can't be used (then the rule-based chat answers)."""
        reply = out.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            return False
        reply = reply.strip()[:900]
        if self.stage == "name":
            nm = self._valid_name(out.get("name")) or info["name"]
            if nm:
                self.name = nm
        if not self.city:
            c = self._valid_city(out.get("city")) or info["city"]
            if c:
                self.city = c
        reply = reply.replace("{name}", self.name) if self.name else re.sub(r"[ ,]*\{name\}", "", reply)
        add = [f for f in out.get("add", []) if f in self.features] + list(info["present"])
        rem = [f for f in out.get("remove", []) if f in self.features] + list(info["negated"])
        add = [f for f in dict.fromkeys(add) if f not in rem]
        new = [f for f in add if f not in self.symptoms]
        self.symptoms += new
        removed = [f for f in dict.fromkeys(rem) if f in self.symptoms]
        self.symptoms = [f for f in self.symptoms if f not in removed]
        ready = bool(out.get("ready")) or info["done"]
        complete = bool(self.name and self.city and self.symptoms)
        if complete and ((ready and self.rounds == 0) or (self.rounds and (new or removed))):
            self.finish(new, removed, lead=reply)
        else:
            self.bot(reply)
        return True

    # ---------- rule-based conversation ----------
    def _rules(self, t, info, stage):
        has_symptoms = bool(info["present"] or info["negated"])

        # --- answer to "where does it hurt?" (e.g. just "my feet") ---
        if self.pain_q:
            self.pain_q = False
            if not has_symptoms:
                alt = self.parse("pain in " + t)
                if alt["present"]:
                    info["present"], info["approx"], info["fixes"] = alt["present"], alt["approx"], alt["fixes"]
                    has_symptoms = True

        # --- "is the fever mild or high?" follow-up answer ---
        if self.fever_q:
            self.fever_q = False
            toks = info["tokens"]
            if not has_symptoms and len(toks) <= 6 and "high_fever" in self.symptoms:
                mild, high = any(w in MILD_Q for w in toks), any(w in HIGH_Q for w in toks)
                if mild and not high and "mild_fever" in self.features:
                    self.symptoms = ["mild_fever" if f == "high_fever" else f for f in self.symptoms]
                    return self.bot("Thanks, I've changed that to **mild fever**. Any other symptoms? "
                                    "Say **that's all** when you're ready.")
                if high and not mild:
                    return self.bot("Okay, I'll keep it as **high fever**. Any other symptoms? "
                                    "Say **that's all** when you're ready.")

        # --- after a result: unhappy / yes / no ---
        if stage == "result" and not has_symptoms:
            if dissatisfied(t, info["tokens"]):
                return self.more_details(prefix=True)
            first = info["tokens"][:1]
            if len(info["tokens"]) <= 3 and first:
                if first[0] in ("yes", "yeah", "yep", "yup", "sure"):
                    return self.more_details()
            if info["done"] or (len(info["tokens"]) <= 3 and first and first[0] in ("no", "nope", "nah")):
                return self.bot(self.say("res_no", [
                    "Alright{n}! 💙 Rest well, drink plenty of fluids, and see a doctor if things get worse.",
                    "Okay{n}, take care of yourself! If it lingers or worsens, please see a doctor."]))

        content = (has_symptoms or info["name"] or (info["city"] and stage in ("name", "city"))
                   or (stage == "symptoms" and info["done"]))
        if info["intents"] and not content:
            return self.smalltalk(info["intents"], stage)

        # --- absorb what we learned ---
        got_name = bool(info["name"])
        got_city = bool(info["city"]) and not self.city
        if got_name:
            self.name = info["name"]
        if got_city:
            self.city = info["city"]
        new = [f for f in info["present"] if f not in self.symptoms]
        self.symptoms += new
        removed = [f for f in info["negated"] if f in self.symptoms]
        self.symptoms = [f for f in self.symptoms if f not in removed]

        if stage == "result":
            if new or removed:
                return self.finish(new, removed)
            if info["negated"]:
                return self.bot(self.ack_symptoms([], [], info) + " Anything else you've noticed?")
            return self.bot(self.say("res_fb", [
                "I'm not sure what to add from that. If you remember another symptom, describe it in your own "
                "words, or tap **Not satisfied** on the result. Type **restart** for a fresh check.",
                "Hmm, I couldn't pick a new symptom out of that. Try something like *'also feeling dizzy'*, or "
                "type **restart** to begin again."]))

        # --- still collecting info ---
        if self.name and self.city and self.symptoms and info["done"]:
            return self.finish(new, removed)

        bits = []
        if got_name:
            bits.append(self.say("name_ack", ["Nice to meet you, {name}! 😊", "Great name, {name}!",
                                              "Welcome, {name}! 🙌"]))
        if got_city:
            bits.append(self.say("city_ack", ["{city}, nice!", "Got it, {city}.", "Ah, {city}! 👍"]))
        if new or removed or info["negated"]:
            bits.append(self.ack_symptoms(new, removed, info))
        if "greeting" in info["intents"] and not got_name and stage != "name":
            bits.insert(0, "Hey! 👋")

        if not self.name:
            bits.append(self.nudge() if (bits or info["intents"]) else self.say("name_fb", [
                "Sorry, I didn't catch a name there. Just your first name is fine. 🙂",
                "Hmm, I need a first name to call you by. What should it be?"]))
        elif not self.city:
            if bits or info["intents"]:
                bits.append(self.nudge())
            else:
                bits.append(self.say("city_fb", [
                    "I don't have that city on my map yet. I cover: " + ", ".join(CITIES) + ". Which is closest to you?",
                    "Couldn't place that one. Try one of these: " + ", ".join(CITIES) + "."]))
        elif not self.symptoms:
            if info["done"]:
                bits.append("I need at least one symptom to work with, even a small one. What are you feeling?")
            elif new or removed or info["negated"]:
                bits.append("Which symptoms do you have?")
            elif got_city or got_name:
                bits.append(self.say("ask_sym", [
                    "Now tell me what's bothering you, in your own words. Something like *'fever, headache and "
                    "body ache'* works, and spelling mistakes are fine.",
                    "So how are you feeling, {name}? List whatever symptoms you have, and don't worry about typos.",
                    "What symptoms are you having? Write it however it comes to you, I'll figure it out."]))
            elif any(is_pain_word(x) for x in info["tokens"]):
                self.pain_q = True
                bits.append(self.say("pain_q", [
                    "Sorry you're in pain. Where does it hurt? A body part helps, like head, chest, stomach, back, "
                    "joints, legs or feet.",
                    "Ouch, that sounds uncomfortable. Where exactly is the pain?"]))
            else:
                self.fails += 1
                if self.fails >= 2:
                    bits.append("I still can't spot a symptom. Try plain words like *cough*, *vomiting*, *chills*, "
                                "*joint pain*, *headache* or *loose motion*.")
                else:
                    bits.append(self.say("sym_fb", [
                        "I couldn't match that to a symptom I know. Could you describe how it feels, e.g. "
                        "*'pain in my joints and feeling feverish'*?",
                        "Hmm, that one went over my head 😅 Try rephrasing, for example *'high fever and headache'*."]))
        else:
            bits.append("Any other symptoms? Even small ones help." if len(self.symptoms) < 2 else
                        self.say("more_q", ["Anything else? Say **that's all** when you're ready and I'll run the analysis.",
                                            "More symptoms, or shall we go ahead? Say **done** when ready.",
                                            "Tell me more if you like, or type **that's all** and I'll analyse."]))
        self.bot(" ".join(bits))

    # ---------- replies ----------
    def ack_symptoms(self, new, removed, info):
        bits = []
        if new:
            names = join_and(["**" + self.label(f) + "**" for f in new])
            bits.append(self.say("ack_new", ["Got it, I've noted {names}.", "Okay, {names} added to the list.",
                                             "Thanks{n}, {names} noted.", "Understood: {names}."], names=names))
        if removed:
            names = join_and(["**" + self.label(f) + "**" for f in removed])
            bits.append(self.say("ack_rem", ["Okay, I've taken out {names}.", "No problem, {names} removed."],
                                 names=names))
        only_neg = [f for f in info["negated"] if f not in removed]
        if only_neg:
            names = join_and(["**" + self.label(f) + "**" for f in only_neg])
            bits.append(self.say("ack_neg", ["Good to know, no {names}.", "Okay, noted that you don't have {names}."],
                                 names=names))
        if new and info["fixes"]:
            typed, feat = info["fixes"][0]
            bits.append("(I read “" + typed + "” as **" + self.label(feat) + "**. Tell me if that's wrong.)")
        if new and info.get("approx"):
            part, feat = info["approx"][0]
            if feat in new:
                bits.append("(I don't have a separate “" + part + " pain” in my data, so I noted it as **" +
                            self.label(feat) + "**, the closest match.)")
        if "high_fever" in new and info.get("generic_fever"):
            bits.append("Is it a mild fever or a really high one?")
            self.fever_q = True
        return " ".join(bits)

    def smalltalk(self, intents, stage):
        order = ["greeting", "how_are_you", "identity", "help", "vague_ill", "praise", "thanks", "ack", "bye"]
        intents = [i for i in order if i in intents][:2]
        h = ist_hour()
        tod = "Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 17 else \
              "Good evening" if 17 <= h < 22 else "Hello"
        parts = []
        closing = False
        for it in intents:
            if it == "greeting":
                parts.append(self.say("g", [tod + "{n}! 👋 Lovely to hear from you.", "Hey{n}! 😊 Good to see you here.",
                                            "Hello{n}! 🌟 Hope you're doing okay."]))
            elif it == "how_are_you":
                parts.append(self.say("h", [
                    "I'm doing great, thanks for asking{n}! 😊 I'm a bot, so my days are pretty smooth. "
                    "How are *you* feeling today?",
                    "Feeling sharp and ready to help{n}! 🙌 And you, how are you holding up?",
                    "All good on my side, thank you{n}! How about you, how's your health today?"]))
            elif it == "identity":
                parts.append("I'm **Radar**, a health-checking sidekick built as a student project 🩺. You describe your "
                             "symptoms, I estimate which common conditions they might point to, and I show you *why* "
                             "using explainable AI. I'm not a doctor and can't diagnose anything, so please treat me as "
                             "a first step, not the last.")
            elif it == "help":
                parts.append("Here's how I work 🩺\n\n1. Tell me your name and city.\n2. Describe your symptoms in your own "
                             "words. Typos are fine!\n3. I'll show the top 3 conditions they may point to, with a "
                             "plain-English reason.\n4. Not happy with the result? Tell me more details, or tap the "
                             "button under it.\n\nType **restart** any time to begin again.")
            elif it == "vague_ill":
                parts.append(self.say("v", [
                    "I'm sorry you're not feeling well{n}. 💙 Let's figure it out together.",
                    "Oh no, sorry to hear that{n}. 💙 I'll do my best to help."]))
                if stage in ("symptoms", "result"):
                    parts.append("What exactly feels wrong? A fever, headache, cough, stomach pain, tiredness... "
                                 "even small things count.")
            elif it == "praise":
                parts.append(self.say("p", ["Aw, thank you{n}! 😊 That's kind of you to say.",
                                            "Thanks{n}, that made my day! 💙"]))
            elif it == "thanks":
                closing = True
                parts.append(self.say("t", ["You're very welcome{n}! 😊", "Anytime{n}! Glad I could help.",
                                            "Happy to help{n}! Take care of yourself. 💙"]))
                if stage == "result":
                    parts.append("If the symptoms last more than a couple of days or get worse, please see a doctor.")
            elif it == "ack":
                if stage == "result":
                    parts.append("Glad that made sense{n}! If something feels off or you remember another symptom, "
                                 "tell me and I'll take another look.".format(n=self.n()))
                    closing = True
                else:
                    parts.append(self.say("a", ["Great! 👍", "Alright!", "Okay, sounds good."]))
            elif it == "bye":
                closing = True
                parts.append(self.say("b", [
                    "Take care{n}! 👋 Rest well, drink water, and see a doctor if things get worse.",
                    "Goodbye{n}! Feel better soon. 💙",
                    "See you! Come back any time you want to check something."]))
        skip_nudge = closing or ("how_are_you" in intents and stage == "symptoms") or \
                     ("vague_ill" in intents and stage in ("symptoms", "result")) 
        if not skip_nudge:
            parts.append(self.nudge())
        self.bot(" ".join(parts))

    def more_details(self, prefix=False):
        opener = self.say("mo", ["Of course, let's dig deeper. 🔍", "Totally fair, let's look closer. 🔍",
                                 "Thanks for telling me. Let's refine this together. 🔍"]) if prefix else "Sure! 🔍"
        sofar = ""
        if self.symptoms:
            sofar = " So far I have: " + join_and(["**" + self.label(f) + "**" for f in self.symptoms]) + "."
        self.bot(opener + sofar + "\n\nTell me anything else in your own words: other symptoms (even small ones), how "
                 "long it's been going on, what makes it better or worse. Write as much as you like, and don't worry "
                 "about spelling. If something I listed isn't true for you, say so (for example *I don't have fever*) "
                 "and I'll remove it. I'll update my analysis as soon as I pick up something new.")

    def satisfied(self):
        self.bot(self.say("sat", [
            "Glad it helped{n}! 😊 If the symptoms last more than a couple of days or get worse, please see a doctor. "
            "Type **restart** for a new check.",
            "Happy that was useful{n}! 💙 Remember I'm an educational tool, so please see a doctor if you're worried."]))

    def finish(self, new=(), removed=(), lead=None):
        res = self.predict_fn(list(self.symptoms))
        top = res[0]
        pct = round(top["prob"] * 100)
        if self.rounds == 0:
            intro = lead or self.say("intro", [
                "Okay{n}, I've crunched the numbers. Here's what I'm seeing 👇",
                "Thanks for your patience{n}! Here's my read on it 👇",
                "Done! Here's what your symptoms point to{n} 👇"])
            if lead:
                intro += " Here's what I'm seeing 👇"
            if len(self.symptoms) == 1:
                intro += " (With just one symptom my read is rough, so take it lightly.)"
        else:
            bits = [lead] if lead else []
            if new and not lead:
                bits.append("Thanks, adding " + join_and(["**" + self.label(f) + "**" for f in new]) + " helps.")
            if removed and not lead:
                bits.append("I've taken out " + join_and(["**" + self.label(f) + "**" for f in removed]) + ".")
            if self.prev_top and self.prev_top != top["disease"]:
                bits.append("My top pick has moved from **" + self.prev_top + "** to **" + top["disease"] + "** (" +
                            str(pct) + "%).")
            else:
                bits.append("My top pick is still **" + top["disease"] + "** (" + str(pct) + "%).")
            intro = " ".join(bits) + " Updated result 👇"
        self.bot(intro)
        self.msgs.append(("result", res))
        if self.rounds == 0 and self.report_fn:
            try:
                self.report_fn(self.city, top["disease"])
            except Exception:
                pass
        self.prev_top = top["disease"]
        self.result_summary = ", ".join(d["disease"] + " " + str(round(d["prob"] * 100)) + "%" for d in res)
        self.rounds += 1
        follow = "Does this feel about right? If not, or if you remember anything else, just tell me more or tap " \
                 "**Not satisfied**. The more detail you give, the better I can narrow it down."
        if self.duration >= 3 or self.severe:
            when = " for " + str(self.duration) + " days" if self.duration >= 3 else ""
            follow += ("\n\n⏱️ Since this has been going on" + when + " or sounds severe, it's worth seeing a doctor "
                       "rather than waiting it out." if self.duration >= 3 else
                       "\n\n⚠️ Since it sounds severe, please don't wait. See a doctor soon.")
        self.bot(follow)