"""
clinical_report.py  -  turns a prediction into a detailed, medical-report-style explanation.

Every sentence comes from the knowledge base in this file (written from standard medical
knowledge), combined with the model's own SHAP values. Nothing is invented at run time, so the
explanation cannot "hallucinate". It uses clinical terms with a plain-English translation next to each.

EDUCATIONAL USE ONLY. It explains why the model ranked a condition highly; it is not a diagnosis.
Have a doctor or textbook review the wording before presenting it as medical information.
"""
import re
import difflib


def nk(x):
    """normalise a name: 'pain_behind_the_eyes' -> 'pain behind the eyes'."""
    x = re.sub(r"\(.*?\)", " ", str(x).lower())          # drop "(typhos)", "(piles)" ...
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", x)).strip()


# ----------------------------------------------------------------------------
# 1. Symptom dictionary:  key -> (clinical term, plain English)
# ----------------------------------------------------------------------------
SYMPTOMS = {
    "itching": ("pruritus", "an urge to scratch caused by irritated or inflamed skin"),
    "skin rash": ("cutaneous eruption", "a visible change in the colour or texture of the skin"),
    "nodal skin eruptions": ("nodular skin lesions", "small raised lumps on the skin"),
    "continuous sneezing": ("recurrent sternutation", "repeated sneezing as the nose tries to clear an irritant"),
    "shivering": ("rigors / tremor", "uncontrollable shaking, often as body temperature rises"),
    "chills": ("chills (rigors)", "a feeling of cold with shaking, usually when a fever is starting"),
    "joint pain": ("arthralgia", "pain inside or around the joints"),
    "stomach pain": ("epigastric pain", "pain in the upper middle part of the abdomen"),
    "acidity": ("gastric hyperacidity", "a burning feeling from excess stomach acid"),
    "ulcers on tongue": ("glossal ulceration", "painful sores on the tongue"),
    "muscle wasting": ("muscle atrophy / cachexia", "loss of muscle bulk and strength"),
    "vomiting": ("emesis", "forceful emptying of the stomach"),
    "burning micturition": ("dysuria", "a burning or stinging feeling when passing urine"),
    "spotting urination": ("possible haematuria", "spots of blood noticed while passing urine"),
    "fatigue": ("fatigue / asthenia", "persistent tiredness that rest does not fix"),
    "weight gain": ("weight gain", "an unexplained rise in body weight"),
    "anxiety": ("anxiety", "worry or unease, sometimes with a racing heart"),
    "cold hands and feets": ("cold extremities", "hands and feet that stay cold, often from slow circulation or metabolism"),
    "mood swings": ("affective lability", "rapid changes in mood"),
    "weight loss": ("unintentional weight loss", "losing weight without trying"),
    "restlessness": ("psychomotor agitation", "being unable to sit still or relax"),
    "lethargy": ("lethargy", "unusual sleepiness and low energy"),
    "patches in throat": ("pharyngeal plaques", "white or red patches in the throat or mouth"),
    "irregular sugar level": ("dysglycaemia", "blood sugar that swings outside the normal range"),
    "cough": ("cough", "a reflex that clears the airways of mucus or irritants"),
    "high fever": ("pyrexia (high-grade fever)", "a body temperature well above normal, usually the immune system fighting infection"),
    "sunken eyes": ("enophthalmos", "eyes that look hollow, a sign of fluid loss"),
    "breathlessness": ("dyspnoea", "difficulty or discomfort in breathing"),
    "sweating": ("diaphoresis", "heavy sweating that is not explained by heat or exercise"),
    "dehydration": ("dehydration", "the body has lost more fluid than it has taken in"),
    "indigestion": ("dyspepsia", "discomfort or fullness in the upper abdomen after eating"),
    "headache": ("cephalalgia", "pain felt in the head"),
    "yellowish skin": ("icterus (jaundice of the skin)", "yellow colouring of the skin from excess bilirubin"),
    "dark urine": ("choluria / concentrated urine", "urine that is dark brown or tea-coloured"),
    "nausea": ("nausea", "the feeling of wanting to vomit"),
    "loss of appetite": ("anorexia (loss of appetite)", "not feeling like eating"),
    "pain behind the eyes": ("retro-orbital pain", "an aching pain deep behind the eyeballs"),
    "back pain": ("dorsalgia", "pain in the back"),
    "constipation": ("constipation", "hard, infrequent or difficult stools"),
    "abdominal pain": ("abdominalgia", "pain anywhere in the belly"),
    "diarrhoea": ("diarrhoea", "frequent loose or watery stools"),
    "mild fever": ("low-grade pyrexia", "a slightly raised temperature"),
    "yellow urine": ("deep-yellow urine", "urine darker than usual, from bilirubin or low fluid intake"),
    "yellowing of eyes": ("scleral icterus", "yellow colour of the whites of the eyes, a classic sign of jaundice"),
    "acute liver failure": ("acute hepatic failure", "sudden, severe loss of liver function"),
    "fluid overload": ("fluid retention (oedema)", "extra fluid held in the body"),
    "swelling of stomach": ("ascites / abdominal swelling", "a swollen belly, often from fluid"),
    "swelled lymph nodes": ("lymphadenopathy", "swollen glands in the neck, armpit or groin"),
    "malaise": ("malaise", "a general feeling of being unwell"),
    "blurred and distorted vision": ("blurred vision", "difficulty seeing things sharply"),
    "phlegm": ("sputum production", "mucus brought up from the chest"),
    "throat irritation": ("pharyngeal irritation", "a scratchy or sore throat"),
    "redness of eyes": ("conjunctival injection", "red, irritated eyes"),
    "sinus pressure": ("sinus pressure", "a feeling of fullness or pressure in the face"),
    "runny nose": ("rhinorrhoea", "a nose that keeps running"),
    "congestion": ("nasal congestion", "a blocked or stuffy nose"),
    "chest pain": ("chest pain (thoracic pain)", "pain, pressure or tightness in the chest"),
    "weakness in limbs": ("limb weakness (paresis)", "arms or legs that feel weak"),
    "fast heart rate": ("tachycardia", "a heart rate faster than normal"),
    "pain during bowel movements": ("dyschezia", "pain while passing stools"),
    "pain in anal region": ("proctalgia", "pain around the anus"),
    "bloody stool": ("haematochezia", "fresh blood in or on the stool"),
    "irritation in anus": ("pruritus ani", "itching or irritation around the anus"),
    "neck pain": ("cervicalgia", "pain in the neck"),
    "dizziness": ("dizziness (presyncope)", "feeling light-headed or faint"),
    "cramps": ("muscle cramps / spasm", "sudden, painful tightening of a muscle"),
    "bruising": ("ecchymosis", "easy bruising of the skin"),
    "obesity": ("obesity", "a body weight well above the healthy range"),
    "swollen legs": ("peripheral oedema (legs)", "fluid swelling in the legs or ankles"),
    "swollen blood vessels": ("dilated veins", "veins that look enlarged or twisted"),
    "puffy face and eyes": ("periorbital and facial oedema", "a puffy look around the eyes and face"),
    "enlarged thyroid": ("goitre", "swelling at the front of the neck from an enlarged thyroid gland"),
    "brittle nails": ("onychoschizia", "nails that split or break easily"),
    "swollen extremeties": ("peripheral oedema", "swelling of the hands or feet"),
    "excessive hunger": ("polyphagia", "constant, unusually strong hunger"),
    "extra marital contacts": ("sexual-exposure risk factor", "a risk factor for some infections, not a symptom"),
    "drying and tingling lips": ("perioral dryness / paraesthesia", "dry lips with a tingling feeling"),
    "slurred speech": ("dysarthria", "speech that is unclear or slow"),
    "knee pain": ("gonalgia", "pain in the knee"),
    "hip joint pain": ("coxalgia", "pain in the hip joint"),
    "muscle weakness": ("myasthenia / muscle weakness", "muscles that feel weak and tire quickly"),
    "stiff neck": ("nuchal rigidity", "difficulty bending the neck forward"),
    "swelling joints": ("synovitis / joint swelling", "swollen, puffy joints"),
    "movement stiffness": ("reduced range of motion (stiffness)", "joints or muscles that feel hard to move"),
    "spinning movements": ("vertigo", "a false feeling that you or the room is spinning"),
    "loss of balance": ("ataxia / imbalance", "difficulty staying steady on your feet"),
    "unsteadiness": ("gait instability", "walking that feels unsteady"),
    "weakness of one body side": ("hemiparesis", "weakness on one side of the body"),
    "loss of smell": ("anosmia", "not being able to smell"),
    "bladder discomfort": ("suprapubic discomfort", "an uncomfortable feeling low in the abdomen near the bladder"),
    "foul smell of urine": ("malodorous urine", "urine with a strong or unpleasant smell"),
    "continuous feel of urine": ("urinary urgency", "a constant urge to pass urine"),
    "passage of gases": ("flatulence", "passing gas"),
    "internal itching": ("visceral pruritus", "an itch felt deep inside rather than on the skin"),
    "toxic look": ("toxic appearance", "looking severely unwell, flushed or apathetic"),
    "depression": ("depressive symptoms", "persistent low mood and loss of interest"),
    "irritability": ("irritability", "getting annoyed or upset very easily"),
    "muscle pain": ("myalgia", "aching muscles all over the body"),
    "altered sensorium": ("altered mental status", "confusion or reduced alertness"),
    "red spots over body": ("petechiae / maculopapular spots", "tiny red or purple spots on the skin"),
    "belly pain": ("abdominal pain", "pain in the belly"),
    "abnormal menstruation": ("menstrual irregularity", "periods that are irregular, heavy or missed"),
    "dischromic patches": ("dyschromia", "patchy changes in skin colour"),
    "watering from eyes": ("epiphora", "eyes that water excessively"),
    "increased appetite": ("hyperphagia", "eating much more than usual"),
    "polyuria": ("polyuria", "passing large amounts of urine"),
    "family history": ("positive family history", "the condition runs in the family"),
    "mucoid sputum": ("mucoid sputum", "clear or white sticky mucus from the chest"),
    "rusty sputum": ("rust-coloured sputum", "brownish-red phlegm, typical of some lung infections"),
    "lack of concentration": ("impaired concentration", "difficulty focusing"),
    "visual disturbances": ("visual disturbances", "flashes, zigzags or other changes in vision"),
    "receiving blood transfusion": ("transfusion exposure", "a past blood transfusion, a risk factor for blood-borne infections"),
    "receiving unsterile injections": ("unsterile injection exposure", "injections with unclean needles, a risk factor for blood-borne infections"),
    "coma": ("coma", "deep unconsciousness"),
    "stomach bleeding": ("upper GI haemorrhage", "bleeding from the stomach or gullet"),
    "distention of abdomen": ("abdominal distension", "a bloated, enlarged belly"),
    "history of alcohol consumption": ("alcohol-use history", "regular or heavy drinking, a risk factor for liver disease"),
    "blood in sputum": ("haemoptysis", "coughing up blood"),
    "prominent veins on calf": ("varicosities of the calf", "visible bulging veins on the lower leg"),
    "palpitations": ("palpitations", "an awareness of a pounding, fluttering or racing heartbeat"),
    "painful walking": ("antalgic gait", "pain that makes walking difficult"),
    "pus filled pimples": ("pustules", "pimples filled with pus"),
    "blackheads": ("open comedones", "small dark plugged pores"),
    "scurring": ("skin scarring", "marks left on the skin after healing"),
    "skin peeling": ("desquamation", "skin that flakes or peels off"),
    "silver like dusting": ("silvery scale", "silvery-white flaky skin patches"),
    "small dents in nails": ("nail pitting", "tiny pits or dents on the nails"),
    "inflammatory nails": ("nail inflammation", "red, swollen or painful nail folds"),
    "blister": ("vesicle / bulla", "a fluid-filled bump on the skin"),
    "red sore around nose": ("perinasal erythematous lesion", "a red, sore patch around the nose"),
    "yellow crust ooze": ("honey-coloured crusting", "yellowish crusts over weeping skin"),
}


def sym_info(key):
    k = nk(key)
    k = re.sub(r" \d+$", "", k)
    if k in SYMPTOMS:
        return SYMPTOMS[k]
    close = difflib.get_close_matches(k, SYMPTOMS, n=1, cutoff=0.85)
    if close:
        return SYMPTOMS[close[0]]
    return (k, "a reported symptom")


def sym_label(key):
    k = re.sub(r" \d+$", "", nk(key))
    return k


# ----------------------------------------------------------------------------
# 2. Disease knowledge base  (41 conditions of the public symptom dataset)
# ----------------------------------------------------------------------------
KB = {
    "fungal infection": dict(
        name="Fungal skin infection (dermatophytosis / candidiasis)",
        what="A superficial infection of the skin, nails or hair by fungi that feed on keratin, the protein in the outer skin.",
        mech="Fungal spores settle on warm, moist skin and release enzymes that break down keratin. The skin reacts with inflammation, which produces itching, scaling and ring-shaped or patchy rashes.",
        typical=["itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"],
        tests="a skin scraping examined under the microscope (KOH preparation) or fungal culture",
        red="rapid spread, high fever, pus, or an infection in someone with diabetes or a weak immune system",
        care="Usually treated with a doctor-prescribed antifungal cream or tablets; keep skin dry and do not share towels."),
    "allergy": dict(
        name="Allergic reaction (IgE-mediated hypersensitivity, e.g. allergic rhinitis)",
        what="An over-reaction of the immune system to a harmless trigger such as dust, pollen, food or pollution.",
        mech="Allergen-specific IgE antibodies sit on mast cells. On re-exposure the cells release histamine, which causes sneezing, watery eyes, itching and sometimes skin rashes.",
        typical=["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "itching", "skin_rash"],
        tests="allergy skin-prick tests or specific IgE blood tests",
        red="swelling of the lips, tongue or throat, wheeze or trouble breathing (possible anaphylaxis)",
        care="Avoid the trigger; antihistamines are commonly used on a doctor's or pharmacist's advice."),
    "gerd": dict(
        name="Gastro-oesophageal reflux disease (GERD)",
        what="Stomach acid repeatedly flows back up into the food pipe (oesophagus).",
        mech="The lower oesophageal sphincter, a muscular valve, relaxes at the wrong times. Acid irritates the lining of the oesophagus, causing burning, regurgitation and sometimes cough or chest discomfort.",
        typical=["acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain", "indigestion", "stomach_pain"],
        tests="a clinical assessment, sometimes upper GI endoscopy or pH monitoring",
        red="difficulty or pain on swallowing, weight loss, black stools, vomiting blood, or chest pain with exertion (which needs a heart check)",
        care="Smaller meals, avoiding late-night eating and trigger foods; medicines that reduce acid are used under medical advice."),
    "chronic cholestasis": dict(
        name="Chronic cholestasis (reduced bile flow)",
        what="Bile made by the liver cannot flow properly into the gut, so bile components build up in the blood.",
        mech="Blocked or impaired bile flow raises bilirubin and bile salts. Bilirubin turns skin and eyes yellow; bile salts deposited in the skin cause intense itching; poor fat digestion causes nausea and loss of appetite.",
        typical=["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
        tests="liver function tests (bilirubin, alkaline phosphatase) and an abdominal ultrasound",
        red="fever with jaundice, confusion, or severe abdominal pain",
        care="Needs a doctor to find and treat the cause of the bile-flow problem."),
    "drug reaction": dict(
        name="Adverse drug reaction (drug hypersensitivity)",
        what="An unwanted reaction of the body to a medicine.",
        mech="The immune system may recognise a drug (or its breakdown product) as foreign and respond with inflammation of the skin and other organs, causing itching, rash and sometimes stomach or urinary symptoms.",
        typical=["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_ urination"],
        tests="a review of recent medicines and their timing, sometimes blood tests",
        red="facial swelling, blisters in the mouth or eyes, peeling skin, fever or breathing difficulty",
        care="Contact a doctor promptly and tell them every medicine you have taken recently; do not restart the suspected drug."),
    "peptic ulcer": dict(
        name="Peptic ulcer disease",
        what="Open sores in the lining of the stomach or the first part of the small intestine.",
        mech="The protective mucus layer is damaged, most often by Helicobacter pylori infection or long-term use of anti-inflammatory painkillers (NSAIDs). Stomach acid then burns the exposed tissue, causing pain, nausea and a feeling of fullness.",
        typical=["vomiting", "abdominal_pain", "indigestion", "loss_of_appetite", "passage_of_gases", "internal_itching"],
        tests="a breath, stool or blood test for H. pylori, or an upper GI endoscopy",
        red="black tarry stools, vomiting blood, or sudden severe abdominal pain",
        care="Needs medical treatment to reduce acid and clear H. pylori if present; avoid unprescribed painkillers."),
    "aids": dict(
        name="HIV infection / AIDS",
        what="Infection with the human immunodeficiency virus that, if untreated, weakens the immune system.",
        mech="HIV infects and destroys CD4 T-lymphocytes. As their number falls, the body struggles against ordinary infections, causing recurrent fevers, weight and muscle loss, and fungal patches in the mouth or throat.",
        typical=["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts", "weight_loss", "swelled_lymph_nodes"],
        tests="HIV antibody/antigen testing with confirmatory testing, and a CD4 count",
        red="persistent fever, severe weight loss, breathlessness or confusion",
        care="Confirmatory testing is essential. Effective antiretroviral treatment exists and is available through government ART centres in India."),
    "diabetes": dict(
        name="Diabetes mellitus",
        what="A long-term condition in which the body cannot keep blood sugar (glucose) in the normal range.",
        mech="Insulin is either too little or the body responds to it poorly, so glucose builds up in the blood. The kidneys flush out extra sugar with water (polyuria), causing thirst and dehydration, while cells that cannot use glucose cause tiredness and hunger.",
        typical=["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision",
                 "obesity", "excessive_hunger", "increased_appetite", "polyuria"],
        tests="fasting and post-meal blood glucose and HbA1c",
        red="vomiting, deep fast breathing, fruity-smelling breath or drowsiness (possible diabetic ketoacidosis)",
        care="Diagnosis is by blood tests; diet, activity and medicines are managed by a doctor."),
    "gastroenteritis": dict(
        name="Acute gastroenteritis",
        what="Inflammation of the stomach and intestines, usually from a viral, bacterial or food-borne infection.",
        mech="The germs or their toxins irritate the gut lining, which pours out fluid and speeds up movement. This causes vomiting and watery diarrhoea and leads to loss of water and salts (dehydration).",
        typical=["vomiting", "sunken_eyes", "dehydration", "diarrhoea", "nausea", "abdominal_pain"],
        tests="usually a clinical assessment; stool tests if severe or prolonged",
        red="blood in the stool, signs of dehydration (dry mouth, sunken eyes, very little urine, dizziness), or illness in infants and older adults",
        care="Oral rehydration solution and rest; seek care early if you cannot keep fluids down."),
    "bronchial asthma": dict(
        name="Bronchial asthma",
        what="A long-term condition in which the airways become inflamed and narrow, causing wheeze and breathlessness.",
        mech="Inflamed airways overreact to triggers such as dust, smoke, cold air or infections. The airway muscles tighten and mucus increases (bronchospasm), which narrows the tubes and makes breathing out difficult.",
        typical=["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
        tests="lung function testing (spirometry) and response to a reliever inhaler",
        red="being unable to speak full sentences, blue lips, or no relief from the reliever inhaler",
        care="Controlled with inhalers prescribed by a doctor; avoid known triggers."),
    "hypertension": dict(
        name="Hypertension (high blood pressure)",
        what="Blood pressure that stays higher than normal over time.",
        mech="The force of blood against the artery walls stays raised. It is often silent, but when pressure is high it can cause headaches, dizziness and, over years, damage to the heart, brain and kidneys.",
        typical=["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
        tests="repeated blood pressure measurements, sometimes with blood and urine tests",
        red="a very high reading with chest pain, confusion, vision changes or weakness on one side",
        care="Confirm with proper blood pressure readings; lifestyle changes and medicines are guided by a doctor."),
    "migraine": dict(
        name="Migraine",
        what="A recurring headache disorder, often with nausea and sensitivity to light.",
        mech="Waves of abnormal activity in the brain change blood flow and trigger the release of pain-signalling chemicals around the blood vessels of the head. This can also cause visual disturbances (aura), nausea and digestive upset.",
        typical=["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck",
                 "depression", "irritability", "visual_disturbances"],
        tests="mainly a clinical history; scans only if there are warning signs",
        red="a sudden 'worst-ever' headache, weakness, trouble speaking, or headache with fever and a stiff neck",
        care="Rest in a dark, quiet room and keep a trigger diary; a doctor can prescribe specific treatment."),
    "cervical spondylosis": dict(
        name="Cervical spondylosis",
        what="Age-related wear of the discs and joints in the neck part of the spine.",
        mech="Discs lose water and height, and bony spurs form. These can narrow the spaces where nerves leave the spine, causing neck pain, stiffness, and sometimes weakness or tingling in the arms.",
        typical=["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
        tests="a neurological examination and an X-ray or MRI of the neck",
        red="progressive weakness in the limbs, loss of bladder or bowel control, or severe numbness",
        care="Physiotherapy, posture correction and pain relief under medical guidance."),
    "paralysis": dict(
        name="Paralysis due to brain haemorrhage (haemorrhagic stroke)",
        what="Bleeding inside the brain that damages the areas controlling movement and speech.",
        mech="A weakened blood vessel in the brain bursts. Blood presses on brain tissue and cuts off normal signals, causing sudden headache, vomiting, weakness on one side and confusion.",
        typical=["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium", "slurred_speech"],
        tests="an urgent CT scan of the brain",
        red="THIS IS AN EMERGENCY: sudden weakness, facial droop, slurred speech or severe headache need immediate hospital care",
        care="Call 112 or go to the nearest hospital at once; every minute matters."),
    "jaundice": dict(
        name="Jaundice (hyperbilirubinaemia)",
        what="Yellow colouring of the skin and the whites of the eyes caused by too much bilirubin in the blood.",
        mech="Bilirubin is a yellow pigment made when old red blood cells break down. If the liver cannot process it, or its drainage is blocked, it builds up in the blood and tissues, giving the yellow colour, dark urine and itching.",
        typical=["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain", "yellowing_of_eyes"],
        tests="liver function tests, an abdominal ultrasound and hepatitis serology",
        red="confusion, bleeding, persistent vomiting, or fever with abdominal pain",
        care="The cause (such as hepatitis or a blocked bile duct) must be found by a doctor; rest and plenty of fluids meanwhile."),
    "malaria": dict(
        name="Malaria (Plasmodium infection)",
        what="A mosquito-borne infection of red blood cells by Plasmodium parasites, spread by the Anopheles mosquito.",
        mech="The parasites multiply inside red blood cells and burst them in cycles. Each burst releases substances that trigger the immune system, causing waves of chills, high fever and heavy sweating, with headache, nausea and body aches.",
        typical=["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea", "muscle_pain"],
        tests="a rapid diagnostic test and a peripheral blood smear",
        red="confusion, seizures, jaundice, severe weakness or breathing difficulty (possible severe malaria)",
        care="Needs a blood test the same day and proper antimalarial treatment; use mosquito nets and repellents."),
    "chicken pox": dict(
        name="Chickenpox (varicella)",
        what="A highly contagious viral illness caused by the varicella-zoster virus.",
        mech="The virus spreads through the air, multiplies in the body and then reaches the skin, where it forms itchy fluid-filled blisters in successive crops, along with fever and tiredness.",
        typical=["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever",
                 "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
        tests="a clinical examination of the typical rash; PCR if the diagnosis is uncertain",
        red="pregnancy, a newborn, a weak immune system, a rash near the eye, breathing difficulty or confusion",
        care="Rest, fluids and fever control as advised by a doctor; avoid scratching and stay away from others until all blisters have crusted."),
    "dengue": dict(
        name="Dengue fever (dengue virus infection)",
        what="A mosquito-borne viral illness spread by the Aedes aegypti mosquito.",
        mech="The dengue virus multiplies in the blood and triggers a strong immune response with release of inflammatory chemicals (cytokines). This causes abrupt high fever, pain behind the eyes, severe body and joint aches (hence 'breakbone fever'), rash, and a fall in platelets that can lead to bleeding.",
        typical=["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite",
                 "pain_behind_the_eyes", "back_pain", "malaise", "muscle_pain", "red_spots_over_body"],
        tests="NS1 antigen or IgM antibody tests and a platelet count",
        red="warning signs, usually when the fever settles: severe abdominal pain, persistent vomiting, bleeding gums or nose, black stools, drowsiness or restlessness, fast breathing",
        care="Rest and plenty of fluids; ask a doctor which fever medicine is safe, as aspirin and ibuprofen are generally avoided in suspected dengue. Get a blood count checked."),
    "typhoid": dict(
        name="Typhoid (enteric) fever",
        what="A bloodstream and gut infection caused by Salmonella Typhi, spread through contaminated food and water.",
        mech="The bacteria invade the intestinal wall and spread through the blood. This causes a fever that climbs step by step, headache, abdominal pain and either constipation or diarrhoea, and a generally 'toxic', worn-out appearance.",
        typical=["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea",
                 "toxic_look_(typhos)", "belly_pain"],
        tests="a blood culture (preferred); Widal is less reliable",
        red="severe abdominal pain, black stools, confusion, or a fever lasting more than a week",
        care="Needs antibiotics prescribed by a doctor; drink only safe water and wash hands."),
    "hepatitis a": dict(
        name="Hepatitis A",
        what="A viral liver infection spread through contaminated food and water; it usually clears by itself.",
        mech="The virus infects liver cells and the immune response inflames the liver. Bile pigment then spills into the blood, causing yellow eyes and skin, dark urine, nausea and loss of appetite.",
        typical=["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea",
                 "mild_fever", "yellowing_of_eyes", "muscle_pain"],
        tests="liver function tests and anti-HAV IgM antibodies",
        red="confusion, bleeding, persistent vomiting or severe drowsiness",
        care="Rest, fluids and a doctor's follow-up; a vaccine and safe water prevent it."),
    "hepatitis b": dict(
        name="Hepatitis B",
        what="A viral liver infection spread through blood, unprotected sex or from mother to baby.",
        mech="The hepatitis B virus infects liver cells; the immune attack causes inflammation. In some people the infection becomes long-term and can lead to scarring (cirrhosis) over years.",
        typical=["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine",
                 "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
        tests="HBsAg and other hepatitis B markers, plus liver function tests",
        red="confusion, vomiting blood, a swollen belly, or deepening jaundice",
        care="Needs a doctor's assessment; vaccination is effective prevention."),
    "hepatitis c": dict(
        name="Hepatitis C",
        what="A blood-borne viral infection of the liver that often becomes long-term.",
        mech="The virus persists in liver cells and causes slow, ongoing inflammation that is often silent for years, with tiredness and, later, jaundice and liver scarring.",
        typical=["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "family_history",
                 "receiving_blood_transfusion", "receiving_unsterile_injections"],
        tests="anti-HCV antibodies followed by an HCV RNA test",
        red="a swollen belly, vomiting blood, confusion or deepening jaundice",
        care="Modern antiviral tablets can cure most people; see a doctor for testing."),
    "hepatitis d": dict(
        name="Hepatitis D (delta hepatitis)",
        what="A liver infection that only occurs in people who also have hepatitis B.",
        mech="The delta virus needs the hepatitis B virus to multiply. Together they cause more severe liver inflammation than hepatitis B alone.",
        typical=["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain",
                 "yellowing_of_eyes"],
        tests="anti-HDV antibodies and HDV RNA, along with hepatitis B markers",
        red="confusion, bleeding, a swollen belly or deepening jaundice",
        care="Needs specialist care; hepatitis B vaccination also prevents hepatitis D."),
    "hepatitis e": dict(
        name="Hepatitis E",
        what="A viral liver infection spread through contaminated water; usually short-lived but dangerous in pregnancy.",
        mech="The virus infects liver cells and the immune response causes inflammation, giving jaundice, nausea and fever. In pregnant women it can cause acute liver failure.",
        typical=["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
                 "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure", "coma", "stomach_bleeding"],
        tests="anti-HEV IgM antibodies and liver function tests",
        red="pregnancy, confusion, drowsiness, or bleeding",
        care="Rest, fluids and medical review; drink safe water."),
    "alcoholic hepatitis": dict(
        name="Alcoholic hepatitis",
        what="Inflammation of the liver caused by heavy, long-term alcohol use.",
        mech="Alcohol breakdown products injure liver cells and trigger inflammation and fat build-up. The liver then struggles to process bilirubin and fluids, causing jaundice and a swollen belly (ascites).",
        typical=["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen",
                 "history_of_alcohol_consumption", "fluid_overload"],
        tests="liver function tests, an abdominal ultrasound, and an alcohol-use history",
        red="confusion, vomiting blood, severe jaundice or a rapidly swelling belly",
        care="Stopping alcohol is essential; needs medical supervision."),
    "tuberculosis": dict(
        name="Tuberculosis (TB)",
        what="A bacterial infection, usually of the lungs, caused by Mycobacterium tuberculosis.",
        mech="The bacteria are inhaled and survive inside immune cells in the lung. The slow, ongoing inflammation causes a long-lasting cough with phlegm, evening fever, night sweats, weight loss and sometimes blood in the sputum.",
        typical=["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating",
                 "loss_of_appetite", "mild_fever", "yellowing_of_eyes", "swelled_lymph_nodes", "malaise", "phlegm", "chest_pain",
                 "blood_in_sputum"],
        tests="sputum tests (smear or CBNAAT) and a chest X-ray",
        red="coughing up blood or severe breathlessness",
        care="A cough lasting more than two weeks needs testing; TB is curable, and free diagnosis and treatment are provided under India's national TB programme."),
    "common cold": dict(
        name="Common cold (viral upper respiratory tract infection)",
        what="A mild viral infection of the nose and throat, most often caused by rhinoviruses.",
        mech="The virus infects the lining of the nose and throat. The immune response makes the lining swell and produce mucus, causing sneezing, a runny or blocked nose, sore throat and cough, usually settling in 7 to 10 days.",
        typical=["continuous_sneezing", "chills", "fatigue", "cough", "headache", "swelled_lymph_nodes", "malaise", "phlegm",
                 "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "loss_of_smell", "muscle_pain"],
        tests="usually none; diagnosed from symptoms",
        red="breathlessness, chest pain, fever lasting more than three days, or ear or facial pain",
        care="Rest, fluids and steam; antibiotics do not work against viral colds."),
    "pneumonia": dict(
        name="Pneumonia (lower respiratory tract infection)",
        what="Infection that inflames the air sacs (alveoli) of one or both lungs.",
        mech="Germs reach the air sacs, which fill with pus and fluid. Oxygen cannot pass into the blood properly, causing cough with phlegm, fever with chills, chest pain on breathing and breathlessness.",
        typical=["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain",
                 "fast_heart_rate", "rusty_sputum"],
        tests="a chest X-ray, oxygen saturation check and blood tests",
        red="breathlessness at rest, confusion, blue lips, or illness in young children and older adults",
        care="Needs medical assessment; antibiotics or hospital care depending on severity."),
    "dimorphic hemmorhoids": dict(
        name="Haemorrhoids (piles)",
        what="Swollen blood-vessel cushions in the lower rectum and anus, which can be internal, external or both.",
        mech="Straining and constipation raise pressure in the veins of the anal canal, so they swell and bulge. This causes pain on passing stools, itching and bright-red bleeding.",
        typical=["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        tests="a clinical examination, sometimes proctoscopy",
        red="heavy bleeding, a change in bowel habit, weight loss, or bleeding after the age of 40 (to rule out other causes)",
        care="More fibre and water, avoiding straining; creams and procedures are decided by a doctor."),
    "heart attack": dict(
        name="Heart attack (myocardial infarction)",
        what="Sudden blockage of a heart artery that starves part of the heart muscle of oxygen.",
        mech="A clot forms on a fatty plaque inside a coronary artery. The heart muscle beyond the blockage becomes ischaemic (oxygen-starved), causing crushing chest pain, sweating, breathlessness and vomiting.",
        typical=["vomiting", "breathlessness", "sweating", "chest_pain", "fast_heart_rate", "dizziness"],
        tests="an ECG and a blood test for troponin",
        red="THIS IS AN EMERGENCY: chest pain or pressure with sweating, breathlessness or pain spreading to the arm or jaw",
        care="Call 112 immediately and do not drive yourself; early treatment saves heart muscle."),
    "varicose veins": dict(
        name="Varicose veins (chronic venous insufficiency)",
        what="Enlarged, twisted veins, most often in the legs.",
        mech="Tiny valves in the leg veins fail, so blood pools instead of flowing back to the heart. The veins stretch and bulge, causing heaviness, cramps, swelling and visible blue cords.",
        typical=["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
        tests="a clinical examination and a Doppler ultrasound of the leg veins",
        red="sudden pain and swelling in one leg (possible clot), skin ulcers, or bleeding from a vein",
        care="Leg elevation, walking and compression stockings; procedures are decided by a vascular doctor."),
    "hypothyroidism": dict(
        name="Hypothyroidism (underactive thyroid)",
        what="The thyroid gland makes too little thyroid hormone, so the body's metabolism slows down.",
        mech="Low thyroid hormone slows the use of energy by almost every organ. This causes tiredness, weight gain, feeling cold, dry skin and brittle nails, low mood, puffy face and sometimes a visible neck swelling (goitre).",
        typical=["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes",
                 "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
        tests="TSH and free T4 blood tests",
        red="extreme drowsiness, confusion or a very low body temperature",
        care="Confirmed by a blood test; treated with daily thyroid hormone tablets prescribed by a doctor."),
    "hyperthyroidism": dict(
        name="Hyperthyroidism (overactive thyroid)",
        what="The thyroid gland makes too much thyroid hormone, speeding up the body's metabolism.",
        mech="Excess thyroid hormone drives the heart and metabolism to work faster. This causes weight loss despite a big appetite, fast heartbeat, sweating, tremor, anxiety, loose stools and muscle weakness.",
        typical=["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger",
                 "muscle_weakness", "irritability", "abnormal_menstruation", "palpitations"],
        tests="TSH and free T4 / T3 blood tests",
        red="a very fast or irregular heartbeat, high fever or confusion",
        care="Confirmed by a blood test; treated by a doctor with medicines or other options."),
    "hypoglycemia": dict(
        name="Hypoglycaemia (low blood sugar)",
        what="Blood glucose falls too low to fuel the brain and body.",
        mech="When glucose drops, the body releases adrenaline, causing sweating, shakiness, palpitations, hunger and anxiety. If it keeps falling, the brain is starved of fuel, causing confusion and slurred speech.",
        typical=["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger",
                 "drying_and_tingling_lips", "slurred_speech", "irritability", "palpitations"],
        tests="a fingertip or laboratory blood glucose reading at the time of symptoms",
        red="confusion, seizure or unconsciousness (emergency)",
        care="If awake and able to swallow, take a quick sugar source such as juice or glucose and then seek advice; call 112 if confused or unconscious."),
    "osteoarthritis": dict(
        name="Osteoarthritis",
        what="Wear and tear of the cartilage that cushions the joints.",
        mech="Cartilage thins and roughens, so bones rub and the joint lining becomes inflamed. This causes joint pain that worsens with use, morning stiffness and sometimes swelling, especially in knees, hips and the spine.",
        typical=["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking", "movement_stiffness"],
        tests="a clinical examination and X-rays of the affected joint",
        red="a hot, red, very swollen joint, fever, or sudden inability to bear weight",
        care="Weight control, exercise and physiotherapy; pain relief and other options are chosen with a doctor."),
    "arthritis": dict(
        name="Arthritis (inflammatory joint disease, e.g. rheumatoid arthritis)",
        what="Inflammation of one or more joints causing pain, swelling and stiffness.",
        mech="In inflammatory arthritis the immune system attacks the joint lining (synovium). The joints swell and stiffen, especially in the morning, and muscles around them weaken.",
        typical=["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking", "joint_pain"],
        tests="blood tests (ESR, CRP, rheumatoid factor, anti-CCP) and joint X-rays",
        red="a single hot, red, swollen joint with fever (possible joint infection)",
        care="Early assessment by a doctor matters because treatment can prevent joint damage."),
    "paroymsal positional vertigo": dict(
        name="Benign paroxysmal positional vertigo (BPPV)",
        what="Short spells of spinning dizziness triggered by changes in head position.",
        mech="Tiny calcium crystals in the inner ear drift into the balance canals. Head movements then send false 'spinning' signals to the brain, causing vertigo, nausea, vomiting and unsteadiness.",
        typical=["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
        tests="a bedside positional test (Dix-Hallpike manoeuvre)",
        red="vertigo with slurred speech, double vision, weakness or a severe headache (possible stroke)",
        care="Often treated with repositioning manoeuvres (such as the Epley manoeuvre) by a clinician."),
    "acne": dict(
        name="Acne vulgaris",
        what="A common skin condition in which hair follicles become blocked and inflamed.",
        mech="Excess oil (sebum) and dead skin plug the follicle, forming blackheads. Skin bacteria then multiply and cause inflammation, producing red bumps and pus-filled pimples that can scar.",
        typical=["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        tests="a clinical examination",
        red="deep painful cysts, scarring, or acne causing significant distress",
        care="Gentle cleansing and not squeezing spots; a dermatologist can prescribe effective treatment."),
    "urinary tract infection": dict(
        name="Urinary tract infection (UTI)",
        what="A bacterial infection of the bladder or urethra, and sometimes the kidneys.",
        mech="Bacteria, usually from the gut, climb the urethra and multiply in the bladder. The lining becomes inflamed, causing burning on urination, urgency, frequent small passages and cloudy or foul-smelling urine.",
        typical=["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine", "spotting_ urination"],
        tests="a urine routine examination and urine culture",
        red="fever with chills or flank pain (kidney infection), blood in urine, pregnancy, or infection in a child or man",
        care="Drink plenty of water; antibiotics are chosen by a doctor, ideally after a urine test."),
    "psoriasis": dict(
        name="Psoriasis",
        what="A long-term immune-related skin condition with thick, scaly patches.",
        mech="The immune system speeds up skin-cell renewal from about a month to a few days. Cells pile up as silvery scales over red plaques, and the nails and joints can also be affected.",
        typical=["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
        tests="a clinical examination, sometimes a skin biopsy",
        red="widespread redness, painful or swollen joints, or infection of the plaques",
        care="A dermatologist can prescribe creams, light therapy or other treatments."),
    "impetigo": dict(
        name="Impetigo",
        what="A contagious bacterial skin infection, common in children.",
        mech="Staphylococcus or Streptococcus bacteria enter small breaks in the skin and form blisters that burst and dry into honey-coloured crusts, often around the nose and mouth.",
        typical=["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
        tests="a clinical examination; a swab if it does not settle",
        red="spreading redness, fever, or lesions in a newborn",
        care="Antibiotic cream or tablets prescribed by a doctor; wash hands and avoid sharing towels as it spreads easily."),
}

# dataset spelling (normalised)  ->  knowledge-base key
ALIASES = {
    "peptic ulcer diseae": "peptic ulcer", "peptic ulcer disease": "peptic ulcer", "osteoarthristis": "osteoarthritis",
    "dimorphic hemmorhoids piles": "dimorphic hemmorhoids", "dimorphic hemorrhoids piles": "dimorphic hemmorhoids",
    "vertigo paroymsal positional vertigo": "paroymsal positional vertigo", "paralysis brain hemorrhage": "paralysis",
    "hepatitis a": "hepatitis a", "chickenpox": "chicken pox", "diabetes": "diabetes", "gerd": "gerd",
}


def kb_for(disease):
    k = nk(disease)
    k = ALIASES.get(k, k)
    if k in KB:
        return KB[k]
    for name in KB:
        if name in k or k in name:
            return KB[name]
    m = difflib.get_close_matches(k, KB, n=1, cutoff=0.75)
    return KB[m[0]] if m else None


# ----------------------------------------------------------------------------
# 3. Report builder
# ----------------------------------------------------------------------------
def _term(key):
    clin, plain = sym_info(key)
    return clin, plain


def _fmt(key, bold=True):
    clin, plain = _term(key)
    name = sym_label(key)
    b = "**" if bold else ""
    return f"{b}{name}{b} (*{clin}*)"


def tier(p):
    return "high" if p >= .6 else "moderate" if p >= .3 else "low"


def build_report(d, reported, res=None, n_conditions=41):
    """d: {'disease','prob','drivers':[(feature, shap, present)]}   reported: list of reported symptom keys
    res: the full top-3 list (for the comparison). Returns markdown."""
    disease, prob = d["disease"], d["prob"]
    kb = kb_for(disease)
    pct = round(prob * 100)
    rank = 1
    if res:
        names = [x["disease"] for x in res]
        rank = names.index(disease) + 1 if disease in names else 1
    reported_n = {nk(re.sub(r"\.\d+$", "", f)): f for f in reported}
    drivers = list(d.get("drivers", []))
    shap_of = {nk(re.sub(r"\.\d+$", "", f)): v for f, v, p in drivers}
    typical = [nk(re.sub(r"\.\d+$", "", f)) for f in (kb["typical"] if kb else [])]

    L = []
    title = kb["name"] if kb else disease
    L.append(f"### 📋 Clinical rationale: {title}")
    L.append(f"**Model confidence:** {pct}% ({tier(prob)}) · ranked **#{rank}** of the top 3 shown, out of "
             f"{n_conditions} conditions the model knows.")

    if kb:
        L.append("#### 1. In plain English")
        L.append(kb["what"])
    else:
        L.append("#### 1. In plain English")
        L.append(f"The model found that your symptoms most resemble the pattern it learned for **{disease}**.")

    # --- why it matched
    L.append("#### 2. Why your symptoms pointed here")
    if not reported:
        L.append("No symptoms were recorded for this result.")
    else:
        rows = sorted(reported_n, key=lambda k: -abs(shap_of.get(k, 0)))
        bullets = []
        for k in rows:
            orig = reported_n[k]
            clin, plain = _term(orig)
            sv = shap_of.get(k)
            classic = k in typical
            if sv is not None and sv > 0.005:
                strength = "strongly supports" if sv > 0.05 else "supports"
            elif sv is not None and sv < -0.005:
                strength = "does not fit well with"
            else:
                strength = "has little influence on"
            fit = "a recognised feature" if classic else "not a classic feature"
            bullets.append(f"- **{sym_label(orig)}** (*{clin}*): {plain}. This **{strength}** "
                           f"{title.split(' (')[0]}, and it is {fit} of this condition.")
        L.append("\n".join(bullets))
        matched = [k for k in reported_n if k in typical]
        if kb:
            L.append(f"\n**Overall:** {len(matched)} of your {len(reported_n)} reported symptom(s) are recognised "
                     f"features of {title.split(' (')[0]}. The model weighs these as a *combined pattern*, not one "
                     f"symptom alone.")

    # --- mechanism
    if kb:
        L.append("#### 3. What happens in the body (pathophysiology)")
        L.append(kb["mech"])

    # --- missing features
    if kb:
        missing = [t for t in typical if t not in reported_n]
        L.append("#### 4. Typical features you did not report")
        if missing:
            show = missing[:6]
            items = ", ".join(f"{sym_label(t)} (*{sym_info(t)[0]}*)" for t in show)
            L.append(f"Doctors would also expect to ask about: {items}. If you have any of these, tell Radar; "
                     f"if you clearly do **not**, that makes this condition less likely.")
        else:
            L.append("You reported all the typical features the model looks for.")

    # --- differential
    if res and len(res) > 1:
        L.append("#### 5. Other possibilities the model considered")
        diff = []
        for x in res:
            if x["disease"] == disease:
                continue
            ok = kb_for(x["disease"])
            other_typ = [nk(re.sub(r"\.\d+$", "", f)) for f in (ok["typical"] if ok else [])]
            shared = [sym_label(reported_n[k]) for k in reported_n if k in other_typ]
            on = ok["name"].split(" (")[0] if ok else x["disease"]
            line = f"- **{on}**, {round(x['prob'] * 100)}%"
            if shared:
                line += f": also fits {', '.join(shared[:4])}"
            if ok:
                line += f". A doctor would tell these apart with {ok['tests']}."
            diff.append(line)
        L.append("\n".join(diff))

    # --- what to do
    L.append("#### 6. What this means and what to do next")
    if prob < .3:
        L.append("The confidence is **low**, so treat this as one of several possibilities and not as a likely answer.")
    elif prob < .6:
        L.append("The confidence is **moderate**: the pattern fits, but other conditions remain possible.")
    else:
        L.append("The confidence is **high**: your symptom pattern matches this condition closely. A match is still "
                 "not a diagnosis.")
    if kb:
        L.append("\n".join([f"- **How it is confirmed:** {kb['tests']}.",
                            f"- **Seek urgent care if:** {kb['red']}.",
                            f"- **General advice:** {kb['care']}"]))
    L.append("\n*This explanation is generated from a medical knowledge base and the model's SHAP values. It shows why "
             "the model ranked this condition and is for education only. It is not a diagnosis, and it cannot replace "
             "an examination by a doctor.*")
    return "\n\n".join(L)
