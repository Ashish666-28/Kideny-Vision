import argparse
import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.table import Table
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Search examples are editorial metadata, not patient answers or medical evidence.
TOPIC_HINTS = {'NORMAL_001': {'questions': ['The scan says normal what now',
                              'Does a normal kidney picture mean I am healthy',
                              'Can normal CT rule out every kidney problem',
                              'What does the green normal result mean'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_002': {'questions': ['How do I know my kidneys are working',
                              'Which blood test checks kidney function',
                              'Explain creatinine and eGFR simply',
                              'Do I need blood tests after a clear image'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_003': {'questions': ['When should I get checked again',
                              'What routine should I follow if normal',
                              'Do I need follow up without symptoms',
                              'How do I keep my kidneys healthy'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_004': {'questions': ['What should a healthy person eat for kidneys',
                              'Do I need a kidney diet if normal',
                              'What is a balanced kidney friendly meal',
                              'Should everyone stop protein and salt'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_005': {'questions': ['Where does urine come from',
                              'Difference between ureter and urethra',
                              'What do kidneys actually do',
                              'Explain how urine reaches the bladder'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_006': {'questions': ['Why is a urine test needed',
                              'What does UACR mean',
                              'Does protein in urine need follow up',
                              'Urine albumin test compared with a scan'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_007': {'questions': ['Can high BP damage kidneys',
                              'How are blood pressure and kidneys linked',
                              'Should I keep taking pressure tablets',
                              'Does normal CT mean my BP is safe'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_008': {'questions': ['Can high blood sugar affect kidneys',
                              'I have diabetes which kidney tests matter',
                              'Why check kidneys when diabetes feels fine',
                              'How does sugar disease affect kidney health'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'NORMAL_009': {'questions': ['Give me breakfast lunch dinner ideas',
                              'What can I cook with less salt',
                              'Are rice roti dal and vegetables okay',
                              'Suggest ordinary healthy meals'],
                'applicability': 'Illustrative food combinations for adults; adapt to kidney function, stone '
                                 'composition, diabetes, allergies and prescribed restrictions.'},
 'NORMAL_010': {'questions': ['How much water should I drink',
                              'Must everyone drink many litres daily',
                              'What if my doctor restricted fluids',
                              'Does drinking extra water clean kidneys'],
                'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_001': {'questions': ['What is a kidney stone in simple words',
                             'Does a stone always block urine',
                             'Where can a urinary stone get stuck',
                             'My result is stone what does it mean'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_002': {'questions': ['How can I know my stone type',
                             'Can CT classification tell stone composition',
                             'Are all stones made of calcium',
                             'Why do different stones need different advice'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_003': {'questions': ['Can stones cause pain and blood in pee',
                             'What stone symptoms are dangerous',
                             'Pain goes from side to groin',
                             'Stone with fever vomiting or chills'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_004': {'questions': ['How is a stone diagnosis confirmed',
                             'What should I look for in my CT report',
                             'Does the label show stone size',
                             'Which tests find stone position and blockage'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_005': {'questions': ['Can a stone come out without surgery',
                             'How do doctors decide to wait',
                             'What should I do while waiting for passage',
                             'Does stone size decide treatment alone'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_006': {'questions': ['What treatments remove kidney stones',
                             'Difference between laser and shock wave treatment',
                             'Will every stone need an operation',
                             'What is PCNL or ureteroscopy'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_007': {'questions': ['How should I drink water for stone prevention',
                             'Does more urine help prevent stones',
                             'Should I force water during severe stone pain',
                             'Can I increase fluids if I have a fluid limit'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_008': {'questions': ['Which diet should I follow for stones',
                             'What food should I take and avoid',
                             'Does stone type change food advice',
                             'Can I get general food advice before stone analysis'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_009': {'questions': ['Should I stop milk because my stone has calcium',
                             'Is curd okay with kidney stones',
                             'Difference between calcium food and calcium tablets',
                             'Why does calcium with meals matter'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_010': {'questions': ['How can I prevent another stone',
                             'What records should I keep after treatment',
                             'Will a stone come back after removal',
                             'How do I monitor prevention progress'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_011': {'questions': ['Why did I get a kidney stone',
                             'Can family history increase stone risk',
                             'Which health conditions affect stone formation',
                             'Is it only because I drank too little water'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_012': {'questions': ['Why test a passed stone',
                             'What is a 24 hour urine collection',
                             'Can stone colour tell its composition',
                             'Which tests help prevent recurrent stones'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_013': {'questions': ['Which foods matter for calcium oxalate stones',
                             'Should I avoid spinach and almonds',
                             'Can I eat calcium with oxalate foods',
                             'Do all vegetables contain too much oxalate'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_014': {'questions': ['How are calcium phosphate stones managed',
                             'Are phosphate stones the same as oxalate stones',
                             'Does urine pH matter for phosphate stones',
                             'Should I change urine acidity by myself'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_015': {'questions': ['What should I eat for uric acid stones',
                             'Do organ meats and purines matter',
                             'Can uric acid stones be treated differently',
                             'Does urine acidity affect uric acid stones'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_016': {'questions': ['What is a cystine stone',
                             'Are infection stones called struvite',
                             'Why do infection related stones need special care',
                             'Is cystinuria the same as kidney cysts'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_017': {'questions': ['How can I reduce sodium for stones',
                             'Are packaged snacks and pickles too salty',
                             'Does salt on labels mean the same as sodium',
                             'Which foods hide a lot of salt'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_018': {'questions': ['Do I have to stop eating protein',
                             'Can a vegetarian eat dal with stones',
                             'Should I limit large meat portions',
                             'How do plant and animal protein choices differ'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_019': {'questions': ['Stone in pregnancy what should I do',
                             'What if a child has a stone',
                             'I have a stone in my only working kidney',
                             'Does CKD change stone treatment'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_020': {'questions': ['Show a simple daily meal pattern for stones',
                             'Suggest breakfast and dinner for stone prevention',
                             'Can I build a meal with roti and dal',
                             'Give a meal example without banning every food'],
               'applicability': 'Illustrative meal adaptations only; not a nutritionally calculated or '
                                'universally low-oxalate menu.'},
 'STONE_021': {'questions': ['How does shock wave stone treatment work',
                             'What happens after ESWL',
                             'Will fragments pass after lithotripsy',
                             'Do I need follow up after shock waves'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_022': {'questions': ['What happens during ureteroscopy',
                             'Why did the doctor insert a DJ stent',
                             'What is a nephrostomy drainage tube',
                             'How is PCNL different from laser through the ureter'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_001': {'questions': ['What is a kidney cyst',
                            'Is a simple cyst a fluid filled sac',
                            'My scan says cyst what does that mean',
                            'Does every kidney cyst cause disease'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_002': {'questions': ['Can a kidney cyst cause pain',
                            'Can cysts cause fever or blood in urine',
                            'What problems can a cyst cause',
                            'Are symptoms definitely from my cyst'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_003': {'questions': ['Does a simple cyst need regular scans',
                            'Can a harmless cyst be left alone',
                            'Why did my doctor say no treatment is needed',
                            'When should a simple cyst be reviewed'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_004': {'questions': ['How are painful simple cysts treated',
                            'Can doctors drain a kidney cyst',
                            'Will all cysts need surgery',
                            'What if a cyst presses on urine drainage'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_005': {'questions': ['What does complex cyst mean',
                            'Explain the Bosniak classification',
                            'Is a thick walled cyst automatically cancer',
                            'What do septations and enhancement mean'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_006': {'questions': ['Is one cyst the same as polycystic kidney disease',
                            'Does PKD run in families',
                            'Why are there many cysts in both kidneys',
                            'How is inherited PKD diagnosed'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_007': {'questions': ['What can I eat with a simple kidney cyst',
                            'Is there a diet to shrink a cyst',
                            'Do I need to stop milk or protein for a cyst',
                            'Can normal balanced meals be okay with a cyst'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_008': {'questions': ['What follow up is needed for PKD',
                            'How do doctors monitor polycystic kidney disease',
                            'Does blood pressure matter in PKD',
                            'Can PKD affect kidney function'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_009': {'questions': ['What should I eat if I have PKD',
                            'Should everyone with PKD drink extra water',
                            'Does reduced kidney function change a PKD diet',
                            'Can salt and fluid advice differ for PKD'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'CYST_010': {'questions': ['What should I ask at a cyst appointment',
                            'Can I work and exercise with kidney cysts',
                            'Which old scans should I bring to review',
                            'How do I keep track of a cyst finding'],
              'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_001': {'questions': ['The app says tumor do I have cancer',
                             'What should I do after a tumor label',
                             'Does tumor mean a confirmed diagnosis',
                             'Can a kidney lump be benign'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_002': {'questions': ['What symptoms can kidney cancer cause',
                             'Does blood in urine always mean cancer',
                             'What are kidney cancer risk factors',
                             'Can a kidney tumor have no symptoms'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_003': {'questions': ['Which tests investigate a kidney mass',
                             'Will I need a CT MRI or biopsy',
                             'How do doctors check if a renal mass is cancer',
                             'What information is missing from the app result'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_004': {'questions': ['How is kidney cancer treated',
                             'Are surgery ablation and medicines options',
                             'What determines the cancer treatment plan',
                             'Is treatment the same for every kidney tumor'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_005': {'questions': ['What is cancer stage compared with grade',
                             'Can the image label tell cancer stage',
                             'Does the tumor score predict survival',
                             'How is spread of kidney cancer assessed'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_006': {'questions': ['Why watch a small kidney mass instead of operating',
                             'What is active surveillance',
                             'Is watchful waiting the same as surveillance',
                             'Does monitoring mean nothing is being done'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_007': {'questions': ['Can only part of the kidney be removed',
                             'Will kidney surgery mean dialysis',
                             'What is partial versus radical nephrectomy',
                             'How do doctors try to preserve kidney function'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_008': {'questions': ['What are immunotherapy and targeted medicines',
                             'Who should I call about treatment side effects',
                             'Can cancer medicines cause diarrhoea or fatigue',
                             'Should I stop treatment because of a new symptom'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_009': {'questions': ['What checks are needed after kidney cancer treatment',
                             'How often will I have scans after surgery',
                             'Why is cancer follow up important',
                             'What should I record between oncology appointments'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_010': {'questions': ['What should I eat during kidney cancer care',
                             'How can I avoid losing weight during treatment',
                             'Should I remove all protein with a kidney tumor',
                             'Can diet cure kidney cancer'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_011': {'questions': ['I feel sick and cannot eat during treatment',
                             'What food helps when appetite is poor',
                             'How can I manage sore mouth or taste changes',
                             'Should I tell the team about ongoing weight loss'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'TUMOR_012': {'questions': ['Are some kidney masses non cancerous',
                             'Can kidney cancer be inherited',
                             'What does angiomyolipoma mean',
                             'Does a family history need specialist advice'],
               'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_001': {'questions': ['When should I go to emergency care',
                               'I cannot pass urine and have severe pain',
                               'Fever chills and side pain what now',
                               'I am vomiting and cannot keep drinks down'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_002': {'questions': ['My pee is red or pink',
                               'Should blood in urine be checked even without pain',
                               'Can blood in urine have several causes',
                               'Can I ignore urine blood because scan looks normal'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_003': {'questions': ['Are painkillers safe for kidneys',
                               'Can I take herbal kidney remedies',
                               'Should I tell the doctor about vitamins and supplements',
                               'Can I change prescribed medicines after the image result'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_004': {'questions': ['What diet applies if kidney function is low',
                               'Do all kidney patients need low potassium food',
                               'I have CKD and a fluid restriction',
                               'Can dialysis change protein and fluid needs'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_005': {'questions': ['How do I handle food safely during cancer treatment',
                               'Should I avoid unsafe raw food if immunity is low',
                               'What kitchen hygiene helps during treatment',
                               'Who sets my food safety restrictions'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_006': {'questions': ['Is the same diet used for stones cysts and tumors',
                               'What if I have two kidney conditions',
                               'Why does food advice differ by result',
                               'How do I combine stone advice with CKD restrictions'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_007': {'questions': ['What should I take to a kidney doctor appointment',
                               'Which questions should I ask my urologist',
                               'How do I explain my symptoms clearly',
                               'Should I bring old scans and medicine names'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'GENERAL_008': {'questions': ['Can the chatbot advise a pregnant person or child',
                               'What if I have a transplant or dialysis',
                               'The scan label and doctor report disagree',
                               'Can this app decide my diagnosis or treatment'],
                 'applicability': 'General adult patient education; individual assessment is required.'},
 'STONE_023': {'questions': ['List foods to include and limit for stones',
                             'Tell me what food to take and not take',
                             'Give an allowed and limited foods list',
                             'I do not know stone type can you still give food examples'],
               'applicability': 'General stone-prevention food choices when composition is unknown; adapt '
                                'for prescribed restrictions.'},
 'STONE_024': {'questions': ['List foods for confirmed calcium oxalate stones',
                             'Which high oxalate foods should I review',
                             'Should all fruit and vegetables be banned',
                             'Why keep normal food calcium with oxalate stones'],
               'applicability': 'Confirmed calcium oxalate stones; restriction depends on urine findings and '
                                'the clinical plan.'},
 'STONE_025': {'questions': ['List food choices for uric acid stones',
                             'What meat and seafood choices matter for purines',
                             'Which foods should I limit for uric acid risk',
                             'Can food alone guarantee uric acid stone clearance'],
               'applicability': 'Confirmed uric acid stones, with individualized urine and nutrition '
                                'assessment.'},
 'STONE_026': {'questions': ['Give me a simple stone prevention menu',
                             'What can I have for breakfast lunch and dinner',
                             'Suggest food swaps using Indian meals',
                             'Can you show everyday meal examples for stones'],
               'applicability': 'Illustrative adult meal combinations; not a personalized or '
                                'nutrient-calculated menu.'},
 'STONE_027': {'questions': ['Which drinks are sensible for stone prevention',
                             'How can I remember to drink through the day',
                             'Is water a practical everyday drink',
                             'Should sugary drinks replace water'],
               'applicability': 'General prevention; prescribed fluid limits and urgent assessment take '
                                'priority.'},
 'STONE_028': {'questions': ['How do I shop for lower salt foods',
                             'What should I check on a sodium label',
                             'Can I reduce pickle papad and salty snacks',
                             'How do I eat out with less salt'],
               'applicability': 'General sodium-reduction examples; individualized targets and potassium '
                                'restrictions may apply.'},
 'NORMAL_011': {'questions': ['List foods for a normal kidney result',
                              'What should I include in ordinary healthy meals',
                              'Do I need to avoid fruit milk or lentils if normal',
                              'Give healthy food examples after a clear scan'],
                'applicability': 'General adult eating with no known restriction; a normal image label does '
                                 'not establish kidney function.'},
 'CYST_011': {'questions': ['List foods if my cyst is confirmed simple',
                            'What can I eat with a harmless kidney cyst',
                            'Can food dissolve a simple cyst',
                            'Are ordinary meals okay if kidney function is normal'],
              'applicability': 'Confirmed simple cyst; not a substitute for complex-cyst or PKD guidance.'},
 'TUMOR_013': {'questions': ['Give food examples for low appetite during cancer care',
                             'What can I eat if treatment makes my mouth sore',
                             'Suggest small frequent meals during treatment',
                             'How do I adapt meals for nausea or taste changes'],
               'applicability': 'Confirmed diagnosis or treatment-related symptoms; coordinate nutrition '
                                'with oncology and kidney care.'},
 'NORMAL_012': {'questions': ['My scan is clear but my back hurts',
                              'Normal result still having side pain',
                              'Can the app miss the reason for my pain',
                              'Do I need care even if the picture looks normal'],
                'applicability': 'Any classifier label; persistent symptoms require assessment independent '
                                 'of the prediction.'},
 'NORMAL_013': {'questions': ['My pee is foamy',
                              'Bubbles in urine mean kidney failure',
                              'Protein in urine with normal CT',
                              'How do I check albumin leakage'],
                'applicability': 'Urine appearance is not diagnostic; use measured urine and blood results.'},
 'NORMAL_014': {'questions': ['My feet are puffy and I hardly pee',
                              'Normal CT but less urine today',
                              'Can I drink more to fix swelling',
                              'I feel full bladder but no urine comes'],
                'applicability': 'Possible fluid or urinary problem; do not infer cause or prescribe fluid '
                                 'or diuretic treatment.'},
 'NORMAL_015': {'questions': ['Creatinine high what does that mean',
                              'Is my eGFR number dangerous',
                              'Blood test abnormal but image normal',
                              'Can one kidney test diagnose chronic disease'],
                'applicability': 'General explanation of laboratory terms; no individual staging or dose '
                                 'adjustment from incomplete results.'},
 'NORMAL_016': {'questions': ['Can I go to the gym with a normal result',
                              'Is walking good for kidney health',
                              'How should I manage sleep and daily routine',
                              'Do I need a kidney detox if my scan is normal'],
                'applicability': 'General health habits; symptom or postoperative restrictions override '
                                 'routine activity examples.'},
 'NORMAL_017': {'questions': ['My mother has kidney disease am I at risk',
                              'Family history but no kidney symptoms',
                              'Should healthy relatives have urine tests',
                              'Does kidney disease always run in families'],
                'applicability': 'Family risk assessment; distinguish inherited disorders from other common '
                                 'kidney conditions.'},
 'NORMAL_018': {'questions': ['I have only one kidney can I live normally',
                              'Will a stone harm my only working kidney',
                              'Do I need a special diet with a single kidney',
                              'What follow up does a kidney donor need'],
                'applicability': 'One functioning kidney; obstruction symptoms require prompt assessment and '
                                 'plans depend on function.'},
 'NORMAL_019': {'questions': ['My sugar is high but kidney CT is normal',
                              'Can I stop BP tablets after a clear scan',
                              'Why do I need urine tests when I feel okay',
                              'Diabetes and high blood pressure kidney monitoring'],
                'applicability': 'Ongoing diabetes and blood-pressure care; no medication changes from an '
                                 'image label.'},
 'STONE_029': {'questions': ['Will a six millimetre stone come out on its own',
                             'Can a small ureter stone pass naturally',
                             'Does an eight mm stone always need an operation',
                             'How many days until my stone passes'],
               'applicability': 'General passage factors; no individual probability, safe waiting period or '
                                'procedure decision from size alone.'},
 'STONE_030': {'questions': ['My kidney is swollen from a blocked pipe',
                             'What does hydronephrosis mean',
                             'Stone plus fever and shivering',
                             'Why drain the kidney before removing the stone'],
               'applicability': 'Suspected obstruction; infection or severe symptoms take priority over '
                                'routine education.'},
 'STONE_031': {'questions': ['Should I save the gravel I passed',
                             'What is a twenty four hour urine collection',
                             'I missed one urine sample during collection',
                             'Can stone colour show its type'],
               'applicability': 'General specimen guidance; local laboratory instructions determine the '
                                'collection method.'},
 'STONE_032': {'questions': ['The DJ stent is making me pee often',
                             'Burning after they put a tube in my ureter',
                             'When will they remove my stent',
                             'There is a string after stone surgery'],
               'applicability': 'Ureteric stent education; severe symptoms and removal timing require the '
                                'treating team.'},
 'STONE_033': {'questions': ['Can I attend college after laser stone treatment',
                             'How soon can I lift weights after an operation',
                             'When can I resume sex after ureteroscopy',
                             'Do I still need a scan when surgery pain is gone'],
               'applicability': 'Procedure-specific recovery; discharge instructions take priority over '
                                'generic timelines.'},
 'STONE_034': {'questions': ['Can I take ibuprofen for kidney pain',
                             'Stone pain tablets are not helping',
                             'What pain medicine dose is safe for me',
                             'Can I combine two painkiller brands'],
               'applicability': 'Medicine safety education only; no personalised product selection or '
                                'dosing.'},
 'STONE_035': {'questions': ['Do I need antibiotics just because a stone was found',
                             'Burning pee is it a stone or UTI',
                             'Can I use antibiotics left from last time',
                             'Will antibiotics dissolve my kidney stone'],
               'applicability': 'Distinguish infection treatment from stone treatment; fever with possible '
                                'obstruction needs urgent assessment.'},
 'STONE_036': {'questions': ['Stones came back even though I followed the diet',
                             'Why do I keep making kidney stones',
                             'Could this be a leftover fragment',
                             'Is recurrent stone disease my fault'],
               'applicability': 'Recurrence assessment and prevention; distinguish new stones from residual '
                                'or growing stones.'},
 'STONE_037': {'questions': ['Can lemon juice melt my stone',
                             'Will coconut water clean my kidneys',
                             'Does beer push stones out',
                             'Can I dissolve a stone using a home remedy'],
               'applicability': 'Distinguish beverages from prescribed stone treatment; no cure or universal '
                                'fluid target.'},
 'STONE_038': {'questions': ['Are tomato seeds banned for stones',
                             'Can I eat banana with kidney stones',
                             'Are eggs allowed with a stone',
                             'Do I have to stop tea and coffee'],
               'applicability': 'Food-group guidance; no invented nutrient tables or individual permitted '
                                'amounts.'},
 'STONE_039': {'questions': ['Can vitamin C tablets cause stone problems',
                             'Should I stop calcium supplements',
                             'Is whey protein safe for my kidneys',
                             'Are herbal kidney capsules harmless'],
               'applicability': 'Supplement review; no individual dose, safety certification or unverified '
                                'ingredient claims.'},
 'STONE_040': {'questions': ['How do I manage stones while travelling',
                             'I work in heat and sweat a lot',
                             'I forget water during college classes',
                             'Can I go on a trip while waiting for stone passage'],
               'applicability': 'Practical planning; fluid limits, current symptoms and procedure follow-up '
                                'determine individual advice.'},
 'STONE_041': {'questions': ['Can a vegetarian get enough protein with stones',
                             'Milk upsets my stomach what can I use',
                             'Is plant milk always safe for kidney stones',
                             'Do I need to stop all dal and beans'],
               'applicability': 'Adaptation of food groups; do not infer product nutrients or calcium '
                                'equivalence from the name.'},
 'STONE_042': {'questions': ['I am pregnant and may have a stone',
                             'My child has a kidney stone result',
                             'Can a child use adult stone medicines',
                             'Are CT and tablets safe in pregnancy'],
               'applicability': 'Special populations: general explanation and referral only; no adult '
                                'treatment extrapolation.'},
 'STONE_043': {'questions': ['No pain now is my stone gone',
                             'I feel better can I skip follow up',
                             'Does painless mean there is no blockage',
                             'I passed a tiny fragment am I stone free'],
               'applicability': 'Symptom improvement is not confirmation of clearance; follow-up depends on '
                                'the original clinical findings.'},
 'STONE_044': {'questions': ['What is renal calculus on my report',
                             'What does bilateral nephrolithiasis mean',
                             'Is a ureter stone different from a kidney stone',
                             'What is a staghorn stone'],
               'applicability': 'Explanation of report terminology; no diagnosis or procedure decision from '
                                'an isolated phrase.'},
 'CYST_012': {'questions': ['Is a kidney cyst cancer',
                            'Will my cyst become a tumour',
                            'Should I be scared of a fluid sac on kidney',
                            'Can all kidney cysts be ignored'],
              'applicability': 'Requires radiological characterization; distinguish confirmed simple cyst '
                               'from complex cystic mass.'},
 'CYST_013': {'questions': ['What does Bosniak mean',
                            'Is a complex renal cyst definitely cancer',
                            'Why does my cyst need contrast imaging',
                            'Does Bosniak mean cancer stage'],
              'applicability': 'Specialist imaging terminology; do not assign a category or individual risk '
                               'from incomplete information.'},
 'CYST_014': {'questions': ['My father has PKD should I be tested',
                            'Do several cysts mean inherited kidney disease',
                            'Should my siblings have a scan',
                            'Can genetic counselling help before pregnancy'],
              'applicability': 'Suspected inherited PKD; family-specific testing and counselling are '
                               'specialist decisions.'},
 'CYST_015': {'questions': ['What tests do I need with PKD',
                            'Should PKD follow up include blood pressure',
                            'Can the app show that PKD is getting worse',
                            'How do I monitor inherited kidney cyst disease'],
              'applicability': 'Confirmed PKD follow-up; no progression estimate or specialist treatment '
                               'selection from the classifier.'},
 'CYST_016': {'questions': ['My kidney cyst hurts suddenly',
                            'Blood in pee when I have a cyst',
                            'Can a renal cyst get infected',
                            'Has my cyst burst because my back hurts'],
              'applicability': 'Symptom assessment; no attribution of bleeding, infection or rupture without '
                               'clinical evidence.'},
 'CYST_017': {'questions': ['Why am I not getting another scan for a simple cyst',
                            'How often should a renal cyst be scanned',
                            'My friends cyst follow up is different',
                            'Does a bigger cyst measurement mean danger'],
              'applicability': 'No universal follow-up interval; use confirmed cyst features and the '
                               'individual clinical plan.'},
 'CYST_018': {'questions': ['Can I go to the gym with a kidney cyst',
                            'Will lifting weights burst my cyst',
                            'Can I keep working with PKD',
                            'Is massage a way to remove a cyst'],
              'applicability': 'General activity discussion; no individual clearance or attempt to '
                               'manipulate a cyst.'},
 'CYST_019': {'questions': ['Can I become pregnant with a kidney cyst',
                            'Does PKD affect planning a baby',
                            'Should I change tablets before pregnancy',
                            'Will a cyst make me infertile'],
              'applicability': 'Pregnancy and fertility questions require individual assessment; no risk '
                               'prediction from a cyst label.'},
 'TUMOR_014': {'questions': ['Tumour result means I definitely have cancer right',
                             'Can a kidney mass be benign',
                             'Is the model percentage my cancer probability',
                             'What happens after a suspicious mass is found'],
               'applicability': 'Unconfirmed mass assessment; classifier confidence is not clinical cancer '
                                'probability.'},
 'TUMOR_015': {'questions': ['Does every renal mass need biopsy',
                             'Why did my doctor skip a kidney biopsy',
                             'What if the biopsy sample is unclear',
                             'What should I ask before a mass biopsy'],
               'applicability': 'General purpose of renal-mass biopsy; no individual recommendation or '
                                'anticoagulant instruction.'},
 'TUMOR_016': {'questions': ['Why watch a kidney tumour rather than operate',
                             'Is active surveillance doing nothing',
                             'What makes doctors stop monitoring and treat',
                             'Do I need follow up if the mass is painless'],
               'applicability': 'Selected renal masses; surveillance requires an individualized specialist '
                                'plan.'},
 'TUMOR_017': {'questions': ['Can they remove the lump and save my kidney',
                             'What does partial nephrectomy mean',
                             'Will kidney removal mean dialysis',
                             'Why is my surgeon recommending radical nephrectomy'],
               'applicability': 'General surgical concepts; anatomy, function and specialist assessment '
                                'determine the operation.'},
 'TUMOR_018': {'questions': ['Feeling very weak after kidney cancer treatment',
                             'Do I wait until my next visit if I develop fever',
                             'New diarrhoea while taking cancer medicine',
                             'Is breathlessness after treatment just fatigue'],
               'applicability': 'Treatment-specific assessment and urgent-contact instructions take '
                                'priority; no remote adverse-effect diagnosis.'},
 'TUMOR_019': {'questions': ['How long will I live with a kidney tumour',
                             'Does the result mean I am dying',
                             'Can kidney cancer ever be cured',
                             'Can you predict my survival from the scan percentage'],
               'applicability': 'No personal survival prediction; discuss prognosis with the team holding '
                                'the confirmed diagnosis and staging.'},
 'TUMOR_020': {'questions': ['I am frightened by the tumour result',
                             'How do I tell my family about the scan',
                             'I forget everything the doctor says',
                             'Can someone come with me to the appointment'],
               'applicability': 'Supportive education; respect patient choice and escalate immediate safety '
                                'concerns.'},
 'TUMOR_021': {'questions': ['Is it okay to get another opinion about my kidney mass',
                             'Can I join a cancer research study',
                             'How much will my treatment cost in my city',
                             'Who can help if I cannot afford appointments'],
               'applicability': 'Explain options and questions; no unverified local prices, programme '
                                'eligibility or active trial listings.'},
 'GENERAL_009': {'questions': ['I cannot pee at all',
                               'Fever shivering and pain near my kidney',
                               'Should I wait until tomorrow with severe symptoms',
                               'Normal scan but I am suddenly very unwell'],
                 'applicability': 'Urgent-action guidance applies across all result classes; not a diagnosis '
                                  'or exhaustive triage system.'},
 'GENERAL_010': {'questions': ['Can a child follow this kidney diet',
                               'I am breastfeeding which tablet can I use',
                               'Does dialysis change the water advice',
                               'Do transplant patients use the same kidney guidance'],
                 'applicability': 'Special-population scope and referral; no individualized pregnancy, '
                                  'paediatric, transplant or dialysis treatment.'},
 'GENERAL_011': {'questions': ['I forgot a tablet should I take two',
                               'Can I mix my kidney medicine with a supplement',
                               'I only know the colour of my pill',
                               'Can you choose a medicine dose for my result'],
                 'applicability': 'Medication information and verification only; no personal dosing, '
                                  'interaction clearance or overdose management protocol.'},
 'GENERAL_012': {'questions': ['Will contrast dye damage my kidneys',
                               'Why ultrasound instead of CT for this problem',
                               'Do I need to fast before my scan',
                               'What should I tell the MRI team'],
                 'applicability': 'Explain imaging concepts; facility and clinician determine test choice, '
                                  'preparation and contrast risk.'},
 'GENERAL_013': {'questions': ['My report says 7 is that serious',
                               'Explain my scan in simple language',
                               'The doctor report disagrees with the app',
                               'What information should I copy from my lab report'],
                 'applicability': 'Terminology and clarification; no interpretation of absent images, units '
                                  'or findings.'},
 'GENERAL_014': {'questions': ['It burns when I pee',
                               'I keep rushing to the toilet but only a little comes',
                               'Does cloudy urine mean I have stones',
                               'Frequent urine with a normal kidney scan'],
                 'applicability': 'Urinary symptom assessment; no diagnosis from burning, frequency, colour '
                                  'or smell alone.'},
 'GENERAL_015': {'questions': ['Can I fast during Ramadan with kidney problems',
                               'Is intermittent fasting safe with stones',
                               'Can I skip water for a religious fast',
                               'How should I change tablets while fasting'],
                 'applicability': 'Individual pre-fast assessment; no universal permission, timetable or '
                                  'medicine adjustment.'},
 'GENERAL_016': {'questions': ['Can I have sex with a kidney stone',
                               'Does a kidney cyst mean infertility',
                               'Burning urine and discharge after sexual contact',
                               'Can intercourse displace a stent string'],
                 'applicability': 'Private health education; procedure advice and sexual-health assessment '
                                  'require the appropriate clinician.'},
 'GENERAL_017': {'questions': ['Kidney stones and CKD diets contradict each other',
                               'What should I eat with diabetes and a stone',
                               'My doctor limits water but stone advice says drink more',
                               'How can cancer nutrition fit a kidney diet'],
                 'applicability': 'Coordinated care for coexisting conditions; prescribed limits take '
                                  'priority over general examples.'}}


# These are retrieval pointers to existing Word topics, never canned answers.
DIET_TOPICS = {
    "stone": ("STONE_023", "STONE_026", "STONE_008", "STONE_009", "STONE_027"),
    "normal": ("NORMAL_011", "NORMAL_004", "NORMAL_009", "NORMAL_010"),
    "cyst": ("CYST_011", "CYST_001", "CYST_007", "CYST_006"),
    "tumor": ("TUMOR_013", "TUMOR_001", "TUMOR_010", "TUMOR_011"),
}
DIET_WORDS = re.compile(
    r"\b(?:diet\w*|food\w*|eat\w*|meal\w*|nutrition\w*|breakfast|lunch|dinner|"
    r"snack\w*|vegetarian|vegan|milk|dairy|curd|rice|roti|dal|spinach|nuts?|"
    r"protein|salt|sodium|oxalate|water|drink\w*|fluid\w*|hydration)\b", re.I
)
CLASS_WORDS = {
    "stone": r"\b(?:stones?|calculus|calculi|urolithiasis)\b",
    "normal": r"\bnormal\b",
    "cyst": r"\b(?:cysts?|polycystic|pkd|bosniak)\b",
    "tumor": r"\b(?:tumou?rs?|cancers?|carcinoma|renal mass)\b",
}
SHARED_TEST_TOPICS = {
    "NORMAL_002", "NORMAL_005", "NORMAL_006", "NORMAL_007", "NORMAL_008",
    "NORMAL_012", "NORMAL_013", "NORMAL_014", "NORMAL_015", "NORMAL_018", "NORMAL_019",
}
SEARCH_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | {
    'match', 'matches', 'matched', 'matching', 'ask', 'tell', 'give',
    'please', 'want', 'need', 'needs', 'usual', 'ordinary', 'suitable',
    'today', 'tomorrow', 'yesterday', 'really', 'actually', 'hello', 'hi',
    'pls', 'plz', 'okay', 'thanks', 'thank',
})


def search_text(text):
    """Normalize search wording only; never modify the patient's message."""
    text = unicodedata.normalize('NFKC', text).lower().replace('\u2019', "'")
    for pattern, replacement in (
        (r'\b(?:peeing|pee|pissing)\b', 'urine urination'),
        (r'\b(?:bp|b p)\b', 'blood pressure'),
        (r'\b(?:dj|double j)\b', 'ureteric stent'),
        (r'\b(?:kidny|kidne|kidneys)\b', 'kidney'),
        (r'\b(?:ston|stons|stones)\b', 'stone'),
        (r'\btumour\b', 'tumor'),
        (r'\b(?:curd|yoghurt)\b', 'yogurt curd'),
        (r'\b(?:sugar disease|high sugar|sugar level)\b', 'blood glucose diabetes'),
        (r'\b(?:cant|cannot|can\'t)\b', 'cannot'),
        (r'\b(?:froth|frothy)\b', 'foamy'),
        (r'\b(?:father|mother|brother|sister|parent|relative)s?\b', 'family relative'),
        (r'\b(?:donated|donating|donor)\b', 'donor single functioning kidney'),
        (r'\b(?:kids?|child|paediatric|pediatric)\b|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[ -]year[ -]old\b', 'child paediatric'),
        (r'\bcolic\b', 'pain'),
        (r'\b(?:vanished|disappeared|ceased)\b', 'stopped gone'),
        (r'\b(?:bodybuilding|protein shakes?)\b', 'protein supplements powder'),
        (r'\b(?:journeys?|travelling|traveling)\b', 'travel'),
        (r'\b(?:conceive|conception|baby)\b', 'pregnancy fertility'),
        (r'\b(?:prediction confidence|scan percentage)\b', 'classifier probability'),
        (r'\b(?:observing|observe)\b', 'surveillance monitoring'),
        (r'\b(?:years? (?:i have )?left|life expectancy)\b', 'prognosis survival'),
        (r'\bdawn to sunset\b', 'religious fasting'),
    ):
        text = re.sub(pattern, replacement, text)
    return text


def broad_food_request(text):
    """A food mention in a surgery or supplement question is not a meal request."""
    return bool(re.search(
        r'\b(?:diet|meal|menu|breakfast|lunch|dinner)s?\b|'
        r'\b(?:what|which)\s+food|\bfood\s+(?:list|choices|examples)|'
        r'\bwhat (?:can|should|do) (?:i|we) (?:eat|take)|'
        r'\b(?:list|give|suggest).{0,24}\bfood', text, re.I
    ))


def character_text(text):
    return ' '.join(word for word in re.findall(r'\w+', search_text(text))
                    if word not in SEARCH_STOP_WORDS)
FORMAT_WORDS = re.compile(r"\b(?:list|bullets?|points|table|examples?|shorter|simpler|simplify)\b", re.I)
FORMAT_FILLER = set(
    'can could would will you u please pls plz it that this them those these '
    'give show make tell me a an the as in to with and just only same answer '
    'response points point form format list bullets bullet table short shorter '
    'simple simpler simplify words again yes ok okay examples example more '
    'explain detail details'.split()
)


def format_only(question):
    """Recognize a request to reformat without mistaking a new topic for it."""
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    return bool(FORMAT_WORDS.search(question)) and bool(tokens) and tokens <= FORMAT_FILLER


def request_context(question, history=None):
    """Resolve short follow-ups using user questions, never prior bot claims."""
    previous = [
        item['content'] for item in (history or [])[-8:]
        if isinstance(item, dict) and item.get('role') == 'user'
        and isinstance(item.get('content'), str)
    ]
    substantive = [q for q in previous if not format_only(q)
                   and q.lower().strip(' .!?') not in {'hi','hello','thanks','thank you','ok','okay'}]
    is_format_request = format_only(question)
    followup = is_format_request or bool(re.search(
        r'^(?:why\b|how much\b|how many\b|what about\b)|\b(?:it|that|those|them|this)\b', question, re.I
    ))
    prior = substantive[-1] if followup and substantive else ''
    relevant = prior if is_format_request and prior else question
    diet = bool(DIET_WORDS.search(question)) or bool(prior and DIET_WORDS.search(prior))
    return dict(previous=previous, prior_question=prior, relevant_question=relevant,
                followup=followup, format_only=is_format_request,
                requested_format='list' if FORMAT_WORDS.search(question) and re.search(r'\b(?:list|bullets?|points)\b', question, re.I)
                else 'table' if re.search(r'\btable\b', question, re.I)
                else 'examples' if re.search(r'\bexamples?\b', question, re.I)
                else 'conversational', diet_question=diet)


class KidneyKnowledgeBase:
    def __init__(self, path=None):
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent
            / "data" / "raw" / "kidney_knowledge.docx"
        )
        if not self.path.is_file():
            raise FileNotFoundError(f"Knowledge base not found: {self.path}")

        paragraphs = []
        for block in Document(str(self.path)).iter_inner_content():
            texts = (
                [cell.text for row in block.rows for cell in row.cells]
                if isinstance(block, Table) else [block.text]
            )
            for text in texts:
                for line in text.splitlines():
                    line = unicodedata.normalize("NFKC", line)
                    line = line.replace("\ufeff", "").replace("\u200b", "")
                    line = line.replace("\u00a6", "|").strip()
                    if line:
                        paragraphs.append(line)
        sources = {}
        for text in paragraphs:
            match = re.match(r"^(S\d{2})\s+(.+)$", text, re.IGNORECASE)
            if match:
                sources[match[1].upper()] = {
                    "id": match[1].upper(), "title": match[2],
                }

        self.topics = []
        current = None
        for text in paragraphs:
            match = re.fullmatch(
                r"(?:(?:NORMAL|STONE|CYST|TUMOR|GENERAL|SHARED\s+CARE)"
                r"\s*[|:]\s*)?"
                r"((?:NORMAL|STONE|CYST|TUMOR|GENERAL)_\d{3})",
                text, re.IGNORECASE
            )
            if match:
                current = {"id": match[1].upper(), "lines": []}
                self.topics.append(current)
            elif text.lower().startswith("glossary "):
                current = None
            elif current is not None:
                current["lines"].append(text)

        if not self.topics:
            raise ValueError(
                f"Opened {self.path} ({len(paragraphs)} nonempty text lines), "
                "but found no structured topic headings such as NORMAL_001. "
                "Replace data/raw/kidney_knowledge.docx with the downloaded "
                "expanded knowledge base from the current ZIP. Renaming the original raw "
                "dataset does not add these headings."
            )

        for topic in self.topics:
            lines = topic.pop("lines")
            if len(lines) < 3:
                raise ValueError(f"Incomplete topic: {topic['id']}")
            topic["title"] = lines[0]
            topic["condition"] = topic["id"].split("_")[0].lower()
            # Source names stay in metadata. They are not patient reply text.
            topic["text"] = "\n".join(
                line for line in lines
                if not line.lower().startswith(("sources:", "http://", "https://", "www."))
            )
            topic["review_status"] = "pending"
            hints = TOPIC_HINTS.get(topic['id'], {})
            topic['question_examples'] = [lines[1]] + hints.get('questions', [])
            topic['applicability'] = hints.get(
                'applicability', 'General education; preserve the conditions stated in the passage.'
            )
            reference_line = next(
                (line for line in lines if line.lower().startswith("sources:")), ""
            )
            ids = [
                source_id.upper() for source_id in
                re.findall(r"\bS\d{2}\b", reference_line, re.IGNORECASE)
            ]
            if not ids or any(source_id not in sources for source_id in ids):
                raise ValueError(f"Missing source names for {topic['id']}")
            topic["sources"] = [sources[source_id] for source_id in ids]

        self.vectorizer = TfidfVectorizer(
            stop_words=SEARCH_STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(
            [search_text(f"{topic['title']} {topic['title']} {topic['text']}") for topic in self.topics]
        )
        # Index example questions separately, so a specific question is not
        # diluted by a longer clinical article. The answers still come from Word.
        examples, self.example_topics = [], []
        for index, topic in enumerate(self.topics):
            for example in [topic['title']] + topic['question_examples']:
                examples.append(search_text(example))
                self.example_topics.append(index)
        self.question_vectorizer = TfidfVectorizer(
            stop_words=SEARCH_STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True
        )
        self.question_matrix = self.question_vectorizer.fit_transform(examples)
        self.character_vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True
        )
        self.character_matrix = self.character_vectorizer.fit_transform(
            [character_text(example) for example in examples]
        )

    def search(self, question, condition=None, top_k=8, min_score=0.04, history=None):
        if not question or not question.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if condition:
            condition = condition.strip().lower()
            if condition not in {"normal", "stone", "cyst", "tumor"}:
                raise ValueError("condition must be normal, stone, cyst or tumor.")

        question = question.strip()
        mentioned = {name for name, pattern in CLASS_WORDS.items() if re.search(pattern, question, re.I)}
        context = request_context(question, history)
        previous = context['previous']
        follow_up = not mentioned and context['followup']
        prior_context = context['prior_question'] if follow_up else ''
        query_text = '\n'.join([prior_context, question, question]).strip()
        target = mentioned
        if not target and follow_up:
            # An explicit new condition overrides the scan context; a short
            # follow-up continues the most recently discussed condition.
            for earlier in reversed(previous):
                target = {name for name, pattern in CLASS_WORDS.items() if re.search(pattern, earlier, re.I)}
                if target:
                    break
        if not target and condition:
            target = {condition}

        diet_question = bool(DIET_WORDS.search(question)) or bool(prior_context and DIET_WORDS.search(prior_context))
        broad_food = diet_question and broad_food_request(context['relevant_question'])
        if re.search(r'\banother\s+stone|\bstone.{0,35}\b(?:again|returned|recurs|back)\b', query_text, re.I):
            query_text += '\n recurrence prevention recurrent stone'
            broad_food = False
        if broad_food:
            query_text += '\n diet food meals eating nutrition'
            if context['requested_format'] == 'list' or re.search(r'what food|which food|what.*(?:eat|take)|foods.*(?:avoid|limit)', context['relevant_question'], re.I):
                query_text += '\n foods to include and limit food list'

        normalized_query = search_text(query_text)
        query = self.vectorizer.transform([normalized_query])
        body_scores = cosine_similarity(query, self.matrix)[0]
        question_matches = cosine_similarity(
            self.question_vectorizer.transform([normalized_query]), self.question_matrix
        )[0]
        character_matches = cosine_similarity(
            self.character_vectorizer.transform([character_text(query_text)]), self.character_matrix
        )[0]
        question_scores, character_scores = body_scores * 0, body_scores * 0
        for row, index in enumerate(self.example_topics):
            question_scores[index] = max(question_scores[index], question_matches[row])
            character_scores[index] = max(character_scores[index], character_matches[row])
        scores = 0.45 * body_scores + 0.40 * question_scores + 0.15 * character_scores
        # Weak accidental spelling overlap is insufficient on its own.
        scores[(body_scores == 0) & (question_scores == 0) & (character_scores < 0.35)] = 0
        ranked_scores = scores.copy()
        for i, topic in enumerate(self.topics):
            if topic['condition'] in target:
                ranked_scores[i] *= 1.3

        indices = sorted(
            range(len(self.topics)), key=lambda i: ranked_scores[i], reverse=True
        )
        ranked = [i for i in indices if scores[i] >= min_score and (
            not target or self.topics[i]['condition'] in target | {'general'}
            or self.topics[i]['id'] in SHARED_TEST_TOPICS
        )]
        selected = ranked[:2]

        def add(index):
            if index not in selected:
                selected.append(index)

        # Keep the strongest specific matches, then ensure a broad dietary
        # question includes the relevant overview and real meal examples.
        if broad_food:
            topic_index = {topic['id']: i for i, topic in enumerate(self.topics)}
            support_ids = []
            if re.search(r"\b(?:ckd|dialysis|egfr)\b|fluid (?:limit|restriction)|reduced kidney function", '\n'.join(previous + [question]), re.I):
                support_ids.append('GENERAL_004')
            if not target or len(target) > 1:
                support_ids.append('GENERAL_006')
            for name in sorted(target):
                support_ids.extend(DIET_TOPICS.get(name, ()))
            # Composition-specific passages must survive a broad diet bundle.
            if 'stone' in target:
                specific = []
                if re.search(r'calcium oxalate|oxalate|spinach|almonds?|rhubarb', query_text, re.I):
                    specific += ['STONE_024', 'STONE_013']
                if re.search(r'uric acid|purine', query_text, re.I):
                    specific += ['STONE_025', 'STONE_015']
                if re.search(r'sodium|salt|pickles?|papad|shopping|restaurant', context['relevant_question'], re.I):
                    specific += ['STONE_028']
                support_ids = specific + support_ids
            for topic_id in support_ids:
                if topic_id in topic_index:
                    add(topic_index[topic_id])
        for i in ranked:
            add(i)
        return [{**self.topics[i], 'retrieval_score': round(float(scores[i]), 4)} for i in selected[:top_k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect kidney knowledge-base retrieval without calling Gemini.')
    parser.add_argument('path', nargs='?', help='Optional Word document path.')
    parser.add_argument('--question', default='what the diet should i follow')
    parser.add_argument('--condition', choices=('normal', 'stone', 'cyst', 'tumor'), default='stone')
    args = parser.parse_args()
    kb = KidneyKnowledgeBase(args.path)
    print(f"Loaded {len(kb.topics)} knowledge-base topics.")
    print(f"Knowledge base file: {kb.path.resolve()}")
    results = kb.search(args.question, condition=args.condition)
    print(f'Retrieved {len(results)} topics for: {args.question}')
    for result in results:
        print(f"\n{result['id']} | {result['title']}")
        print(result["text"][:350])
