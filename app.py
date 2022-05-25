import json
import torch
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
stop_words.remove("no")
stop_words.remove("nor")
stop_words.add("feeling")
stop_words.add("feel")

from flask import Flask, render_template, url_for, request, jsonify

random.seed(datetime.now())

device = torch.device('cpu')
FILE = "models/data.pth"
model_data = torch.load(FILE)

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))

symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.applymap(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)


with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

with open('models/fitted_model.pickle', 'rb') as modelFile:
    prediction_model = pickle.load(modelFile)

with open('models/new.pickle', 'rb') as modelFile:
    my_model = pickle.load(modelFile)

user_symptoms = set()

app = Flask(__name__)

def get_symptom(test):
    
    
    '''sentence = nltk.word_tokenize(sentence)
    
    X = bag_of_words(sentence, all_words)'''
     
    
    import re, string

        
    all_words = ["'s",
             '(',
             ')',
             'a',
             'abdomen',
             'abdominal',
             'able',
             'abnormal',
             'ache',
             'acid',
             'acidic',
             'acidity',
             'acne',
             'activity',
             'acute',
             'adam',
             'affair',
             'alcohol',
             'also',
             'altered',
             'always',
             'anal',
             'angry',
             'ankle',
             'annoyed',
             'anus',
             'anxiety',
             'anxious',
             'anything',
             'appears',
             'appetite',
             'apple',
             'area',
             'argyria',
             'arm',
             'around',
             'ascites',
             'atrophy',
             'attack',
             'back',
             'bacterial',
             'bad',
             'bag',
             'balance',
             'balancing',
             'beat',
             'beating',
             'behind',
             'belly',
             'bigger',
             'bilirubin',
             'black',
             'blackhead',
             'bladder',
             'bleeding',
             'bleeds',
             'blindness',
             'blister',
             'blistered',
             'bloated',
             'bloating',
             'blood',
             'bloody',
             'blow',
             'blue',
             'blue-grey',
             'blurred',
             'blurry',
             'body',
             'breath',
             'breathe',
             'breathing',
             'breathlessness',
             'brief',
             'bright',
             'brittle',
             'broken',
             'bronchitis',
             'brown',
             'bruise',
             'bruised',
             'bruising',
             'bump',
             'burn',
             'burning',
             'burp',
             'burping',
             'butt',
             'calm',
             'calorie',
             'caughing',
             'cervical',
             'change',
             'chattering',
             'chest',
             'chill',
             'cholestrol',
             'circle',
             'clearly',
             'clot',
             'cold',
             'color',
             'colored',
             'coma',
             'comfortable',
             'common',
             'concentrate',
             'concentration',
             'confusion',
             'congested',
             'congestion',
             'constantly',
             'constipated',
             'constipation',
             'consumption',
             'contact',
             'continuous',
             'contract',
             'coordination',
             'cough',
             'coughing',
             'cramp',
             'cramping',
             'crust',
             'danger',
             'dark',
             'darker',
             'darkness',
             'deep',
             'dehrydration',
             'dehydrated',
             'dehydration',
             'dent',
             'depressed',
             'depression',
             'diarrhoea',
             'different',
             'difficulty',
             'dischromic',
             'discoloration',
             'discomfort',
             'disease',
             'disorder',
             'disorientation',
             'distention',
             'distorted',
             'disturbance',
             'dizziness',
             'dizzy',
             'doom',
             'double',
             'dried',
             'drink',
             'drinking',
             'dry',
             'drying',
             'dust',
             'dusting',
             'easily',
             'eat',
             'eating',
             'eczema',
             'emotion',
             'emphasized',
             'emptiness',
             'energy',
             'enlarged',
             'enthusiasm',
             'equilibrium',
             'even',
             'everything',
             'everywhere',
             'excessive',
             'excessively',
             'exhaustion',
             'expanded',
             'experienced',
             'experiencing',
             'extra',
             'extreme',
             'extremeties',
             'extremity',
             'eye',
             'eyeball',
             'face',
             'facial',
             'failure',
             'fall',
             'family',
             'fart',
             'farting',
             'fast',
             'fat',
             'fatigue',
             'fatigued',
             'fatter',
             'feel',
             'feeling',
             'feets',
             'fell',
             'fever',
             'fevered',
             'filled',
             'flashing',
             'flow',
             'flu',
             'fluctuating',
             'fluid',
             'focus',
             'food',
             'foot',
             'forehead',
             'foul',
             'freezing/cold',
             'frequent',
             'friction',
             'front',
             'frustrated',
             'frustration',
             'full',
             'gain',
             'gained',
             'gaining',
             'gas',
             'gassy',
             'general',
             'genetically',
             'genetics',
             'getting',
             'gland',
             'go',
             'going',
             'good',
             'got',
             'grassburn',
             'greenish',
             'grey',
             'hand',
             'hanot',
             'happy',
             'hard',
             'have',
             'having',
             'head',
             'headache',
             'heart',
             'heartbeat',
             'heartburn',
             'heat',
             'heavy',
             'hemiparesis',
             'hepatitis',
             'hiccup',
             'high',
             'highered',
             'hip',
             'history',
             'hoarseness',
             'hobby',
             'hopelessness',
             'hormonal',
             'hunger',
             'hungry',
             'hurt',
             'imbalanced',
             'impending',
             'inability',
             'increased',
             'indigestion',
             'infected',
             'infection',
             'inflamed',
             'inflammation',
             'inflammatory',
             'injected',
             'injecting',
             'injection',
             'injury',
             'inner',
             'inside',
             'interest',
             'internal',
             'intolerable',
             'irregular',
             'irritability',
             'irritable',
             'irritated',
             'irritation',
             'itcg',
             'itch',
             'itchiness',
             'itching',
             'itchy',
             'jaundice',
             'joint',
             'kidney',
             'knee',
             'lack',
             'large',
             'larger',
             'last',
             'le',
             'left',
             'leg',
             'lesion',
             'let',
             'lethargy',
             'leukemia',
             'level',
             'like',
             'limb',
             'lip',
             'liquid',
             'little',
             'liver',
             'long',
             'longer',
             'look',
             'looking',
             'loosing',
             'losing',
             'loss',
             'lost',
             'lot',
             'low',
             'lymph',
             'marital',
             'matter',
             'menstrual',
             'menstruation',
             'micturition',
             'mind',
             'mood',
             'mouth',
             'move',
             'movement',
             'moving',
             'much',
             'mucous',
             'mucus',
             'mumbling',
             'muscle',
             'nail',
             'nasal',
             'nausea',
             'nauseous',
             'neck',
             'need',
             'needle',
             'nervous',
             'no',
             'node',
             'normal',
             'normally',
             'nose',
             'not',
             'obese',
             'often',
             'ofurine',
             'one',
             'ooze',
             'oozed',
             'oozy',
             'outburst',
             'overall',
             'overload',
             'pain',
             'painful',
             'paining',
             'palpitating',
             'palpitation',
             'panic',
             'parent',
             'part',
             'partial',
             'partially',
             'patch',
             'patched',
             'patchy',
             'pee',
             'peeing',
             'peeling',
             'pemphigus',
             'perform',
             'period',
             'phlegm',
             'physical',
             'pigmentation',
             'pimple',
             'pit',
             'pitting',
             'plaque',
             'plasma',
             'platelet',
             'pleasure',
             'poo',
             'poop',
             'pooped',
             'pooping',
             'possibility',
             'pounding',
             'pressure',
             'problem',
             'prolonged',
             'prominent',
             'pronounce',
             'psoriasis',
             'pu',
             'puffy',
             'puked',
             'puking',
             'pumping',
             'racing',
             'rapidly',
             'rash',
             'rashing',
             'rate',
             'rbc',
             'reach',
             'really',
             'receiving',
             'recent',
             'red',
             'redness',
             'reflux',
             'region',
             'relax',
             'rest',
             'restlessness',
             'right',
             'risk',
             'running',
             'runny',
             'rust',
             'rusty',
             'sadness',
             'scab',
             'scratchiness',
             'scratchy',
             'scurring',
             'secretion',
             'see',
             'sensation',
             'sense',
             'sensorium',
             'severe',
             'sex',
             'shady',
             'shake',
             'sharp',
             'shiver',
             'shivering',
             'short',
             'shortness',
             'show',
             'showing',
             'sick',
             'side',
             'significantly',
             'silver',
             'sinus',
             'sitiing',
             'sitting',
             'skin',
             'skinnier',
             'skipped',
             'sleepy',
             'slow',
             'sluggy',
             'slurred',
             'small',
             'smaller',
             'smell',
             'sneeze',
             'sneezing',
             'sore',
             'speaking',
             'speech',
             'spinning',
             'sport',
             'spot',
             'spotted',
             'spotting',
             'sputum',
             'stamen',
             'stand',
             'stiff',
             'stiffly',
             'stiffness',
             'stink',
             'stomach',
             'stool',
             'stranger',
             'strong',
             'stuck',
             'stuffy',
             'sudden',
             'sugar',
             'sunburn',
             'sunken',
             'sweat',
             'sweating',
             'swelling',
             'swing',
             'swolen',
             'swollen',
             'tear',
             'tearfulness',
             'tearing',
             'teary',
             'teeth',
             'temperature',
             'tenderness',
             'tension',
             'thicker',
             'think',
             'thinner',
             'thirst',
             'thirsty',
             'threw',
             'thristy',
             'throat',
             'throbbing',
             'throwing',
             'thyroid',
             'tightness',
             'time',
             'tingling',
             'tingly',
             'tired',
             'tiredness',
             'toilet',
             'tongue',
             'tonsil',
             'touch',
             'toxic',
             'toxicity',
             'transfusion',
             'trauma',
             'tremble',
             'trembling',
             'turned',
             'typhos',
             'ulcer',
             'unable',
             'understand',
             'uneven',
             'unexplained',
             'unintentionally',
             'unsteadiness',
             'unsteady',
             'unsterile',
             'unsterilising',
             'unwell',
             'upper',
             'upset',
             'urge',
             'urgency',
             'urgent',
             'urinate',
             'urination',
             'urine',
             'use',
             'vein',
             'vertigo',
             'vessel',
             'viral',
             'visible',
             'vision',
             'visual',
             'vitamin',
             'voice',
             'volume',
             'vomit',
             'vomited',
             'vomiting',
             'walk',
             'walking',
             'want',
             'water',
             'watering',
             'watery',
             'weak',
             'weaker',
             'weakness',
             'weight',
             'weird',
             'well',
             'wet',
             'wheezing',
             'white',
             'winded',
             'without',
             'woozy',
             'word',
             'worried',
             'worsening',
             'wrist',
             'yellow',
             'yellowing',
             'yellowish']
    
    finalString = []
    finalString2 = []
    finalString3 = []
    tags = ['abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feets', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dischromic_patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'foul_smell_ofurine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs', 'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin']
        
    testWithoutAnd = test.split(" and ")
    
    for item in testWithoutAnd:
      finalString.insert(len(finalString), item)
        
    for ss in finalString: 
      testWithoutOr = ss.split(" or ")
      for newItem in testWithoutOr:
        finalString2.append(newItem)
        
    for ss in finalString2: 
      testWithoutPunc = re.split("[" + string.punctuation + "]+", ss)
      for newItem in testWithoutPunc:
        finalString3.append(newItem.strip())
        
    for sentence in finalString3: 
        sentence = nltk.word_tokenize(sentence)
    #print(sentence)
    filtered_sentence=[]
    for words in sentence:
      if words not in stop_words:
        filtered_sentence.append(words)
    to_predict = [[0]*601 for _ in range(1)]

    for words in filtered_sentence:
      if(words in all_words):
        to_predict[0][all_words.index(words)]=1
        
    tag = my_model.predict(to_predict)[0]
    
    return tags[tag]

@app.route('/')
def index():
    data = []
    user_symptoms.clear()
    file = open("static/assets/files/ds_symptoms.txt", "r")
    all_symptoms = file.readlines()
    for s in all_symptoms:
        data.append(s.replace("'", "").replace("_", " ").replace(",\n", ""))
    data = json.dumps(data)

    return render_template('index.html', data=data)


@app.route('/symptom', methods=['GET', 'POST'])
def predict_symptom():
    print("Request json:", request.json)
    sentence = request.json['sentence']
    if sentence.replace(".", "").replace("!","").lower().strip() == "done":

        if not user_symptoms:
            response_sentence = random.choice(
                ["I can't know what disease you may have if you don't enter any symptoms :)",
                "Meddy can't know the disease if there are no symptoms...",
                "You first have to enter some symptoms!"])
        else:
            x_test = []
            
            for each in symptoms_list: 
                if each in user_symptoms:
                    x_test.append(1)
                else: 
                    x_test.append(0)

            x_test = np.asarray(x_test)            
            disease = prediction_model.predict(x_test.reshape(1,-1))[0]
            print(disease)

            description = diseases_description.loc[diseases_description['Disease'] == disease.strip(" ").lower(), 'Description'].iloc[0]
            precaution = disease_precaution[disease_precaution['Disease'] == disease.strip(" ").lower()]
            precautions = 'Precautions: ' + precaution.Precaution_1.iloc[0] + ", " + precaution.Precaution_2.iloc[0] + ", " + precaution.Precaution_3.iloc[0] + ", " + precaution.Precaution_4.iloc[0]
            response_sentence = "It looks to me like you have " + disease + ". <br><br> <i>Description: " + description + "</i>" + "<br><br><b>"+ precautions + "</b>"
            
            severity = []

            for each in user_symptoms: 
                severity.append(symptom_severity.loc[symptom_severity['Symptom'] == each.lower().strip(" ").replace(" ", ""), 'weight'].iloc[0])
                
            if np.mean(severity) > 4 or np.max(severity) > 5:
                response_sentence = response_sentence + "<br><br>Considering your symptoms are severe, and Meddy isn't a real doctor, you should consider talking to one. :)"

            user_symptoms.clear()
            severity.clear()
 
    else:
        symptom = get_symptom(sentence)
        
        response_sentence = f"Hmm, you probably have " + symptom + "."
        user_symptoms.add(symptom)
    

        print("User symptoms:", user_symptoms)

    return jsonify(response_sentence.replace("_", " "))