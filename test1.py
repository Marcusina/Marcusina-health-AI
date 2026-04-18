
'''
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Marcusina AI is running NLP pipelines.")
print([(ent.text, ent.label_) for ent in doc.ents])

'''
number =0
for i in range (1,4):
    number+=i
print(number)