# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:00:14 2025

@author: soura
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name="google/flan-t5-base"
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)
 
def chat_with_bot():
    while True:
        #to get input from user
        input_text=input("You: ")
        
        #exit condiotion
        if input_text.lower() in ["quit","exit","bye"]:
            print("Chatbot: Goodbye!")
            break
        
        #tokenize input and generate response
        inputs=tokenizer.encode(input_text,return_tensors="pt")
        outputs=model.generate(inputs,max_new_tokens=150)
        response=tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        print("Chatbot: ", response)
chat_with_bot()