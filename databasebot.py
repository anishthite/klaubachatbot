#Simple Database chatbot
#Anish Thite

#Flow:
# 1. Hi/Hello/Hey --> Hello! Please state the room number where you would like to go? (GOTO 2)
# 2. [Some response] --> For clarification, you want to go [number]? Answer yes (GOTO 3) or no (GOTO 4).
# 
from chatterbot import ChatBot
chatbot = ChatBot("Klauba")

#train chatterbot
from chatterbot.trainers import ListTrainer

conversation= [
"Hello!",
"Hi, I am Kaluba, please state the room you want to go to",
"I want to go to Room 2337",
"Thank you! I will take you to room 2337"
]
trainer = ListTrainer(chatbot)
trainer.train(conversation)

while True:
    try:
        bot_input = chatbot.get_response(input("Enter input:"))
        print(bot_input)

    except(KeyboardInterrupt, EOFError, SystemExit):
        break