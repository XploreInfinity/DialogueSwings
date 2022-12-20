# DialogueSwings

*version 0.01(beta)*

DialogueSwings uses the power of NLP to extract the hidden stats behind your Whatsapp chats

*Usage:*

- Create a file `.secrets` to store your django secret key
- Install all python packages mentioned in `requirements.txt` : `pip install -r requirements.txt`
- Run `python3 -m manage runserver`
- Upload the exported .txt (media omitted) whatsapp chat file
- Get the results

*Features:*

- Supports both two-people and group chats
- Generated graphs directly rendered on the frontend
- Supports exported chats of both 12 and 24 hour formats

*Statistics displayed:*

- Type of messages exchanged by senders (message classification)
- Top 10 emojis and their counts: for every sender
- Mean sentiment throughout the chat, over time : for every sender
- Number of messages collectively exchanged in the chat for each hour of a 24-hour day
