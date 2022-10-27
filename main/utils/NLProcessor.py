from collections import defaultdict
import re,nltk,os,pickle,emoji,pandas as pd
#For directly creating images using the backend:
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from io import StringIO #for storing image files in memory
from collections import Counter #for counting emojis

class NLProcessor:
    def __init__(self,chatFile):
        lines = []
        #extract the contents:
        for chunk in chatFile.chunks():
            #chunks are binary, decode them into str,then split line-wise:
            lines.extend(chunk.decode().split('\n'))
        #The first line in chats is the default whatsapp "e2ee" rant; Remove it:
        lines.pop(0)

        #Create the dataframe:
        self._createDataframe(lines)
        #Classify Messages and store the plotted image:
        self.msgClassifiedGraph = self._classifyMessages()
        #Get emoji stats:
        self.labelledEmojiCount = self._getEmojiStats()
    
    #returns all our results/conclusions
    def get_analysis(self):
        return (self.msgClassifiedGraph,
                self.labelledEmojiCount,
                self.orderedSenders)

    #This function creates the globally accesible pandas dataframe
    def _createDataframe(self,lines):
        #stores date time and messages(values) tied to a sender(key):
        ppl = defaultdict(list)

        # Compile a regexp that segregates (for each line) the date-time from the rest of the text:
        reg = re.compile(r'^\d{1,2}/\d{1,2}/\d{1,2}, \d{1,2}:\d{1,2} - ')
        # We will initialise a pandas dataframe, columns values will be stored in these lists:
        df_sender,df_msg,df_date,df_time = [],[],[],[]
        for line in lines:
            matched = reg.match(line)
            if matched:
                #the index at which the message begins
                begin_idx = line[matched.end():].find(':')+1+matched.end()+1
                #Extract date and time:
                date,time= line[:matched.end()-3].split(", ")
                #Extract the sender:
                sender = line[matched.end():begin_idx-2]
                #extract the message:
                message = line[begin_idx:]
                #we will omit the message if it contains the text 'null' or <Media omitted>:
                if message !="null" and message!= "<Media omitted>":
                    #push these values to the dictionary:
                    ppl[sender].append({
                        "date":date,
                        "time":time,
                        "msg": [message.strip()]
                    })
                    #Also add the sender and msg to the df column lists:
                    df_sender.append(sender)
                    df_msg.append(message.strip())
                    df_date.append(date)
                    df_time.append(time)
            else:
                #we assume it is a multiline message, and append the message texts to the previous message:
                message = line.strip()
                if message:
                    ppl[sender][-1]["msg"].append(message)
                    #for the df column lists:
                    df_sender.append(df_sender[-1])
                    df_msg.append(message)
                    df_date.append(df_date[-1])
                    df_time.append(df_time[-1])
        #Our data:
        data = {
            "sender": df_sender,
            "msg": df_msg,
            "date": df_date,
            "time": df_time
        }
        #Our global dataframe:
        self.df = pd.DataFrame(data)
        #For frontend display purposes, keep a sequential list of senders:
        self.orderedSenders = [sender for sender in self.df['sender']]
        # convert date and time strings into the apropriate type:
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['time'] = pd.to_datetime(self.df['time']).dt.strftime('%H')
    
    #This function classifies messages(using a NaiveBayes classifier trained on nps_chat):
    def _classifyMessages(self):
        #We'll train a naive-bayes classifier on this sample corpus
        #Check if we already created the classifier:
        nb_classifier = None
        if os.path.exists('nb-msg-classifier.pickle'):
            f = open('nb-msg-classifier.pickle','rb')
            nb_classifier = pickle.load(f)
            f.close()
        else:
            #Training data for the classifier:
            posts = nltk.corpus.nps_chat.xml_posts()
            label_posts = [(extract_features(p.text),p.get('class')) for p in posts]
            #10% test size:
            test_size = int(len(label_posts)*0.1)
            train_set,test_set = label_posts[test_size:],label_posts[:test_size]
            nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
            #Save the model:
            f = open('nb-msg-classifier.pickle','wb')
            pickle.dump(nb_classifier,f)
            f.close()
        #essentially,take each word and tell the classfier it was present as a 'feature' in this post:
        def extract_features(post):
            features = {} # a dictionary storing all the words in this post
            for word in nltk.word_tokenize(post):
                features['contains({})'.format(word.lower())] = True
            return features
        #Classify each message from the dataframe, and add its predicted class to the dataframe: 
        df_msgclass = []
        for message in self.df['msg']:
            msg_class = nb_classifier.classify(extract_features(message))
            df_msgclass.append(msg_class)
        self.df['msgclass'] = df_msgclass
        #get the plotted graph of this data and return it:
        return self._plotMsgClassification()

    #Plot the classified messages as a countplot,store it in a bytes format:
    def _plotMsgClassification(self):
        import seaborn as sns, matplotlib.pyplot as plt
        fig = Figure(figsize=(10,7),dpi=100)
        axis = fig.subplots()
        sns.set(font_scale=2)
        #axis = plt.subplots(figsize=(10,7))[1]
        plot = sns.countplot(data=self.df,y="msgclass",hue="sender",ax=axis)
        plot.set(xlabel='Frequency', ylabel='Types of Messages',title='Message Classification')
        displayImage = StringIO()
        fig.savefig(displayImage,format='svg')
        displayImage.seek(0)
        data=displayImage.getvalue() #return the binary as a string
        return data
    
    #Get Emoji stats per sender:
    def _getEmojiStats(self):
        #to store most commonly encountered emojis per sender:
        commonCount = []
        for label,senderGroupedDf in self.df.groupby('sender'):
            df_msg = senderGroupedDf.reset_index()['msg']
            emojis=''
            for msg in df_msg:
                emojis+=''.join(c for c in msg if emoji.is_emoji(c))
            commonCount.append([label,Counter(emojis).most_common()[:10]])
        return commonCount
'''         
def extract_emojis(msg):
  return ''.join(c for c in msg if emoji.is_emoji(c))
    for key, val in ppl.items():
        emojis=extract_emojis(str(ppl[key]))
        count = Counter(emojis).most_common()[:10]
    print("{}'s emojis:\n {} \n".format(key, emojis))
    print("Most common: {}\n\n".format(count))
'''
