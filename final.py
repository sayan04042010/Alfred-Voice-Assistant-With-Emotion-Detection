import threading
import pyttsx3
import datetime
import os
import speech_recognition as sr
import cv2
import time
from collections import Counter
from ultralytics import YOLO
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# --- Shared State ---
emotion_history = []
emotion_lock = threading.Lock()

def get_current_emotion():
    with emotion_lock:
        current_time = time.time()
        global emotion_history
        # Keep only emotions from the last 3 seconds
        emotion_history = [e for e in emotion_history if current_time - e[0] <= 3.0]
        if emotion_history:
            recent_emotions = [e[1] for e in emotion_history]
            return Counter(recent_emotions).most_common(1)[0][0]
        return "neutral"

# --- Voice Assistant Setup ---
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

template="""
You are Alfred, a helpful AI voice assistant.
Answer the user's question concisely. Your reply MUST be short (1 to 2 sentences maximum).

The user is currently feeling: {emotion}.
Please tailor the tone and empathy of your concise response based on this emotion.

Here is the conversation history: {context}

Question={question}

Answer: 
"""
# Keep max tokens low to drastically improve response speed
model = OllamaLLM(model="llama3", num_predict=50)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def speak(audio):
    emotion = get_current_emotion()
        
    # Tone changes based on emotion
    try:
        if emotion.lower() in ["sad", "sadness", "fear", "surprise"]:
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 0.8)
        elif emotion.lower() in ["happy", "happiness", "joy", "excited"]:
            engine.setProperty('rate', 220)
            engine.setProperty('volume', 1.0)
        elif emotion.lower() in ["angry", "anger", "disgust"]:
            engine.setProperty('rate', 190)
            engine.setProperty('volume', 0.9)
        else:
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
    except Exception:
        pass

    engine.say(audio)
    engine.runAndWait()


def wishMe():
    print("\n" + "="*50)
    print("...ALFRED ONLINE...")
    print("="*50 + "\n")
    speak("ALFRED ONLINE!")
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning Sir!!")
    elif hour >= 12 and hour < 18:
        speak("Good afternoon Sir!!")
    else:
        speak("Good Evening Sir!!")
    speak("How can I help You?")


def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
         print("\n[ALFRED is Listening...]")
         r.pause_threshold = 1.0  # Reduced from 3 to 1 to respond much faster once you stop speaking
         audio = r.listen(source)
    try:
         print("[ALFRED is Recognising...]")
         # Setting energy threshold and making it slightly faster to parse 
         # audio on the fly
         query = r.recognize_google(audio, language='en-in')
    except Exception as e:
         print("[ALFRED Error: Could not understand]")
         speak("Sir!!!  Can you repeat that again.... please!")
         return None
    return query


def chatbot(user_input, context):
     emotion = get_current_emotion()
     result = chain.invoke({"context": context, "question": user_input, "emotion": emotion})
     
     # Print the chat cleanly
     print("\n" + "-"*50)
     print(f"USER: {user_input}")
     print(f"ALFRED [Detected Emotion: {emotion}]:\n{result}") 
     print("-"*50 + "\n")
     
     speak(result)
     context = (context or "") + f"\nUser::{user_input}\nAI:{result}"  
     return context



def voice_assistant_thread():
    speak("greetings!")
    wishMe()
    context = ""
        
    while True:
        query = takecommand()
        if query:
            query = query.lower()
            if 'exit' in query or 'shutdown' in query or 'terminate' in query:
                  speak("Thank You Sir! Have A Nice day! Shutting Down initialized....")
                  speak("Going offline!")
                  print("\n...ALFRED IS OFFLINE NOW...\n")
                  os._exit(0)  
            else:
                  context = chatbot(query, context)


def emotion_detection_thread():
    global emotion_history
    
    # 1. Load your custom model
    model = YOLO("last.pt") 

    # 2. Initialize the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Run Inference - disabling YOLO's verbose logging to keep terminal clean
        results = model(frame, conf=0.5, stream=True, verbose=False)
        annotated_frame = frame

        for r in results:
            annotated_frame = r.plot() 
            
            # Access specific emotions detected
            for box in r.boxes:
                class_id = int(box.cls[0])
                emotion_name = r.names[class_id]
                
                with emotion_lock:
                    emotion_history.append((time.time(), emotion_name))
                
                # We intentionally don't print the emotion here anymore 
                # to prevent chat history from getting buried!
                break

        # 4. Display the results
        cv2.imshow("YOLOv11 Emotion Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os._exit(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    # Start the voice assistant loop in a daemon thread
    va_thread = threading.Thread(target=voice_assistant_thread, daemon=True)
    va_thread.start()
    
    # Run the emotion detection heavily reliant on OpenCV UI in the main thread
    emotion_detection_thread()
