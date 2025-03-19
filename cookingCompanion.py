import os
import openai
import gradio as gr
import pygame
import time
import speech_recognition as sr

openai_client = openai.Client(api_key=os.environ['API_KEY'])

SYSTEM_PROMPT = """
You are a knowledgeable and friendly cooking expert with years of experience helping people create delicious and customized meals.
You are here solely to discuss food, recipes, and cooking techniques.

Please understand what kind of food the user wants by having a conversation with them.
Your messages should be short and concise as if you were exchanging text messages.
Ask questions and suggest recipes based on their preferences.
Describe the recipes to the user.
Guide the users to make a choice.
Provide the recipe and directions in a detailed and highly structured format.
The user may also ask you for instructions as they cook, just respond and guide them through the process.
Keep your guidance clear and concise as if you were there with them in the kitchen.
You can also suggest alternative ingredients or cooking methods.
Remember to be friendly and engaging.

When you give the recipe, please provide the following additional information:
- Difficulty level
- Cooking time
- Cooking steps
- Serving size
- Nutritional information
- Any additional tips or information

Follow these guidelines gently:
1. Use ingredients wisely - Transform leftovers and food scraps into new meals to reduce waste and enhance flavor.
2. Trust intuition over rigid recipes - Encourage users to adjust flavors, textures, and cooking methods based on their senses.
3. Basic techniques matter - Teach essential cooking methods (boiling, roasting, seasoning) to maximize ingredient potential.
4. Favor real, whole ingredients - Guide users toward naturally flavorful, high-quality foods instead of processed substitutes.
5. Let taste guide nutrition - Encourage listening to natural cravings for real flavors, which often signal what the body needs.
6. Avoid artificial flavor traps - Help users recognize and minimize processed foods that manipulate taste without real nourishment.
7. Customize for comfort & mood - Suggest adaptations based on how someone feels, whether they want indulgence, simplicity, or nostalgia.
8. Encourage experimentation - Remind users that cooking alone is the perfect time to tweak flavors, techniques, or ingredients without judgment.
9. Make solo cooking enjoyable - Provide simple, satisfying recipes that empower users to embrace cooking as an act of self-care.

THIS IS YOUR CONVERSATION SO FAR:
{prev_conversation}
"""

COOKING_PROMPT = """
You are a personal cooking guide, helping someone prepare a meal in real-time. 
Your job is to provide clear and concise step-by-step cooking instructions.
Keep the conversation friendly and engaging. 

If the user asks about anything outside of cooking, do not answer. Instead, gently redirect them by saying: 
"I'm here to help with cooking! What step do you need help with?" 

Keep your responses short (1-3 sentences) and focus only on cooking-related topics.
Never ask questions back to the user.

THIS IS YOUR CONVERSATION SO FAR:
{prev_conversation}
"""

# Audio Setup
pygame.mixer.init()

def format_chat_history(history):
    """Format chat history as a string for the prompt."""
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

def recipe_mode(message, history):
    """Generate chatbot response based on conversation history."""
    RECIPE_PROMPT = SYSTEM_PROMPT.format(prev_conversation=format_chat_history(history))
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": RECIPE_PROMPT}, {"role": "user", "content": message}]
    )
    
    bot_message = response.choices[0].message.content
    
    if len(bot_message.split()) > 70:  # If response is a recipe
        lines = bot_message.split("\n", 1)
        title = lines[0]
        content = lines[1]
        return gr.ChatMessage(role="assistant", content=content, metadata={"title": title, "status": "done"})
    else:
        return bot_message

def record_audio(filename="input.wav", samplerate=44100):
    """Record audio from microphone and save to file."""
    recognizer = sr.Recognizer()
    
    with sr.Microphone(sample_rate=samplerate) as source:       
        recognizer.adjust_for_ambient_noise(source)
        recognizer.pause_threshold = 1.5
        recognizer.non_speaking_duration = 0.5 
        
        audio = recognizer.listen(source, timeout=None)
        
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())

def transcribe_audio(filename="input.wav"):
    """Transcribe recorded audio using Whisper API."""
    with open(filename, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcription.text.strip()

def generate_speech(text, filename="response.mp3"):
    """Convert text response to speech and play it."""
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.mixer.init()
    
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except PermissionError:
            time.sleep(0.5)
            os.remove(filename)
    
    response = openai_client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    
    with open(filename, "wb") as f:
        f.write(response.content)
    
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def cooking_mode(history):
    """Demo Mode: Click to give verbal input and recieve verbal output."""
    while True:
        record_audio()
        user_message = transcribe_audio()
        history.append({"role": "user", "content": user_message})
        
        yield history
        
        RECIPE_PROMPT = COOKING_PROMPT.format(prev_conversation=format_chat_history(history))
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": RECIPE_PROMPT}, {"role": "user", "content": user_message}]
        )
        
        response_message = response.choices[0].message.content
        history.append({"role": "assistant", "content": response_message})
        
        yield history
        
        generate_speech(response_message)

# Interface
with gr.Blocks(title="Cooking Companion 👨‍🍳", css="""
    footer {visibility: hidden}; 
    }
""", theme=gr.themes.Citrus()) as demo:
    chatbot = gr.ChatInterface(
        fn=recipe_mode,
        type="messages",
        autofocus=False,
        title="Cooking Companion 👨‍🍳"
    )
    
    start_button = gr.Button("Start Cooking")
    start_button.click(cooking_mode, chatbot.chatbot, chatbot.chatbot)

# Launch the interface
demo.launch(share=False)