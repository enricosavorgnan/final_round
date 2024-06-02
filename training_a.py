from openai import OpenAI
import speech_recognition as sr

client = OpenAI(api_key='YOUR_OPENAI_API_KEY_GOES_HERE')


def get_target(my_query):
    content = ("""You are an helpful home-assistant. Your response will be used to make some actions in the home, so you need to be precise.
                You will receive a sentence from me. You have to return a string made like this: "x_1, x_2, x_3, x_4, x_5". 
                Please make sure to always check the structure of your response, that has to be the string I just gave you.
                Values x_1, x_2, x_3, x_4, x_5 are numbers. 
                The first one is temperature in my home, measured in Celsius degrees. Values go from 15°C to 30°C.
                The second one is pressure, measured in atm. Values go from 0.9atm to 1.2atm.
                The third one is CO2 concentration, measured in ppm. Values go from 400ppm to 700ppm.
                The fourth one is brightness, ranged 1 to 10 in integral numbers.
                The last is humidity, a percentage, usually going from 10% to 60%.   
                
                Make sure to answer ONLY the string "x_1, x_2, x_3, x_4, x_5". 
                Double check the answer before answering.
               """)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": my_query}
        ],
        temperature=0.9
    )

    result = response.choices[0].message.content
    result = [float(x) for x in result.split(',')]
    print(result)
    return result


def get_transcription(my_audio):
    with open("speech.wav", "wb") as f:
        print(1)
        f.write(my_audio.get_wav_data())
        speech = open("speech.wav", "rb")
        print(2)
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=speech,
            response_format="text"
        )
        return transcription


# Recognize the user prompt
recognizer = sr.Recognizer()
microphone = sr.Microphone()
with microphone as source:
    # listen to the user prompt
    print("Listening to the user prompt...")
    recognizer.adjust_for_ambient_noise(source, duration=0.5)  # set di default
    my_audio = recognizer.listen(source)
    print("Listening done!")

    # Get user prompt
    try:
        # transcription of the prompt
        print("Transcribing the user prompt...")
        user_input = get_transcription(my_audio)
        print("User input:", user_input)
    except sr.UnknownValueError:
        print("No transcription available")  # No transcription
    except sr.RequestError as e:
        print(f"Error in the request: {e}")

    # Get target data
    try:
        print("Getting target data...")
        target = get_target(user_input)
        print("Target data:", target)
    except Exception as e:
        print(f"Error in getting target data: {e}")


